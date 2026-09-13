"""D4RT 单元测试与冒烟测试

运行: .venv/bin/python -m pytest tests/ -v
(未安装 pytest 时: .venv/bin/python tests/test_d4rt.py 直接执行全部测试)

覆盖:
  - Umeyama 数值正确性 (随机 R, t 往返 + 尺度恢复)
  - 模型前向/反向 shape (13 维输出头) 与梯度流通
  - 各 ViT 变体可构造; ViT-g 参数规模 ≈ 编码器 1B / 解码器 144M (meta device)
  - 查询构造器 shape (Tab.1 各任务)
  - 合成数据 GT 自洽性 (p_2d ≈ 投影(p_3d); 背景位移为 0)
  - 密集跟踪 Alg.1 冒烟 (覆盖率 ∈ (0, 1])
"""

import sys
from pathlib import Path

# 直接以 python tests/test_d4rt.py 运行时, 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.synthetic import SyntheticDataset
from models import D4RTModel, D4RTLoss, VARIANTS
from tracking import run_dense_tracking
from utils.camera import (sim3_alignment, stitch_chunks, umeyama,
                          weighted_umeyama, dynamic_aware_umeyama,
                          estimate_camera_intrinsics)
from utils.query_factory import D4RTQueryFactory

torch.manual_seed(0)


def _random_rot():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    return q


# ---------------------------------------------------------------------------
# Umeyama
# ---------------------------------------------------------------------------

def test_umeyama_roundtrip():
    """随机刚性变换往返: umeyama(src, tgt) 应恢复 src→tgt 的 R, t"""
    for _ in range(5):
        R_true = _random_rot()
        t_true = torch.randn(3) * 2
        src = torch.randn(200, 3) * 3
        tgt = src @ R_true.T + t_true  # 行向量: tgt = R @ src + t
        R_est, t_est = umeyama(src, tgt)
        assert torch.allclose(R_est, R_true, atol=1e-4), "旋转恢复失败"
        assert torch.allclose(t_est.squeeze(1), t_true, atol=1e-4), "平移恢复失败"


def test_umeyama_reflection():
    """反射变换应被修正 (det R = +1)"""
    R_true = torch.diag(torch.tensor([1.0, 1.0, -1.0]))  # 反射
    src = torch.randn(100, 3)
    tgt = src @ R_true.T
    R_est, _ = umeyama(src, tgt)
    assert torch.det(R_est) > 0
    # 反射不能作为相机旋转，故只验证返回 proper rotation。


def test_sim3_reflection_correction_uses_signed_singular_value():
    src = torch.randn(200, 3)
    tgt = src @ torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    scale, R, t = umeyama(src, tgt, with_scale=True)
    assert torch.det(R) > 0
    assert scale <= 1.0 + 1e-6
    assert torch.isfinite(t).all()


def test_umeyama_scale():
    """with_scale=True 应恢复相似比 s"""
    s_true = 2.3
    R_true = _random_rot()
    t_true = torch.randn(3)
    src = torch.randn(100, 3) * 2
    tgt = s_true * src @ R_true.T + t_true
    s_est, R_est, t_est = umeyama(src, tgt, with_scale=True)
    assert abs(s_est - s_true) < 1e-3
    assert torch.allclose(R_est, R_true, atol=1e-4)
    assert torch.allclose(t_est.squeeze(1), t_true, atol=1e-3)


def test_rigid_umeyama_never_applies_estimated_scale():
    """刚体模式的平移不能乘拟合出的相似尺度。"""
    src = torch.randn(100, 3) + torch.tensor([4.0, -2.0, 3.0])
    tgt = 1.8 * src + torch.tensor([0.5, -1.0, 2.0])
    R, t = umeyama(src, tgt, with_scale=False)
    assert torch.allclose(t.squeeze(), tgt.mean(0) - R @ src.mean(0), atol=1e-5)


def test_weighted_umeyama_rejects_low_weight_outliers():
    """动态感知配准的核心性质: 低权重动态离群点不应污染刚体变换。"""
    s = 1.0
    R_true = _random_rot()
    t_true = torch.randn(3)
    n, n_out = 120, 24
    src = torch.randn(n, 3) * 2
    tgt = src @ R_true.T + t_true
    tgt[-n_out:] += torch.randn(n_out, 3) * 20.0
    weights = torch.ones(n)
    weights[-n_out:] = 1e-4
    R_est, t_est = weighted_umeyama(src, tgt, weights)
    assert torch.allclose(R_est, R_true, atol=2e-3)
    assert torch.allclose(t_est.squeeze(1), t_true, atol=2e-2)


def test_dynamic_aware_umeyama_uses_motion_signal():
    """高运动点即使置信度相同，也不应主导动态感知配准。"""
    R_true = _random_rot()
    t_true = torch.randn(3)
    n, n_dyn = 100, 25
    src = torch.randn(n, 3)
    tgt = src @ R_true.T + t_true
    tgt[-n_dyn:] += torch.randn(n_dyn, 3) * 10.0
    motion = torch.zeros(n)
    motion[-n_dyn:] = 10.0
    R_est, t_est, diag = dynamic_aware_umeyama(
        src, tgt, confidence=torch.ones(n), motion=motion, robust_iters=2)
    assert torch.allclose(R_est, R_true, atol=5e-2)
    assert torch.allclose(t_est.squeeze(1), t_true, atol=1e-1)
    assert diag["weights"][:-n_dyn].mean() > diag["weights"][-n_dyn:].mean()


def test_sim3_alignment_confidence_top85():
    """附录 B: 置信度筛选 — 低置信离群点应被 top-85% 剔除"""
    s_true, R_true, t_true = 1.5, _random_rot(), torch.randn(3) * 2
    n, n_out = 100, 10  # 内点 90 个 > top-85% 的 85 个, 保证筛选后全为内点
    src = torch.randn(n, 3) * 3
    tgt = s_true * src @ R_true.T + t_true
    # 离群点: 一致指向 50× 尺度 (无噪声), 无置信度时必然把尺度估计拉偏
    tgt[-n_out:] = 50 * (src[-n_out:] @ R_true.T) + t_true
    conf = torch.ones(n)
    conf[-n_out:] = 1e-3
    s_est, R_est, t_est = sim3_alignment(src, tgt, conf_src=conf, conf_tgt=conf,
                                         top_frac=0.85)
    assert abs(s_est - s_true) < 1e-2
    assert torch.allclose(R_est, R_true, atol=1e-3)
    assert torch.allclose(t_est.squeeze(1), t_true, atol=1e-1)
    # 无置信度时受离群点污染, 尺度误差应显著更大
    s_bad, _, _ = sim3_alignment(src, tgt)
    assert abs(s_bad - s_true) > 0.05


def test_stitch_chunks():
    """附录 B chunk 拼接: 恢复 chunk b → a 的 Sim(3) 变换"""
    s_true, R_true, t_true = 1.2, _random_rot(), torch.randn(3)
    pts_b = torch.randn(200, 3) * 2
    pts_a = s_true * pts_b @ R_true.T + t_true
    conf = torch.rand(200)
    s_est, R_est, t_est = stitch_chunks(pts_a, pts_b, conf, conf)
    assert abs(s_est - s_true) < 1e-3
    assert torch.allclose(R_est, R_true, atol=1e-4)
    assert torch.allclose(t_est.squeeze(1), t_true, atol=1e-3)


def test_intrinsics_unwarps_crop_coordinates():
    """裁剪后视频的查询坐标不能直接代入原图主点公式。"""
    crop, source_size, f = (12, 8, 64, 48), (96, 96), 0.9

    class AnalyticDecoder:
        def __call__(self, uv, latent):
            x0, y0, cw, ch = crop
            h, w = source_size
            u = (x0 + uv[:, 0] * cw) / w
            v = (y0 + uv[:, 1] * ch) / h
            p = torch.stack(((u - 0.5) / f, (v - 0.5) / f,
                             torch.ones_like(u)), dim=-1)
            return {"p_3d": p}

    class AnalyticModel:
        decoder = AnalyticDecoder()

        def eval(self):
            return self

        def encode(self, video, aspect):
            return None

        def query_embedder(self, video, uv, *times):
            return uv

    video = torch.zeros(1, 4, 3, 64, 64)
    fx, fy, fx_px, fy_px = estimate_camera_intrinsics(
        AnalyticModel(), video, 0, crop=crop, source_size=source_size)
    assert abs(fx - f) < 1e-5 and abs(fy - f) < 1e-5
    assert abs(fx_px - f * 96) < 1e-3 and abs(fy_px - f * 96) < 1e-3


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def test_model_forward_backward():
    """tiny 模型前向: 13 维输出头 shape 正确, 反向梯度流通"""
    model = D4RTModel(variant="tiny", image_size=64, num_frames=4, max_frames=4)
    video = torch.rand(1, 4, 3, 64, 64)
    N = 32
    q = {"uv": torch.rand(N, 2),
         "t_src": torch.randint(0, 4, (N,)),
         "t_tgt": torch.randint(0, 4, (N,)),
         "t_cam": torch.randint(0, 4, (N,))}
    preds = model(video, q)
    assert preds["p_3d"].shape == (N, 3)
    assert preds["p_2d"].shape == (N, 2)
    assert preds["vis"].shape == (N, 1)
    assert preds["disp"].shape == (N, 3)
    assert preds["normal"].shape == (N, 3)
    assert preds["conf"].shape == (N, 1)
    assert torch.allclose(preds["normal"].norm(dim=-1), torch.ones(N), atol=1e-5)
    assert (preds["conf"] >= 1e-4).all() and (preds["conf"] <= 1.0).all()

    # 反向: 以解析 GT 计算损失 (跳过数据集, 构造合理 targets)
    targets = {"p_3d": torch.randn(N, 3), "p_2d": torch.rand(N, 2),
               "vis": (torch.rand(N, 1) > 0.5).float(),
               "disp": torch.randn(N, 3), "normal": torch.randn(N, 3)}
    targets["normal"] = targets["normal"] / targets["normal"].norm(dim=-1, keepdim=True)
    loss, _ = D4RTLoss()(preds, targets)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum() for p in model.parameters()
                    if p.grad is not None and p.requires_grad)
    assert grad_norm > 0 and torch.isfinite(grad_norm)


def test_loss_all_invalid_queries_is_finite():
    n = 4
    p = torch.randn(n, 3, requires_grad=True)
    predictions = {"p_3d": p, "p_2d": torch.zeros(n, 2),
                   "vis": torch.ones(n, 1) * 0.5,
                   "disp": torch.zeros(n, 3),
                   "normal": torch.nn.functional.normalize(torch.ones(n, 3)),
                   "conf": torch.ones(n, 1) * 0.5}
    targets = {key: value.detach().clone() for key, value in predictions.items()
               if key != "conf"}
    loss, _ = D4RTLoss()(predictions, targets, mask=torch.zeros(n, 1, dtype=torch.bool))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_variants_construct_and_counts():
    """全部变体可在 meta device 构造; ViT-g 规模 ≈ 论文的 1B / 144M"""
    for name in VARIANTS:
        m = D4RTModel(variant=name, image_size=256, num_frames=48,
                      max_frames=48).to("meta")
        counts = m.param_counts()
        assert counts["encoder"] > 0 and counts["decoder+query"] > 0, name

    g = D4RTModel(variant="vit-g", image_size=256, num_frames=48,
                  max_frames=48).to("meta").param_counts()
    # 论文 §4: "the ViT encoder and ... cross-attention decoder contain 1 B
    # and 144 M parameters, respectively"
    assert 0.9e9 <= g["encoder"] <= 1.2e9, f"编码器 {g['encoder']/1e6:.0f}M 偏离 1B"
    assert 130e6 <= g["decoder+query"] <= 160e6, f"解码器 {g['decoder+query']/1e6:.0f}M 偏离 144M"


def test_decoder_no_self_attention():
    """解码器块内只有一个 MultiheadAttention (cross), 查询间零交互 (论文 §2.2)"""
    from models.d4rt import CrossAttnBlock, D4RTDecoder
    blk = CrossAttnBlock(32, 4)
    n_attn = sum(isinstance(m, torch.nn.MultiheadAttention) for m in blk.modules())
    assert n_attn == 1
    dec = D4RTDecoder(dim=32, nhead=4, num_layers=2, enc_dim=32)
    assert len(dec.blocks) == 2


def test_query_patch_center_matches_video_pixel():
    from models.d4rt import QueryPatchEmbed
    video = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 1, 3, 4, 4)
    uv = torch.tensor([[(1 + 0.5) / 4, (2 + 0.5) / 4]])
    patch = QueryPatchEmbed(dim=8, patch_size=1).extract(video, torch.tensor([0]), uv)
    assert torch.allclose(patch[0], video[0, 0, :, 2, 1], atol=1e-6)


def test_global_attention_updates_video_tokens():
    """全局注意力的残差必须进入输出，不能被旧 token 覆盖。"""
    from models.d4rt import EncoderBlock
    block = EncoderBlock(16, 4, local=False)
    with torch.no_grad():
        for layer in block.mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
    x = torch.randn(1, 5, 16)
    assert not torch.allclose(block(x, (1, 4)), x)


def test_aspect_ratio_reaches_video_tokens():
    model = D4RTModel(variant="tiny", image_size=32, num_frames=4, max_frames=4).eval()
    video = torch.rand(1, 4, 3, 32, 32)
    with torch.no_grad():
        a = model.encode(video, 0.75)[:, 1:]
        b = model.encode(video, 1.25)[:, 1:]
    assert (a - b).abs().max() > 1e-6


def test_repeated_query_is_deterministic_in_eval_mode():
    model = D4RTModel(variant="tiny", image_size=32, num_frames=4, max_frames=4).eval()
    video = torch.rand(1, 4, 3, 32, 32)
    queries = {"uv": torch.tensor([[0.25, 0.75]]),
               "t_src": torch.tensor([1]), "t_tgt": torch.tensor([2]),
               "t_cam": torch.tensor([3])}
    with torch.no_grad():
        latent = model.encode(video, 1.0)
        first = model.decode_queries(video, queries, latent)
        second = model.decode_queries(video, queries, latent)
    for key in first:
        assert torch.equal(first[key], second[key]), key


# ---------------------------------------------------------------------------
# 查询构造器 (论文 Tab.1)
# ---------------------------------------------------------------------------

def test_query_factory_shapes():
    depth = D4RTQueryFactory.build_depth_queries(T=4, H=8, W=8)
    assert len(depth) == 4
    for d in depth:
        assert d["uv"].shape == (64, 2)
        assert torch.equal(d["t_src"], d["t_tgt"]) and torch.equal(d["t_tgt"], d["t_cam"])
    assert torch.allclose(depth[0]["uv"][0], torch.tensor([0.5 / 8, 0.5 / 8]))

    pc = D4RTQueryFactory.build_point_cloud_queries(T=4, H=8, W=8, ref_cam=2)
    assert len(pc) == 4 and (pc[0]["t_cam"] == 2).all()

    tr = D4RTQueryFactory.build_point_track_queries(torch.tensor([0.3, 0.4]), t_src_idx=1, T=6)
    assert tr["uv"].shape == (6, 2)
    assert (tr["t_src"] == 1).all()
    assert torch.equal(tr["t_tgt"], torch.arange(6))

    q_i, q_j = D4RTQueryFactory.build_extrinsics_queries(0, 3, (4, 4))
    assert q_i["uv"].shape == (16, 2)
    assert (q_i["t_cam"] == 0).all() and (q_j["t_cam"] == 3).all()
    assert (q_i["t_src"] == 0).all() and (q_j["t_src"] == 0).all()


# ---------------------------------------------------------------------------
# 合成数据 GT
# ---------------------------------------------------------------------------

def test_gt_consistency():
    """GT 自洽: p_2d 应为 p_3d 的针孔投影; 静态背景 disp=0"""
    ds = SyntheticDataset(num_frames_out=4, num_frames_full=8,
                          render_res=96, out_res=64, seed=1)
    sample = ds.make_sample()
    queries, targets, mask = ds.sample_queries(sample, N=256)
    valid = mask[:, 0]
    assert valid.sum() > 200, "有效查询过少"

    f = sample["f_norm"]
    x0, y0, cw, ch = sample["crop"]
    proj_u = ((0.5 + f * targets["p_3d"][valid, 0] / targets["p_3d"][valid, 2])
              * ds.render_res - x0) / cw
    proj_v = ((0.5 + f * targets["p_3d"][valid, 1] / targets["p_3d"][valid, 2])
              * ds.render_res - y0) / ch
    assert (targets["p_2d"][valid, 0] - proj_u).abs().max() < 1e-4
    assert (targets["p_2d"][valid, 1] - proj_v).abs().max() < 1e-4

    # 背景不运动: 找一个帧 0 的背景像素, 构造 ttgt≠tcam 查询, disp 应为 0
    bg = torch.nonzero(sample["obj_id"][0] == 0)
    if bg.numel() > 0:
        py, px = bg[0]
        uv0 = torch.tensor([[(px + 0.5) / 96, (py + 0.5) / 96]])
        q = {"uv": uv0, "t_src": torch.tensor([0]),
             "t_tgt": torch.tensor([3]), "t_cam": torch.tensor([3])}
        tgt, m = ds.compute_gt(sample, q)
        if m.all():
            assert tgt["disp"].norm() < 1e-3, "静态背景位移应为 0"


def test_same_frame_2d_gt_uses_cropped_video_coordinates():
    ds = SyntheticDataset(num_frames_out=4, num_frames_full=8,
                          render_res=96, out_res=64, seed=12)
    sample = ds.make_sample()
    line = torch.linspace(0.1, 0.9, 12)
    vv, uu = torch.meshgrid(line, line, indexing="ij")
    uv = torch.stack((uu.flatten(), vv.flatten()), dim=-1)
    t = torch.zeros(len(uv), dtype=torch.long)
    q = {"uv": uv, "t_src": t, "t_tgt": t, "t_cam": t}
    target, mask = ds.compute_gt(sample, q)
    err = (target["p_2d"][mask[:, 0]] - uv[mask[:, 0]]).norm(dim=-1)
    assert err.numel() > 100
    assert err.median() < 0.01, f"同帧 GT 投影与查询坐标不一致: {err.median()}"


def test_tcam_eq_ratio():
    """训练采样: ttgt=tcam 比例 = 0.4 (强制) + 0.6/T (随机碰撞) (附录 A)"""
    ds = SyntheticDataset(num_frames_out=4, num_frames_full=8,
                          render_res=96, out_res=64, seed=2)
    sample = ds.make_sample()
    T = 4
    queries, _, _ = ds.sample_queries(sample, N=1000, rng=torch.Generator().manual_seed(3))
    ratio = (queries["t_cam"] == queries["t_tgt"]).float().mean().item()
    expect = 0.4 + 0.6 / T
    assert abs(ratio - expect) < 0.05, f"tcam=t_tgt 比例 {ratio:.3f} 偏离期望 {expect:.3f}"


def test_default_query_sampling_advances_rng():
    ds = SyntheticDataset(num_frames_out=4, num_frames_full=8,
                          render_res=64, out_res=32, seed=8)
    sample = ds.make_sample()
    q1, _, _ = ds.sample_queries(sample, N=64)
    q2, _, _ = ds.sample_queries(sample, N=64)
    assert not torch.equal(q1["uv"], q2["uv"])


def test_sphere_render_follows_rigid_pose():
    from data.synthetic import SphereSurface
    surf = SphereSurface(torch.tensor([0.0, 0.0, 5.0]), 1.0,
                         torch.tensor([0.5, 0.5, 0.5]), {"kind": "waves"})
    ray = torch.tensor([[[0.0, 0.0, 1.0]]])
    origin = torch.zeros(3)
    R = torch.eye(3)
    d1 = surf.render(origin, ray, torch.tensor([0.0, 0.0, 5.0]), R)["depth"]
    d2 = surf.render(origin, ray, torch.tensor([0.0, 0.0, 8.0]), R)["depth"]
    assert torch.allclose(d1, torch.tensor([[4.0]]))
    assert torch.allclose(d2, torch.tensor([[7.0]]))


def test_subsampled_object_gt_matches_source_render():
    from data.synthetic import SyntheticScene
    scene = SyntheticScene(96, 96, rng=torch.Generator().manual_seed(4))
    full = scene.render_all(8)
    t_map = torch.tensor([1, 3, 5, 7])
    sample = {k: full[k][t_map] for k in
              ("depth", "X_world", "normal_world", "obj_id", "C", "R_c2w")}
    sample.update({"t_map": t_map, "crop": (0, 0, 96, 96),
                   "obj_params": full["obj_params"], "f_norm": scene.f_norm})
    pix = torch.nonzero(sample["obj_id"][0] > 0)
    assert pix.numel() > 0, "测试场景应含可见运动物体"
    py, px = pix[len(pix) // 2]
    ox, oy = int(px), int(py)
    uv = torch.tensor([[(ox + 0.5) / 96, (oy + 0.5) / 96]])
    q = {"uv": uv, "t_src": torch.tensor([0]),
         "t_tgt": torch.tensor([0]), "t_cam": torch.tensor([0])}
    ds = SyntheticDataset(num_frames_out=4, num_frames_full=8,
                          render_res=96, out_res=96)
    target, mask = ds.compute_gt(sample, q)
    expected = sample["R_c2w"][0].T @ (sample["X_world"][0, oy, ox] - sample["C"][0])
    assert bool(mask[0, 0])
    assert torch.allclose(target["p_3d"][0], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# 密集跟踪 Alg.1
# ---------------------------------------------------------------------------

def test_dense_tracker_smoke():
    model = D4RTModel(variant="tiny", image_size=32, num_frames=4, max_frames=4)
    video = torch.rand(1, 4, 3, 32, 32)
    tracks, coverage = run_dense_tracking(model, video, grid=(8, 8),
                                          batch_size=16, max_iters=5)
    assert tracks.ndim == 3 and tracks.shape[1] == 4 and tracks.shape[2] == 3
    assert tracks.shape[0] >= 1
    assert 0.0 < coverage <= 1.0


if __name__ == "__main__":
    # 无 pytest 时的简易运行器
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} 个测试全部通过")
