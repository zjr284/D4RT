"""D4RT 本地合成场景评测脚本。

指标参照论文任务定义，但不是公开基准的官方评测器；请勿横向比较论文表格。

本地指标:
  - 深度    : AbsRel, S = 全序列单一全局尺度, SS = 全局尺度+平移 (§4.3:
              "both settings determine a single global scale factor ...
              across the sequence", 后者另加平移项)
  - 相机外参: 相邻帧相对旋转 (°)/平移误差 + 链式轨迹 ATE (Sim(3) 对齐)
              + Pose AUC@30 [48]
  - 点云重建: 共享参考帧下全部像素 3D 位置, 均值平移对齐后平均 L1
              (§4.3 "mean-shifting following Li et al. [29]")
  - 点跟踪  : 简化的 APD3D / AJ / OA 代理指标, 附可见帧平均 L1
  - 密集跟踪: 有 max_iters 预算时的占位栅格覆盖率 (不是论文准确率指标)
  - 内参    : 已知合成裁剪框时原图归一化焦距的相对误差

用法:
  python eval.py --ckpt checkpoints/repro_v3_seed0/tiny_last.pt --samples 5 --grid 64 --device cuda
"""

import argparse
import json
import math
from pathlib import Path

import torch

from data.synthetic import SyntheticDataset
from models import D4RTModel
from tracking import run_dense_tracking
from utils.camera import (estimate_camera_extrinsics,
                          estimate_camera_intrinsics, sim3_alignment)


def rot_err_deg(R_est, R_gt):
    """旋转矩阵间夹角 (度)"""
    R = R_est.T @ R_gt
    cos = ((torch.trace(R) - 1) / 2).clamp(-1, 1)
    return math.degrees(math.acos(cos.item()))


@torch.no_grad()
def eval_depth(model, sample, ds, grid=64, device="cpu"):
    """深度图: 每帧整幅网格查询, tsrc=ttgt=tcam=t, 取 p_3d 的 Z 分量

    对齐协议 (论文 §4.3): 两种设置都在**整个序列**上确定单一全局尺度,
    SS 另加一个全局平移项 (affine-invariant)。
    """
    T = sample["video"].shape[0]
    v = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
    u = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)  # (K, 2) CPU (GT 提取用)
    K = uv.shape[0]
    video = sample["video"].unsqueeze(0).to(device)
    latent_F = model.encode(video, sample["aspect"])

    zps, zgs = [], []
    for t in range(T):
        ts = torch.full((K,), t, dtype=torch.long)
        q = {"uv": uv, "t_src": ts, "t_tgt": ts, "t_cam": ts}
        q_dev = {k: v.to(device) for k, v in q.items()}
        preds = model.decode_queries(video, q_dev, latent_F)
        z_pred = preds["p_3d"][:, 2].cpu()  # (K,)

        targets, mask = ds.compute_gt(sample, q)
        z_gt = targets["p_3d"][:, 2]  # tcam=t 相机系下的 Z (即光线深度沿光轴分量)
        # 负深度是模型错误，不能剔除，否则会人为改善评测结果。
        valid = mask[:, 0] & (z_gt > 1e-3) & torch.isfinite(z_pred)
        zps.append(z_pred[valid])
        zgs.append(z_gt[valid])

    zp, zg = torch.cat(zps), torch.cat(zgs)
    if zp.numel() < 10:
        return float("nan"), float("nan")

    # S: 全序列单一全局尺度 (中位比例)
    positive = zp > 1e-3
    s = torch.median(zg[positive] / zp[positive]) if positive.any() else zp.new_tensor(1.0)
    absrel_s = (torch.abs(s * zp - zg) / zg).mean().item()
    # SS: 全局最小二乘 s·zp + b ≈ zg (全序列单一尺度与平移)
    A = torch.stack([zp, torch.ones_like(zp)], dim=1)
    sol = torch.linalg.lstsq(A, zg.unsqueeze(1)).solution.squeeze(1)
    absrel_ss = (torch.abs(sol[0] * zp + sol[1] - zg) / zg).mean().item()
    return absrel_s, absrel_ss


@torch.no_grad()
def eval_pose(model, sample, device="cpu", grid_size=(16, 16), dynamic_aware=False):
    """相邻帧相对位姿 + 链式轨迹 ATE (Sim(3) 对齐) + Pose AUC@30 [48]"""
    video = sample["video"].unsqueeze(0).to(device)
    T = sample["video"].shape[0]
    C, R = sample["C"].to(device), sample["R_c2w"].to(device)  # (T,3), (T,3,3)
    latent_F = model.encode(video, sample["aspect"])

    rot_errs, trans_errs = [], []
    pred_rel = []  # 累积估计相对位姿 (R_k, t_k): cam k → cam k+1 的变换
    for i in range(T - 1):
        j = i + 1
        R_est, t_est = estimate_camera_extrinsics(
            model, video, i, j, grid_size, dynamic_aware=dynamic_aware,
            latent_F=latent_F)
        # GT: cam i 系点 → cam j 系点
        R_gt = R[j].T @ R[i]
        t_gt = R[j].T @ (C[i] - C[j])
        rot_errs.append(rot_err_deg(R_est, R_gt))
        trans_errs.append((t_est.squeeze(1) - t_gt).norm().item())
        pred_rel.append((R_est, t_est.squeeze(1)))

    # 链式轨迹 (cam0 系): cam k+1 原点在 cam k 系中为 −R_kᵀ·t_k,
    # 再经累积旋转变换到 cam0 系; GT: cam k 原点在 cam0 系
    def _chain(rel):
        dev = rel[0][0].device
        R_acc = torch.eye(3, device=dev)
        t_acc = torch.zeros(3, device=dev)
        traj = [t_acc.clone()]
        for Rk, tk in rel:
            t_acc = t_acc - (Rk @ R_acc).T @ tk
            R_acc = Rk @ R_acc
            traj.append(t_acc.clone())
        return torch.stack(traj)  # (T, 3)

    traj_pred = _chain(pred_rel)
    traj_gt = torch.stack([R[0].T @ (C[t] - C[0]) for t in range(T)])
    s, R_a, t_a = sim3_alignment(traj_pred, traj_gt)
    aligned = traj_pred @ R_a.T * s + t_a.squeeze(1)
    ate = (aligned - traj_gt).norm(dim=-1).mean().item()

    # Pose AUC@30 [48]: 平均裁剪 (1 − err/30), err 为相对旋转角 (度)
    auc = float(torch.tensor([max(0.0, 1.0 - e / 30.0) for e in rot_errs]).mean())

    return (sum(rot_errs) / len(rot_errs), sum(trans_errs) / len(trans_errs), ate, auc)


@torch.no_grad()
def eval_tracking(model, sample, ds, num_tracks=32, device="cpu", seed=0):
    """点跟踪 (论文 Tab.4 相机系协议, TAPVid-3D [27]): 帧 0 随机源点, 全视频轨迹

    指标:
      - APD3D: 逐轨迹在 GT 可见帧中 3D 误差 < δ 的占比, 再对轨迹平均;
        δ = 0.1 × 点在首帧 (t=0 相机系) 的深度 (相对阈值, [27])
      - AJ (3D Average Jaccard): 逐轨迹 |正确∩GT可见| / |正确∪GT可见|,
        正确 = 预测可见 且 误差 < δ
      - OA (Occlusion Accuracy): (点, 帧) 可见性二分类正确率
      - L1: GT 可见帧 3D 误差均值 (Tab.4 世界系表格的补充指标)
    """
    T = sample["video"].shape[0]
    track_rng = torch.Generator().manual_seed(seed)
    uv = torch.rand(num_tracks, 2, generator=track_rng) * 0.8 + 0.1
    t_src = torch.zeros(num_tracks, dtype=torch.long)

    uv_all = uv.repeat_interleave(T, 0)
    tt = torch.arange(T).repeat(num_tracks)
    q = {"uv": uv_all, "t_src": t_src.repeat_interleave(T), "t_tgt": tt, "t_cam": tt}
    video = sample["video"].unsqueeze(0).to(device)
    latent_F = model.encode(video, sample["aspect"])
    preds = model.decode_queries(video, {k: v.to(device) for k, v in q.items()}, latent_F)
    targets, mask = ds.compute_gt(sample, q)

    ok = mask[:, 0].view(num_tracks, T)          # GT 有效 (光线击中几何体)
    err = (preds["p_3d"].cpu() - targets["p_3d"]).norm(dim=-1).view(num_tracks, T)
    vis_gt = (targets["vis"][:, 0] > 0.5).view(num_tracks, T)
    vis_pred = (preds["vis"][:, 0].cpu() > 0.5).view(num_tracks, T)
    depth0 = targets["p_3d"].view(num_tracks, T, 3)[:, 0].norm(dim=-1)  # 首帧深度

    apd3d, ajs, l1s = [], [], []
    for i in range(num_tracks):
        if not ok[i].any():
            continue
        delta = 0.1 * max(depth0[i].item(), 1e-3)
        vis = vis_gt[i] & ok[i]
        correct = vis_pred[i] & ok[i] & (err[i] < delta)
        if vis.sum() > 0:
            apd3d.append(correct[vis].float().mean().item())
            l1s.append(err[i][vis].mean().item())
        union = correct | vis
        if union.sum() > 0:
            ajs.append((correct & vis).float().sum().item() / union.float().sum().item())

    oa = (vis_pred[ok] == vis_gt[ok]).float().mean().item() if ok.any() else float("nan")

    def _m(l):
        return sum(l) / len(l) if l else float("nan")

    return _m(apd3d), _m(ajs), oa, _m(l1s), vis_gt.float().mean().item()


@torch.no_grad()
def eval_point_cloud(model, sample, ds, ref_cam=0, grid=64, device="cpu"):
    """点云重建 (论文 §4.3 / Tab.5 左): 所有帧所有像素预测在共享参考帧 tcam=ref_cam

    查询按 Tab.1 点云行: tsrc=1..T 全部像素, tcam 固定为参考帧;
    评测协议 (论文 §4.3): 预测与 GT 点云先做均值平移 (mean-shifting,
    Li et al. [29]) 对齐, 再报告平均 L1 距离。
    """
    T = sample["video"].shape[0]
    v = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
    u = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)  # (K, 2) CPU (GT 提取用)
    K = uv.shape[0]
    video = sample["video"].unsqueeze(0).to(device)
    latent_F = model.encode(video, sample["aspect"])

    pts_pred, pts_gt = [], []
    for t in range(T):
        ts = torch.full((K,), t, dtype=torch.long)
        cam = torch.full((K,), ref_cam, dtype=torch.long)
        q = {"uv": uv, "t_src": ts, "t_tgt": ts, "t_cam": cam}
        preds = model.decode_queries(video, {k: v.to(device) for k, v in q.items()}, latent_F)
        targets, mask = ds.compute_gt(sample, q)
        valid = mask[:, 0]
        pts_pred.append(preds["p_3d"].cpu()[valid])
        pts_gt.append(targets["p_3d"][valid])

    P = torch.cat(pts_pred)
    G = torch.cat(pts_gt)
    if P.numel() == 0:
        return float("nan")
    # 均值平移对齐 (MegaSaM 评测协议, 论文 §4.3)
    P = P - P.mean(dim=0)
    G = G - G.mean(dim=0)
    return (P - G).abs().mean().item()  # 所有 (点, 坐标维) 的平均 L1


@torch.no_grad()
def eval_intrinsics(model, sample, ds, device="cpu"):
    """合成原图焦距的相对误差；已知裁剪参数时先反变换查询坐标。"""
    video = sample["video"].unsqueeze(0).to(device)
    fx, fy, _, _ = estimate_camera_intrinsics(
        model, video, 0, aspect_ratio=sample["aspect"],
        crop=sample["crop"], source_size=(ds.render_res, ds.render_res))
    gt = sample["f_norm"]
    return (abs(fx - gt) + abs(fy - gt)) / (2 * gt)


def main():
    ap = argparse.ArgumentParser(description="D4RT 评测")
    ap.add_argument("--ckpt", required=True, help="检查点路径")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--grid", type=int, default=64, help="深度/点云评测网格分辨率")
    ap.add_argument("--tracks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="固定评测场景种子")
    ap.add_argument("--json-out", default=None, help="可选 JSON 结果文件")
    ap.add_argument("--device", default=None, help="cuda/cpu (默认自动)")
    ap.add_argument("--dynamic-aware", action="store_true",
                    help="使用置信度+运动残差+鲁棒 IRLS 的动态感知相机配准")
    args = ap.parse_args()
    if args.samples < 1 or args.grid < 2 or args.tracks < 1:
        ap.error("--samples、--tracks 必须为正整数，--grid 至少为 2")

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[eval] 检查点: {args.ckpt} | step={ckpt.get('step')} | 配置={cfg}")

    model = D4RTModel(variant=cfg["config"], image_size=cfg["res"],
                      num_frames=cfg["frames"], max_frames=cfg["frames"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SyntheticDataset(num_frames_out=cfg["frames"], num_frames_full=max(2 * cfg["frames"], 16),
                          render_res=cfg.get("render_res", 256),
                          out_res=cfg["res"], seed=args.seed)

    rows = []
    for k in range(args.samples):
        sample = ds.make_sample()
        absrel_s, absrel_ss = eval_depth(model, sample, ds, args.grid, device)
        r_err, t_err, ate, auc = eval_pose(model, sample, device,
                                           dynamic_aware=args.dynamic_aware)
        apd3d, aj, oa, l1, frac_vis = eval_tracking(
            model, sample, ds, args.tracks, device, seed=args.seed + k)
        pc_l1 = eval_point_cloud(model, sample, ds, grid=args.grid, device=device)
        focal_rel = eval_intrinsics(model, sample, ds, device)
        cov = run_dense_tracking(model, sample["video"].unsqueeze(0).to(device),
                                 grid=(48, 48), batch_size=48, max_iters=20,
                                 aspect_ratio=sample["aspect"])[1]
        rows.append((absrel_s, absrel_ss, r_err, t_err, ate, auc,
                     apd3d, aj, oa, l1, pc_l1, cov, focal_rel))
        print(f"  样本 {k+1}: AbsRel(S)={absrel_s:.4f} AbsRel(SS)={absrel_ss:.4f} "
              f"| 旋转={r_err:.2f}° 平移={t_err:.3f} ATE={ate:.3f} AUC@30={auc:.2f} "
              f"| APD3D={apd3d:.3f} AJ={aj:.3f} OA={oa:.3f} L1={l1:.3f} "
              f"点云L1={pc_l1:.3f} 覆盖={cov:.2f} 焦距相对误差={focal_rel:.3f}")

    n = len(rows)
    m = lambda i: sum(r[i] for r in rows) / n
    print(f"\n[eval] 平均 ({n} 样本): AbsRel(S)={m(0):.4f} AbsRel(SS)={m(1):.4f} "
          f"| 旋转={m(2):.2f}° 平移={m(3):.3f} ATE={m(4):.3f} AUC@30={m(5):.2f} "
          f"| APD3D={m(6):.3f} AJ={m(7):.3f} OA={m(8):.3f} L1={m(9):.3f} "
          f"点云L1={m(10):.3f} 覆盖={m(11):.2f} 焦距相对误差={m(12):.3f}")
    if args.json_out:
        names = ("absrel_s", "absrel_ss", "rot_deg", "trans", "ate", "pose_auc30",
                 "apd3d", "aj", "oa", "tracking_l1", "point_cloud_l1", "coverage",
                 "focal_rel")
        def clean(value):
            return value if math.isfinite(value) else None
        payload = {
            "checkpoint": args.ckpt, "checkpoint_step": ckpt.get("step"),
            "seed": args.seed, "samples": args.samples, "grid": args.grid,
            "tracks": args.tracks, "dynamic_aware": args.dynamic_aware,
            "per_sample": [dict(zip(names, map(clean, row))) for row in rows],
            "mean": {name: clean(m(i)) for i, name in enumerate(names)},
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
