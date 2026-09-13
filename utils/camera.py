"""相机估计工具 (论文 §2.1 与附录 B)

- Umeyama 算法 [46]: 通过 3×3 SVD 求解两个 3D 点集间的刚性变换 (论文 §2.1
  用它对两套参考帧下的同一批 3D 点求相对相机位姿)
- 内参估计: 针孔模型 + 主点 (0.5, 0.5), fx = pz(u−0.5)/px, fy = pz(v−0.5)/py,
  对网格估计取中位数 (论文 §2.1)
- Sim(3) 变体: 附录 B 长视频 chunk 拼接与评测轨迹对齐用
"""

import torch


def grid_uv(h: int, w: int, margin: float = 0.0, device="cpu"):
    """在归一化坐标下生成 (h, w) 网格上的 (u, v) 采样点, (K=h*w, 2)"""
    u = torch.linspace(margin, 1 - margin, w, device=device)
    v = torch.linspace(margin, 1 - margin, h, device=device)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([gu.flatten(), gv.flatten()], dim=-1)


def umeyama(src_pts: torch.Tensor, tgt_pts: torch.Tensor, with_scale: bool = False):
    """Umeyama [46]: 求解 tgt = s·R·src + t 的最小二乘解

    src_pts, tgt_pts: (K, 3)。协方差约定 H = Σ srcᶜᵀ tgtᶜ, 此时
    R = V Uᵀ 为 src→tgt 的旋转 (用 double 计算 SVD 保证数值稳定)。
    返回 (R, t) 或 (s, R, t), t 为 (3, 1)。
    """
    src = src_pts.double()
    tgt = tgt_pts.double()
    mu_src = src.mean(dim=0, keepdim=True)
    mu_tgt = tgt.mean(dim=0, keepdim=True)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    H = src_c.T @ tgt_c  # (3, 3)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T  # src → tgt
    sign = 1.0
    # 反射修正: 翻转最小奇异值对应的列 (标准 Umeyama)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        sign = -1.0

    var_src = (src_c ** 2).sum()
    signed_sum = S[:-1].sum() + sign * S[-1]
    scale = (signed_sum / var_src).item() if var_src > 0 else 1.0
    # 平移须含尺度: t = mu_tgt − s·R·mu_src
    t = mu_tgt.T - (scale if with_scale else 1.0) * R @ mu_src.T
    if not with_scale:
        return R.float(), t.float()
    return scale, R.float(), t.float()


def weighted_umeyama(src_pts: torch.Tensor, tgt_pts: torch.Tensor,
                     weights: torch.Tensor, with_scale: bool = False,
                     eps: float = 1e-8):
    """带非负点权重的 Umeyama: ``tgt ≈ s·R·src + t``。

    这是动态感知相机估计的基础：置信度高、运动残差小的点获得更大权重，
    动态物体或遮挡产生的可疑对应不会像普通最小二乘那样等比例污染位姿。
    ``weights`` 只影响配准，不参与模型反向传播；输入可为任意形状，最终展平
    为 (K,)。当权重总和接近 0 时回退为均匀权重。
    """
    src = src_pts.double()
    tgt = tgt_pts.double()
    w = weights.reshape(-1).double().clamp_min(0)
    if w.numel() != src.shape[0] or tgt.shape != src.shape:
        raise ValueError("src_pts/tgt_pts 必须是同形状的 (K,3)，weights 必须是 (K,)")
    w_sum = w.sum()
    if not torch.isfinite(w_sum) or w_sum <= eps:
        w = torch.ones_like(w)
        w_sum = w.sum()
    w = w / w_sum

    mu_src = (src * w[:, None]).sum(dim=0, keepdim=True)
    mu_tgt = (tgt * w[:, None]).sum(dim=0, keepdim=True)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    H = src_c.T @ (tgt_c * w[:, None])
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    sign = 1.0
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        sign = -1.0

    var_src = (w * (src_c ** 2).sum(dim=-1)).sum()
    signed_sum = S[:-1].sum() + sign * S[-1]
    scale = (signed_sum / var_src).item() if var_src > eps else 1.0
    t = mu_tgt.T - (scale if with_scale else 1.0) * R @ mu_src.T
    if not with_scale:
        return R.float(), t.float()
    return scale, R.float(), t.float()


def _transform_points(points: torch.Tensor, R: torch.Tensor,
                      t: torch.Tensor, scale: float = 1.0):
    """应用 ``s·R·p+t``，兼容列向量形式的平移 (3,1)。"""
    return scale * (points @ R.T) + t.reshape(1, 3)


def dynamic_aware_umeyama(src_pts: torch.Tensor, tgt_pts: torch.Tensor,
                          confidence: torch.Tensor = None,
                          motion: torch.Tensor = None,
                          visibility: torch.Tensor = None,
                          robust_iters: int = 3):
    """用置信度、运动线索和 IRLS 鲁棒化估计刚体位姿。

    ``motion`` 表示每个对应点在同一目标相机坐标系中的自身运动幅度；
    ``visibility`` 是可选的软可见性门控。该函数不依赖模型，便于在真实
    预测、合成真值和离线消融中复用。返回 ``(R, t, diagnostics)``。
    """
    K = src_pts.shape[0]
    weights = torch.ones(K, device=src_pts.device, dtype=src_pts.dtype)
    if confidence is not None:
        weights = weights * confidence.reshape(-1).to(weights).clamp_min(1e-6)
    if visibility is not None:
        weights = weights * visibility.reshape(-1).to(weights).clamp_min(1e-4)
    if motion is not None:
        motion = motion.reshape(-1).to(weights)
        motion_scale = torch.median(motion).clamp_min(1e-4)
        weights = weights / (1.0 + (motion / motion_scale).square())
    weights = weights.detach().clamp_min(1e-6)

    R, t = weighted_umeyama(src_pts, tgt_pts, weights)
    residual = torch.linalg.vector_norm(_transform_points(src_pts, R, t) - tgt_pts, dim=-1)
    for _ in range(max(0, int(robust_iters))):
        residual_scale = torch.median(residual).clamp_min(1e-4)
        robust = 1.0 / (1.0 + (residual / (2.5 * residual_scale)).square())
        weights = (weights * robust).detach().clamp_min(1e-6)
        R, t = weighted_umeyama(src_pts, tgt_pts, weights)
        residual = torch.linalg.vector_norm(_transform_points(src_pts, R, t) - tgt_pts, dim=-1)
    diagnostics = {
        "weights": weights,
        "residual": residual,
    }
    if motion is not None:
        diagnostics["motion"] = motion
    diagnostics["static_weight_fraction"] = (weights > weights.median()).float().mean()
    return R, t, diagnostics


@torch.no_grad()
def estimate_camera_extrinsics(model, video, frame_i: int, frame_j: int,
                               grid_size=(16, 16), aspect_ratio=None,
                               dynamic_aware: bool = False,
                               robust_iters: int = 3,
                               return_diagnostics: bool = False,
                               latent_F: torch.Tensor = None):
    """论文 §2.1: 帧 i → 帧 j 的相对相机位姿

    对同一批网格点构造两套查询 q_i=(u,v,i,i,i) 与 q_j=(u,v,i,i,j),
    解码得到两套参考系下的同一组 3D 点, 用 Umeyama 求刚性变换。
    """
    model.eval()
    device = video.device
    # 多个相邻帧对共享同一个全局场景表示；评测时可传入 latent_F，
    # 避免每个 frame pair 重复执行昂贵的编码器。
    F = model.encode(video, aspect_ratio) if latent_F is None else latent_F
    uv = grid_uv(*grid_size, margin=0.05, device=device)
    K = uv.shape[0]

    t_i = torch.full((K,), frame_i, dtype=torch.long, device=device)
    t_j = torch.full((K,), frame_j, dtype=torch.long, device=device)

    q_i = model.query_embedder(video, uv, t_i, t_i, t_i)
    q_j = model.query_embedder(video, uv, t_i, t_i, t_j)
    out_i = model.decoder(q_i, F)
    out_j = model.decoder(q_j, F)
    P_i = out_i["p_3d"]  # 帧 i 坐标系下的点 (K, 3)
    P_j = out_j["p_3d"]  # 帧 j 坐标系下的同一点 (K, 3)

    if not dynamic_aware:
        R, t = umeyama(P_i, P_j)  # 点云 i → j 的变换
        if return_diagnostics:
            return R, t, {"weights": torch.ones(K, device=device),
                          "residual": torch.linalg.vector_norm(_transform_points(P_i, R, t) - P_j, dim=-1)}
        return R, t

    # 额外查询同一点在目标时刻 j、目标相机 j 中的位置。与 P_j（源点在时刻 i
    # 的状态、表达在相机 j）相减，可估计物体自身运动；相机运动已被消除。
    q_state_j = model.query_embedder(video, uv, t_i, t_j, t_j)
    out_state_j = model.decoder(q_state_j, F)
    P_state_j = out_state_j["p_3d"]
    motion = torch.linalg.vector_norm(P_state_j - P_j, dim=-1)

    # 置信度是软权重；可见性只作为温和门控，避免未训练好的可见性头直接删除点。
    conf = torch.sqrt((out_i["conf"].reshape(-1) * out_j["conf"].reshape(-1)).clamp_min(1e-6))
    vis_gate = torch.sqrt((out_j["vis"].reshape(-1) *
                           out_state_j["vis"].reshape(-1)).clamp_min(1e-4))
    R, t, diagnostics = dynamic_aware_umeyama(
        P_i, P_j, confidence=conf, motion=motion, visibility=vis_gate,
        robust_iters=robust_iters)

    if return_diagnostics:
        return R, t, diagnostics
    return R, t


@torch.no_grad()
def estimate_camera_intrinsics(model, video, frame_idx: int,
                               grid_size=(16, 16), aspect_ratio=None,
                               crop=None, source_size=None,
                               latent_F: torch.Tensor = None):
    """论文 §2.1: 帧 frame_idx 的内参焦距

    假设针孔模型且主点位于 (0.5, 0.5), 由 P=(px,py,pz) 反推
    fx = pz(u−0.5)/px, fy = pz(v−0.5)/py, 对网格估计取中位数。
    返回归一化焦距 (fx, fy) 与像素焦距 (fx_pix, fy_pix)。若视频经过裁剪，
    须同时给出原图像素裁剪框 crop=(x0,y0,cw,ch) 和 source_size=(H,W)，
    先把查询坐标映回原图；此时返回值也相对于原图。否则假设当前视频的
    主点位于中心，返回值相对于当前视频。缺少裁剪元数据时无法恢复原内参。
    """
    model.eval()
    device = video.device
    if (crop is None) != (source_size is None):
        raise ValueError("crop 与 source_size 必须同时提供")
    F = model.encode(video, aspect_ratio) if latent_F is None else latent_F
    uv = grid_uv(*grid_size, margin=0.1, device=device)
    K = uv.shape[0]

    t_seq = torch.full((K,), frame_idx, dtype=torch.long, device=device)
    q_tokens = model.query_embedder(video, uv, t_seq, t_seq, t_seq)
    P = model.decoder(q_tokens, F)["p_3d"]  # (K, 3)
    px, py, pz = P[:, 0], P[:, 1], P[:, 2]

    # 避开主点附近与异常深度 (健壮性过滤)
    valid = (pz > 1e-3) & (torch.abs(px) > 1e-3) & (torch.abs(py) > 1e-3)
    if crop is not None:
        x0, y0, cw, ch = crop
        H, W = source_size
        uv_intrinsic = torch.stack(((x0 + uv[:, 0] * cw) / W,
                                    (y0 + uv[:, 1] * ch) / H), dim=-1)
    else:
        H, W = video.shape[-2], video.shape[-1]
        uv_intrinsic = uv
    fx_est = pz[valid] * (uv_intrinsic[valid, 0] - 0.5) / px[valid]
    fy_est = pz[valid] * (uv_intrinsic[valid, 1] - 0.5) / py[valid]
    if valid.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    fx = torch.median(fx_est).item()
    fy = torch.median(fy_est).item()
    return fx, fy, fx * W, fy * H


@torch.no_grad()
def sim3_alignment(src_pts: torch.Tensor, tgt_pts: torch.Tensor,
                   conf_src: torch.Tensor = None, conf_tgt: torch.Tensor = None,
                   top_frac: float = 0.85):
    """Sim(3) 对齐 (附录 B 长视频拼接 / 评测 ATE 用): 返回 (s, R, t)

    对齐后的 src 点: s·R·src + t。
    附录 B: "基于重叠区置信度最高的 top-85% 的点估计 Sim(3) 变换" —
    传入 conf_src/conf_tgt 时按两者置信度乘积筛选 top-85%, 剔除低置信离群点。
    """
    if conf_src is not None or conf_tgt is not None:
        c = torch.ones(src_pts.shape[0], device=src_pts.device, dtype=src_pts.dtype)
        if conf_src is not None:
            c = c * conf_src.reshape(-1)
        if conf_tgt is not None:
            c = c * conf_tgt.reshape(-1)
        k = max(4, int(c.numel() * top_frac))  # Umeyama 至少需要 3 个非退化点
        top = torch.topk(c, k).indices
        src_pts, tgt_pts = src_pts[top], tgt_pts[top]
    s, R, t = umeyama(src_pts, tgt_pts, with_scale=True)
    return s, R, t


@torch.no_grad()
def stitch_chunks(points_a: torch.Tensor, points_b: torch.Tensor,
                  conf_a: torch.Tensor = None, conf_b: torch.Tensor = None,
                  top_frac: float = 0.85):
    """附录 B 长视频 chunk 拼接: 把 chunk b 的点云对齐到 chunk a 的坐标系

    points_a / points_b: (K, 3) 两 chunk 在重叠区的对应点 (各自局部坐标系);
    用置信度乘积筛选 top-85% 的点做 Sim(3) Umeyama, 返回 (s, R, t) 使
    s·R·points_b + t ≈ points_a (即点云 b 变换到 a 系的相似变换)。
    论文做法与 VGGT-Long [6] 第一阶段一致 (省略环路检测与全局优化)。
    """
    s, R, t = sim3_alignment(points_b, points_a, conf_b, conf_a, top_frac)
    return s, R, t
