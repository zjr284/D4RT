"""高效密集全像素跟踪 (论文 Algorithm 1: Efficient Dense Tracking of All Pixels)

朴素做法需要 O(T²HW) 次查询, 大量冗余。算法用占位栅格 G ∈ {0,1}^{T×H×W}
记录已被某条轨迹"可见通过"的时空像素, 只从**未访问**的像素发起新轨迹,
每条完整轨迹把它可见经过的所有时空像素标记为已访问。论文报告此策略带来
5–15× 的自适应加速 (取决于场景运动复杂度)。

本实现按 Alg.1 忠实实现, 并把每个批次的 (B×T) 个查询合并为一次解码器前向
(解码器对查询天然并行, 论文 §2.2)。
"""

import torch


@torch.no_grad()
def run_dense_tracking(model, video, grid=(64, 64), batch_size=64,
                       aspect_ratio=None, max_iters=None):
    """对视频所有 (栅格) 像素做 3D 密集跟踪, 返回 (tracks, coverage)

    - tracks: (M, T, 3) 每条轨迹 T 个时刻的 3D 位置
    - coverage: 占位栅格最终覆盖率 (1.0 = 全部时空像素被覆盖)
    video: (1, T, 3, H, W)
    """
    model.eval()
    device = video.device
    T = video.shape[1]
    H, W = grid

    # Alg.1 第 1-3 行: 场景表示 F 只算一次, 初始化占位栅格与轨迹集合
    F = model.encode(video, aspect_ratio)
    G = torch.zeros(T, H, W, dtype=torch.bool, device=device)
    tracks = []

    tt = torch.arange(T, device=device, dtype=torch.long)
    # 栅格像素中心坐标 (归一化)
    xs = (torch.arange(W, device=device) + 0.5) / W
    ys = (torch.arange(H, device=device) + 0.5) / H

    iters = 0
    while not G.all():  # Alg.1 第 4 行
        iters += 1
        if max_iters is not None and iters > max_iters:
            break
        unvisited = torch.nonzero(~G)  # (K, 3) [t, y, x]
        if unvisited.shape[0] == 0:
            break
        # Alg.1 第 5 行: 批量采样未访问源点
        idx = torch.randperm(unvisited.shape[0], device=device)[:batch_size]
        batch = unvisited[idx]  # (B, 3)
        B = batch.shape[0]

        t_src = batch[:, 0]
        uv = torch.stack([xs[batch[:, 2]], ys[batch[:, 1]]], dim=-1)  # (B, 2)

        # Alg.1 第 7-8 行: 每个源点展开为 T 个查询 (ttgt = tcam = 1..T),
        # 合并为一次前向 (解码器对查询并行)
        q = {
            "uv": uv.repeat_interleave(T, 0),
            "t_src": t_src.repeat_interleave(T),
            "t_tgt": tt.tile(B),
            "t_cam": tt.tile(B),
        }
        preds = model.decode_queries(video, q, F)
        p3d = preds["p_3d"].view(B, T, 3)
        p2d_raw = preds["p_2d"].view(B, T, 2)
        inside = ((p2d_raw >= 0) & (p2d_raw <= 1)).all(dim=-1)
        vis = (preds["vis"].view(B, T) > 0.5) & inside
        p2d = p2d_raw.clamp(0, 1)

        # Alg.1 第 9 行: 轨迹可见通过的时空像素标记为已访问
        px = (p2d[..., 0] * W - 0.5).round().long().clamp(0, W - 1)
        py = (p2d[..., 1] * H - 0.5).round().long().clamp(0, H - 1)
        ttile = tt.unsqueeze(0).expand(B, T)
        G[ttile[vis], py[vis], px[vis]] = True
        # 源点自身按构造可见, 强制标记 (防御模型漏报导致死循环)
        G[t_src, batch[:, 1], batch[:, 2]] = True

        # Alg.1 第 10 行: 轨迹加入输出
        tracks.append(p3d)

    tracks = torch.cat(tracks, dim=0) if tracks else torch.zeros(0, T, 3, device=device)
    return tracks, G.float().mean().item()
