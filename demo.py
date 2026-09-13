"""D4RT 推理演示: 如何用训练好的模型做深度 / 点跟踪 / 相机位姿 / 点云

核心思想 (论文 §2.1): 场景表示 F = E(V) 只算一次, 之后任何时空点查询
P = D(q, F) 并行解码, 一个接口统一所有任务 (Tab.1)。

用法:
  # 1. 合成场景 (开箱即用, 直接演示各任务输出)
  .venv/bin/python demo.py --ckpt checkpoints/run1/tiny_last.pt

  # 2. 你自己的视频 (自动抽 12 帧 → 方形 160×160 → [0,1])
  .venv/bin/python demo.py --ckpt checkpoints/run1/tiny_last.pt --video 视频.mp4

输出 (默认存到 demo_out/):
  depth.npz    每帧深度图 (T, S, S), 注意绝对尺度未校准 (见 README)
  tracks.npz   稀疏点 3D 轨迹 (M, T, 3)
  dense.npz    密集跟踪轨迹 (Alg.1) 与覆盖率
  traj.npz     相机轨迹 (链式相对位姿, 帧 0 为原点)
  points.ply   共享参考帧 (tcam=0) 点云, 可用 MeshLab/CloudCompare 打开

模型输入约定 (训练配置决定):
  12 帧、正方形 160×160、像素值 [0,1]、宽高比 token = 原始 W/H。
  本 tiny 模型只在合成数据上训了 300 步, 用于演示接口; 真实视频为
  零样本泛化, 效果有限; 深度/点云的绝对尺度未校准 (论文 L3D 按均值
  深度归一化, 绝对尺度不受监督), 需与真值对齐后才有米制意义。
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic import SyntheticDataset
from models import D4RTModel
from tracking import run_dense_tracking
from utils.camera import estimate_camera_extrinsics, estimate_camera_intrinsics


# ---------------------------------------------------------------------------
# 1. 加载模型
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device="cuda"):
    """从训练检查点重建模型 (检查点里保存了训练时的配置)"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[demo] 检查点: step={ckpt.get('step')} 配置={cfg}")
    model = D4RTModel(variant=cfg["config"], image_size=cfg["res"],
                      num_frames=cfg["frames"], max_frames=cfg["frames"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_video(path: str, num_frames: int = 12, size: int = 160):
    """把真实视频变成模型输入: 均匀抽帧 → [0,1] → 正方形 resize

    返回 (video (1,T,3,S,S), aspect (原始 W/H, 宽高比 token 输入))。
    """
    from torchvision.io import read_video
    v, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")  # (T,3,H,W) uint8
    if v.shape[0] < 2:
        raise RuntimeError("视频太短")
    idx = torch.linspace(0, v.shape[0] - 1, num_frames).round().long()
    v = v[idx].float() / 255.0
    aspect = v.shape[-1] / v.shape[-2]  # 原始 W/H (resize 前)
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    return v.unsqueeze(0), aspect


def get_synthetic_video(seed=7, num_frames=12, size=160, render_res=256):
    """合成场景样本 (与训练同分布, 效果最好); 返回 (video, aspect, sample)"""
    ds = SyntheticDataset(num_frames_out=num_frames,
                          num_frames_full=max(2 * num_frames, 16),
                          render_res=render_res, out_res=size, seed=seed)
    sample = ds.make_sample()
    video = sample["video"].unsqueeze(0)
    return video, sample["aspect"], sample


# ---------------------------------------------------------------------------
# 2. 各任务: 全部通过同一个解码接口 (论文 Tab.1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_depth(model, video, F, aspect, grid=128):
    """深度图 (Tab.1 Depth Map 行): tsrc=ttgt=tcam=t, 取 p_3d 的 Z 分量"""
    T, S = video.shape[1], video.shape[-1]
    uv = _grid_uv(grid, device=video.device)
    depth = []
    for t in range(T):
        ts = torch.full((uv.shape[0],), t, dtype=torch.long, device=video.device)
        q = {"uv": uv, "t_src": ts, "t_tgt": ts, "t_cam": ts}
        preds = _decode(model, video, F, aspect, q)
        depth.append(preds["p_3d"][:, 2].reshape(grid, grid).cpu().numpy())
    return np.stack(depth)  # (T, grid, grid)


@torch.no_grad()
def run_point_tracks(model, video, F, aspect, num_tracks=8, grid_span=(0.15, 0.85)):
    """点跟踪 (Tab.1 Point Track 行): 帧 0 固定源点, ttgt=tcam=1..T"""
    T = video.shape[1]
    torch.manual_seed(0)
    uv0 = torch.rand(num_tracks, 2) * (grid_span[1] - grid_span[0]) + grid_span[0]
    uv0 = uv0.to(video.device)
    t_src = torch.zeros(num_tracks, dtype=torch.long, device=video.device)
    tt = torch.arange(T, device=video.device)
    q = {"uv": uv0.repeat_interleave(T, 0), "t_src": t_src.repeat_interleave(T),
         "t_tgt": tt.tile(num_tracks), "t_cam": tt.tile(num_tracks)}
    preds = _decode(model, video, F, aspect, q)
    tracks = preds["p_3d"].reshape(num_tracks, T, 3).cpu().numpy()   # (M,T,3)
    vis = preds["vis"].reshape(num_tracks, T).cpu().numpy()          # (M,T)
    return tracks, vis, uv0.cpu().numpy()


@torch.no_grad()
def run_pose(model, video, F, aspect):
    """相机位姿 (论文 §2.1): 相邻帧 Umeyama + 链式轨迹 (帧 0 为原点)"""
    T = video.shape[1]
    rel = []
    for i in range(T - 1):
        R_est, t_est = estimate_camera_extrinsics(
            model, video, i, i + 1, (16, 16), aspect_ratio=aspect, latent_F=F)
        rel.append((R_est, t_est.squeeze(1)))
    # 链式累积: cam k+1 原点在 cam k 系中为 −R_kᵀ·t_k, 再变换到 cam0 系
    R_acc = torch.eye(3, device=video.device)
    t_acc = torch.zeros(3, device=video.device)
    traj = [t_acc.clone()]
    for Rk, tk in rel:
        t_acc = t_acc - (Rk @ R_acc).T @ tk
        R_acc = Rk @ R_acc
        traj.append(t_acc.clone())
    return torch.stack(traj).cpu().numpy()  # (T, 3)


@torch.no_grad()
def run_point_cloud(model, video, F, aspect, grid=96):
    """点云 (Tab.1 Point Cloud 行): 所有帧像素预测在共享参考帧 tcam=0,
    按可见性过滤, 附带每点 RGB 颜色 (取自源帧像素)"""
    T, S = video.shape[1], video.shape[-1]
    uv = _grid_uv(grid, device=video.device)
    pts, cols = [], []
    for t in range(T):
        ts = torch.full((uv.shape[0],), t, dtype=torch.long, device=video.device)
        cam = torch.zeros_like(ts)
        q = {"uv": uv, "t_src": ts, "t_tgt": ts, "t_cam": cam}
        preds = _decode(model, video, F, aspect, q)
        keep = (preds["vis"][:, 0] > 0.5)
        # 颜色: 在 t 帧上双线性采样 (uv 归一化坐标 → 像素)
        gx = uv[:, 0] * S - 0.5
        gy = uv[:, 1] * S - 0.5
        col = _bilinear_color(video[0, t], gx, gy).cpu().numpy()  # (K, 3) 0..1
        pts.append(preds["p_3d"][keep].cpu().numpy())
        cols.append(col[keep.cpu().numpy()])
    return np.concatenate(pts), np.concatenate(cols)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _grid_uv(grid, device):
    v = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid, device=device)
    u = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid, device=device)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([gu.flatten(), gv.flatten()], dim=-1)  # (K, 2)


def _decode(model, video, F, aspect, q):
    """论文范式: F 已算好, 任意查询批量解码 (训练时 model(video, q) 一步完成)"""
    q_tokens = model.query_embedder(video, q["uv"], q["t_src"], q["t_tgt"], q["t_cam"])
    return model.decoder(q_tokens, F)


def _bilinear_color(frame, gx, gy):
    """frame: (3,S,S); (gx,gy): (K,) 像素坐标 → (K,3) 双线性颜色"""
    S = frame.shape[-1]
    gx = gx.clamp(0, S - 1)
    gy = gy.clamp(0, S - 1)
    x0 = gx.floor().long().clamp(0, S - 2)
    y0 = gy.floor().long().clamp(0, S - 2)
    wx, wy = gx - x0, gy - y0
    c = (frame[:, y0, x0] * (1 - wx) * (1 - wy)
         + frame[:, y0, x0 + 1] * wx * (1 - wy)
         + frame[:, y0 + 1, x0] * (1 - wx) * wy
         + frame[:, y0 + 1, x0 + 1] * wx * wy)
    return c.T


def save_ply(path, pts, colors):
    """写 ASCII PLY (xyz + rgb), 供 MeshLab/CloudCompare 查看"""
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="D4RT 推理演示")
    ap.add_argument("--ckpt", required=True, help="检查点路径")
    ap.add_argument("--video", default=None, help="可选: 真实视频路径 (mp4 等)")
    ap.add_argument("--out", default="demo_out", help="输出目录")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)

    # ---- 输入视频 ----
    if args.video is not None:
        video, aspect = load_video(args.video, num_frames=cfg["frames"], size=cfg["res"])
        video = video.to(device)
        print(f"[demo] 加载真实视频: {args.video} → {tuple(video.shape)}, 宽高比={aspect:.3f}")
    else:
        video, aspect, sample = get_synthetic_video(
            num_frames=cfg["frames"], size=cfg["res"],
            render_res=cfg.get("render_res", 256))
        video = video.to(device)
        print(f"[demo] 合成场景 (与训练同分布), 宽高比={aspect:.3f}")

    # ---- 核心范式: F = E(V) 只算一次 ----
    F = model.encode(video, aspect)
    print(f"[demo] 场景表示 F 已编码, shape={tuple(F.shape)}")

    os.makedirs(args.out, exist_ok=True)
    low_memory = cfg.get("low_memory", False)

    # ---- 深度图 ----
    depth = run_depth(model, video, F, aspect, grid=64 if low_memory else 128)
    np.savez(os.path.join(args.out, "depth.npz"), depth=depth)
    print(f"[demo] 深度图: {depth.shape} (T,H,W) → depth.npz"
          f" | 范围 [{depth.min():.2f}, {depth.max():.2f}] (尺度未校准, 仅相对关系有效)")

    # ---- 点跟踪 ----
    tracks, vis, uv0 = run_point_tracks(model, video, F, aspect)
    np.savez(os.path.join(args.out, "tracks.npz"), tracks=tracks, vis=vis, uv0=uv0)
    print(f"[demo] 稀疏点跟踪: {tracks.shape[0]} 条轨迹 × {tracks.shape[1]} 帧 → tracks.npz"
          f" | 平均可见率 {vis.mean():.2f}")

    # ---- 相机位姿 ----
    traj = run_pose(model, video, F, aspect)
    np.savez(os.path.join(args.out, "traj.npz"), traj=traj)
    print(f"[demo] 相机轨迹: {traj.shape} (帧 0 为原点) → traj.npz"
          f" | 总位移 {np.linalg.norm(traj[-1] - traj[0]):.2f}")

    # ---- 点云 (共享参考帧) ----
    pts, cols = run_point_cloud(model, video, F, aspect, grid=48 if low_memory else 96)
    save_ply(os.path.join(args.out, "points.ply"), pts, cols)
    print(f"[demo] 点云: {len(pts)} 个点 (tcam=0 参考系) → points.ply")

    # ---- 密集跟踪 (论文 Alg.1) ----
    dense_grid = (32, 32) if low_memory else (48, 48)
    dense, coverage = run_dense_tracking(model, video, grid=dense_grid,
                                         batch_size=32 if low_memory else 48,
                                         max_iters=10 if low_memory else 20,
                                         aspect_ratio=aspect)
    np.savez(os.path.join(args.out, "dense.npz"), tracks=dense.cpu().numpy())
    print(f"[demo] 密集跟踪 (Alg.1): {dense.shape[0]} 条轨迹, 时空栅格覆盖率 {coverage:.2f} → dense.npz")

    print(f"\n[demo] 全部输出在 {args.out}/ (depth.npz / tracks.npz / traj.npz / points.ply / dense.npz)")


if __name__ == "__main__":
    main()
