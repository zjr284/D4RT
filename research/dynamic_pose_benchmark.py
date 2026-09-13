"""小型动态点集压力测试：比较普通与动态感知 Umeyama。

这不是论文数据集评测，而是科研选题的可控 sanity check：固定真实刚体
变换，逐步增加动态离群点，观察运动线索能否保护旋转和平移估计。

运行:
  .venv/bin/python research/dynamic_pose_benchmark.py --seeds 5
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# 允许直接以 ``python research/...py`` 运行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.camera import dynamic_aware_umeyama, umeyama


def random_rotation(generator):
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    return q


def rotation_error_deg(R_est, R_true):
    rel = R_est.T @ R_true
    cosine = ((torch.trace(rel) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def run_case(dynamic_ratio, seed, n_points=256, motion_scale=8.0):
    generator = torch.Generator().manual_seed(seed)
    R_true = random_rotation(generator)
    t_true = torch.randn(3, generator=generator)
    src = torch.randn(n_points, 3, generator=generator) * 2.0
    tgt = src @ R_true.T + t_true

    n_dynamic = int(round(dynamic_ratio * n_points))
    motion = torch.zeros(n_points)
    if n_dynamic:
        idx = torch.arange(n_points - n_dynamic, n_points)
        displacement = torch.randn(n_dynamic, 3, generator=generator) * motion_scale
        tgt[idx] += displacement
        motion[idx] = displacement.norm(dim=-1)

    R_base, t_base = umeyama(src, tgt)
    R_dyn, t_dyn, _ = dynamic_aware_umeyama(
        src, tgt, confidence=torch.ones(n_points), motion=motion, robust_iters=3)
    return {
        "base_rot": rotation_error_deg(R_base, R_true),
        "base_trans": float((t_base.squeeze(1) - t_true).norm()),
        "dynamic_rot": rotation_error_deg(R_dyn, R_true),
        "dynamic_trans": float((t_dyn.squeeze(1) - t_true).norm()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--points", type=int, default=256)
    args = parser.parse_args()

    print("ratio | baseline rot/trans | dynamic-aware rot/trans")
    print("------|---------------------|-------------------------")
    for ratio in (0.0, 0.10, 0.25, 0.50):
        rows = [run_case(ratio, seed, args.points) for seed in range(args.seeds)]
        mean = lambda key: sum(row[key] for row in rows) / len(rows)
        print(f"{ratio:>4.0%}  | {mean('base_rot'):>6.2f}° / {mean('base_trans'):.3f}"
              f"       | {mean('dynamic_rot'):>6.2f}° / {mean('dynamic_trans'):.3f}")


if __name__ == "__main__":
    main()
