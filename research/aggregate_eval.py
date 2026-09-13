"""汇总同一固定评测设置下的多随机种子模型结果。

运行示例:
  .venv/bin/python research/aggregate_eval.py checkpoints/repro_v2_seed*/eval_seed42_baseline.json

这里只统计各模型在同一组测试场景上的平均指标，再对训练随机种子计算
均值与样本标准差；不是论文公开数据集的评测协议。
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="汇总固定评测场景下的训练种子结果")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.results]
    reference = payloads[0]
    fields = ("seed", "samples", "grid", "tracks", "dynamic_aware",
              "checkpoint_step")
    for path, payload in zip(args.results[1:], payloads[1:]):
        differences = [name for name in fields if payload.get(name) != reference.get(name)]
        if differences:
            parser.error(f"{path} 的评测设置不同：{', '.join(differences)}")

    print(f"评测种子={reference['seed']}，场景数={reference['samples']}，"
          f"模型数={len(payloads)}，动态感知={reference['dynamic_aware']}")
    for name in reference["mean"]:
        values = [p["mean"].get(name) for p in payloads]
        values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not values:
            print(f"{name}: 无有效值")
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{name}: {mean:.6f} ± {std:.6f} (n={len(values)})")


if __name__ == "__main__":
    main()
