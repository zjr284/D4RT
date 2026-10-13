# 已完成实验的结果快照

这些 JSON 是本地完整实验汇总的提交快照，不包含数据、特征或模型权重。

- `background_ablation_summary.json`：第二轮背景伪标签对照。
- `sampling_ablation_summary.json`：第三轮时间/类别采样 2×2 对照。
- `oracle_probe_summary.json`：第四轮真值监督诊断。

协议、结果解释和复跑入口见上级目录 `OVIS_*_PROTOCOL.md` 与 `OVIS_*_RESULTS.md`。JSON 中的 `semantic_artifacts/` 和原本机绝对路径表示原实验产物位置；重新运行时会在本机生成对应文件。首轮基线指标也保留在第二轮快照中。
