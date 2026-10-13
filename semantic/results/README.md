# 已完成实验的结果快照

这些 JSON 是本地完整实验汇总的提交快照，不包含数据、特征或模型权重。

- `background_ablation_summary.json`：第二轮背景伪标签对照。
- `sampling_ablation_summary.json`：第三轮时间/类别采样 2×2 对照。
- `oracle_probe_summary.json`：第四轮真值监督诊断。

协议、结果解释和复跑入口见上级目录 `OVIS_*_PROTOCOL.md` 与 `OVIS_*_RESULTS.md`。JSON 中的 `semantic_artifacts/` 和原本机绝对路径表示原实验产物位置；重新运行时会在本机生成对应文件。首轮基线指标也保留在第二轮快照中。

- `head_budget_probe_summary.json`：第五轮线性/非线性分类头和500/2000步预算诊断。

- `weaklabel_budget_probe_summary.json`：第六轮两种伪标签、分类头与500/2000步预算的复核。

- `background_calibration_summary.json`：第七轮仅验证选择的背景分数校准。

- `source_gt_control_summary.json`：第八轮同预算、同校准规则的纯首帧GT公平对照。

- `fresh_video_confirmation_summary.json`：第九轮40段未使用视频的冻结确认（保留负结果）。
