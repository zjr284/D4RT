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

- `frozen_error_audit_summary.json`：第十轮冻结输出错误分解及独立训练视频覆盖审计。

- `video_coverage_intervention_summary.json`：第十一轮等80段随机/覆盖训练集×纯GT/SAM四方案、三种子及新40段确认结果。
- `video_coverage_intervention_verification.json`：第十一轮全部查询、24份权重预算、24份输出及验证选择的独立核验与哈希。

- `transfer_error_audit_summary.json`：第十二轮冻结模型的源帧GT拟合、共同14类、源/后续帧错误分解及逐类跨视频诊断。

- `semantic_feature_control_summary.json`：第十三轮D4RT冻结查询特征与DINOv2局部图像特征、三种子及40段模型未评价确认。
- `semantic_feature_control_verification.json`：第十三轮真值查询重建、采样匹配、六模型输出复算和源/后续帧诊断。

- `feature_coverage_control_summary.json`：第十四轮D4RT/DINOv2×随机/覆盖80视频训练集的2×2对照和新40段确认。
- `feature_coverage_control_verification.json`：第十四轮查询、覆盖、十二模型输出、旧模型冻结和新头选择核验。

- `dino_sam_control_summary.json`：第十五轮固定DINOv2的源帧GT/SAM监督、两训练集三种子及新40段确认。
- `dino_sam_control_verification.json`：第十五轮教师查询、源帧特征前缀、十二模型输出、采样预算及类别视频覆盖核验。

- `dino_interface_probe_summary.json`：第十六轮三个种子的真实几何/语义接口、时间一致性、输入精度及运行成本。
- `dino_interface_probe_verification.json`：第十六轮案例闭合、帧/缓存/头/骨干指纹和运行环境记录。
