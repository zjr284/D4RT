# 第106轮核验范围修订：完整LoRA轨迹未通过，保留冻结结果核验

原verify_d4rt_input_resolution.py及失败日志不覆盖：80套RGB/query/特征逐值重建和seed0仅头500步通过，source_GT第7步严格loss比较失败。两次原配置临时克隆在第1步参数已不同，第8步最大差4.47e-6，且实际经过FlashAttention backward。此诊断证明该运行训练路径存在重复数值差异，不能把一个种子的训练轨迹完整重放当作既定事实。

[PyTorch SDPA文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)说明CUDA实现可能涉及非确定性，[确定性算法接口](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)提供约束开关；文档仅是背景，实际诊断才是本轮证据。没有进一步把原因唯一锁定为一个算子，也不估计对最终F1的敏感性。

新增recovered核验程序明确缩小训练重放声称：三仅头模型继续逐步重放500步；九LoRA模型仅从保存初始/末步前状态独立重放首梯度与末优化器，保持原5e-4/5e-6梯度、3e-4/2e-6参数容差，不扩大容差。所有12份原500步日志逐行检查序号/有限/损失代数/预算；LoRA中间498步损失不称独立重算通过，完整LoRA轨迹标记未验证。未通过原严格完整重放的事实进入最终报告和verification，而非从记录中消失。

其它要求保持：全部80输入/特征和12新完整验证分数、72同输入几何接口、12旧模型复用、全部对照/资格与指纹仍须逐值/原标准通过。所有模型、分数、训练程序及资格规则不改、不重训；不得因训练重放范围修订产生新的分类改进结果。

verification_scope_amendment.json在新核验前绑定原失败代码/日志、两次原配置诊断、确定性诊断及所有原输出指纹和新核验代码。本轮结果只能报告此真实核验范围。未来训练需在协议中显式固定确定性设置并实测，不能追溯修改106结果。
