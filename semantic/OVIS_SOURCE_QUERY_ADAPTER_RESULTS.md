# 第104轮：独立源查询语义支路预检通过

新独立入口SourceQuerySemanticD4RT已用真实公开复现权重完成预检及独立核验。原几何主路冻结并使用用户原query；语义支路共享视频memory，把t_tgt/t_cam固定为t_src，仅复制的末两层FF LoRA与类别头可更新。**这只保证本轮接口约束，不是识别准确率提升或几何真值精度测试。** 当前默认仍原native_gt入口，没有部署替换。

固定训练13/19、每段64点、8帧128，B1两例及真实B2。原/目标相机变化/源时间变化/aspect四变体，更新前后共24个案例。第一段64GT固定5步CE，head_lr1e-3/LoRA_lr1e-4，217626参数更新。全部案例六几何与相同输入/批量原base逐值相同，也在前后逐值相同；目标/相机语义分数在B1/B2、更新前后均逐值不变。

语义支路和几何base参数存储不共享，base所有梯度None、全参数指纹不变，指定LoRA与头确实变化。独立脚本未调用包装入口，手工构造canonical query、前六层和语义tail，重建全部数组；首梯度和最后AdamW一步的最大重放误差均0，输入/检查点指纹未变。

本次峰值已分配2489.88MiB、预留2686MiB，包含完整BF16 base及FP32语义tail；复制冻结末层增加常驻和解码开销。顺序计时包含adapter与base参照，不是标准速度排名。只测8帧128和固定64点/B2，不能外推任意视频/查询规模。

语义仍依赖整段视频memory与aspect，不是仅源帧RGB。预检点采用(u+.5)/128像素中心约定，与原缓存sample_queries的u/127约定不同，本轮没有声称旧缓存分数复现。后续训练需固定沿用原u/127及原查询时间，raw/canonical在同轮从相同video/memory重新提特征，避免坐标变化被当成结构收益。

原几何主路保持相同预测，不等于这些预测相对真实三维标签准确。独立语义LoRA属于新增解码支路的适配，不冒称微调整个原几何网络。没有使用SAM监督或产生F1结果，不能由五步CE下降推断泛化。

代码source_query_adapter.py、source_query_adapter_probe.py与verify_source_query_adapter_probe.py；聚合/核验results/source_query_adapter_probe_summary.json、source_query_adapter_probe_verification.json；原图、权重和输出数组本地保留。下一轮固定匹配训练比较raw/canonical×GT/SAM，并包含canonical头-only和等额GT前景对照；所有几何主路冻结，旧60/20仍仅探索，不消费旧test20或外部40。
