# 第二十三轮：原JPEG语义模型的来源一致接口验证

第二十二轮提交后立即进入本轮。冻结其按验证选择的native_gt三头与−1.5；不按确认改选native_sam，不训练/校准、不评价新视频或最终40保留。

新增独立NativeDinoSemanticD4RT及load_native_selected_adapter，旧224/128→448入口不改。新batch使用原geometry video `[B,T,3,128,128]`及原query，另提供source_frames（B套T个原RGB帧路径）。每个原帧只解码一次，分别PIL bilinear生成128与448两份RGB，CPU float32除255、连续NCHW。128版本量化为传入video.dtype须逐值等于video，否则拒绝；不能拿不同画面充作高精度语义源。不接受混入旧semantic_video字段以免歧义。语义用同源448 RGB、32×32patch、既定像素中心采样/量化/FP32头，原batch直接交原D4RT。仅读原RGB，不读GT/类别提示；语义仅u/v/t_src。

固定检查：两视频替身单元验证来源、几何字段、源时间/坐标依赖、tgt/cam不变、非法/缺失/错误顺序路径、混合旧字段拒绝、重载与输入不变。真实第22已有40段全部2048查询×三种子B=1；与旧native_gt缓存/输出/原F1比较，原/改变t_tgt+1 t_cam+3模8六几何同输入逐值保持，语义时间逐值不变。真实B=2固定前两段345/574，2048查询×三种子，语义与对应B=1比较。不假设跨批几何相同。

记录完整前向单次顺序耗时、CUDA峰值（**包含原JPEG读取及来源校验**），与仅GPU原几何计时有范围差异，不作标准速度排名；失败/OOM记录，不缩减预注册范围。来源路径读取带来CPU/磁盘成本，先做正确性再优化。现有224/448默认接口只跑必要旧单元回归，不重复旧40段评分。

保存固定代码、帧、旧缓存/输出/三头/模型/报告指纹，独立重算245760个B=1及12288个B=2类别比较、指标、几何计数、保留集合未消费。提交仅自有代码/协议/报告/文本快照。通过后复盘，冻结模型/偏移再设计最终保留40一次独立确认；不能用最终集合选参数。

入口`.venv/bin/python -m semantic.dino_native_interface_probe`；产物`semantic_artifacts/dino_native_interface_probe/`。
