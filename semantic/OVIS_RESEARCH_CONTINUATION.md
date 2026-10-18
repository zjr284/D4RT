# 连续研究当前接续点

用户要求每轮复盘、独立核验、独立Git提交并推送GitHub main后，当前运行立即下一轮。现有分钟心跳仅为空闲/中断恢复。先检查最新聊天消息、semantic最新报告/协议、Git状态及实际进程/结果，避免重复训练、覆盖历史或启动两个研究运行。

第80研究bb91fb7已推送a603db5e：新COCO40唯一主aux空间cal旧/raw/新宏.595823/.637919/.644733；新−旧三seed正、图像+.03231/CI[.01161,.05278]/28正12负；新−raw两正一负、图像+.000888/CI[−.006484,.008105]跨零。BG误报16.12→5.06%，FG漏BG5.10→20.36%，不自动部署或按extraGT排名改选，COCO共200已评分。第81官方MIT/SegFormer-B0资源已独立核验：MIT commit80983a413c30d36a39c20203974ae7807835e2b4、14380029B/SHA4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc；ADE commit489d5cd81a0b59fab9b7ea758d3548ebe99677da、15036944B/SHA6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9。3权重GET/20全HTTP含元数据，官方匿名TLS/不记录签名query/不读token；MIT weights_only与ADE safetensor严格加载。私有env semantic_artifacts/segformer_resource_control/env，transformers4.57.6/hub.36/safetensors.8，父.venv hub2/Torch2.14+cu130版本文件不变。源258仅RGB预检/不读GT指标：MITencoder/ADEencoder256×14×14、ADEdecoder256×112×112，独立手动融合与64点逐值复算通过；BF16 T1/T8首帧max差.192312/.132193/.046875，不能声称跨批相同或速度比较。报告OVIS_SEGFORMER_RESOURCE_RESULTS.md，NV非商业研究许可/ADE额外密集监督与重合未知披露，模型第三方不提交。提交推送后立即82实际对照训练：新独立脚本不改81冻结helper；三个分支×FP32/BF16×seed0/1/2共18头，同80源20480GT查询/16FG16BG/2000步64000曝光/AdamW .001/.01/256维6682参数，B14_GT32旧768/19994三头作完整方法参考。原JPEG448/T1训练T8val、各层网格14/112，关闭matmul和cudnn TF32，精度由RGB预检预注册而非确认分数挑。主FP32 ADEencoder−MITencoder（同架构/头/网格，额外ADE训练完整差异），次decoder−encoder/三个BF16−FP32及对B14；balanced raw val每50步选checkpoint，balanced与78空间val共享偏移各按原grid独立选、raw0全保留，不按新确认混profile。两val集查询原样、所有18seed负结果/采样hash/独立特征与42输出审计后提交推送。COCO79另reserve40仍元数据未RGB/评分、LVVIS状态以原协议为准，不消费确认调参数，默认nativeGT−1.5不变。

第37有效COCO image oracle：19096图像/79604 RLE注释/1196类，23严格共同词表但第43核验仅14实际注释类，固定8帧Person87.41%，Bear等五类仅1目录；不称完整25类泛化。缺Poultry/Vehical；未知/重叠/crowd GT忽略，始终26预测输出不依GT限类。普通VIS JSON解析失败保留。无官方video_id/track_id/fps，路径目录作组。第41固定48帧24空保留。第42全部42149 OVIS原帧索引/文件SHA与固定48像素核验通过，完全相同组0，但外部近似重复仍须盲复核。pilot00000永久排除，元数据/RGB已观察、辅助SAM标注与预训练重合未知需披露。

OVIS607段均用于研究，无剩余独立视频。第33验证选中GT32的新27段主对照视频区间跨零，一个种子下降；第35GT32实例准确率三种子都低于GT256，宏小幅上升集中少数类别。SAM在第33/35局部正收益与第24负收益都保留，不能一概有效/无用，不按确认最高quarter改选。

外部准备固定GT256/GT32/lambda0/nativeSAM四组三头及源校准偏移−1.5/−1.25/−1.5/−1.75。GT32选择来自此前OVIS验证；目前部署仍第22native_gt（GT256）。不训练/重校准、不把GT类别/掩码输入RGB特征或裁剪、不过滤预测候选至GT共同类别。后续新模型评价前必须另固定真实图像/重复筛查/候选确认及预留集合/查询和完整recipe；元数据已检查、标注工具含SAM、骨干预训练重合未知的限制保留。

标准文件读取工具在当前Linux会话可用；Git写入、网络需按当前权限经批准执行。临时发布脚本/tmp/d4rt_publish_research.py必须先检查，保护远程历史正常合并、不强推，复用Windows Git凭据助手不读取输出令牌。研究白名单自有代码/报告/文本快照；不上传数据、缓存、模型、第三方源码，保持用户README/论文/PPT整理不变。

当前连接未提供automation_update调用，已检索可用工具，不能宣称已更新调度或手写修改应用配置。以本文件和最新semantic结果为接续依据；工具恢复后再通过正式接口更新现有d4rt，勿新建重复调度。用户暂停/停止立即停止实验。无变化时安静，仅有意义结果/完成/失败需用户处理时通知。不付费，不公开发文，不联系他人。
