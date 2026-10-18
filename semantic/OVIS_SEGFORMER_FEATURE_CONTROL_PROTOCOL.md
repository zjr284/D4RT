# 第82轮：分割预训练×精度的冻结查询特征实际控制

第81研究d01ff78已核验推送maindb49d032后立即开始。官方MIT/ADE权重可用，但BF16源RGB预检存在T1/T8数值差；因此在任何新GT成绩之前固定FP32/BF16两种算术，不能按确认成绩挑精度。只原OVIS80train/20反复val，不新确认、不读新COCO/LVVIS，不改默认部署。

六新feature arm：mit_encoder_fp32/ade_encoder_fp32/ade_decoder_fp32及对应bf16，各seed0/1/2=18线性头。三分支取81严格加载的同官方checkpoint：ImageNet MIT最后stage256×14×14，ADE相同encoder，ADE原classifier前融合256×112×112。源JPEG448/PIL bilinear/CPU FP32÷255/ImageNet normalization不变，无GT类/掩码/裁剪/文本提示。模型权重始终FP32冻结，算术仅autocast BF16开/关，两种均关闭matmul和cudnn TF32；原81冻结helper不修改，新增特征模块。无额外层归一化，128像素中心bilinear border align_corners=False→FP16查询缓存→FP32头。

同80首源帧GT20480查询（每视频128FG128BG）、T1训练/T8验证，256维26类头6682参数、2000步AdamW .001/.01、每步16FG16BG、64000曝光，seed与完整逐批索引/类直方图必须同58 B14_GT32。复用B14_GT32三个768维19994参数头/旧结果作完整方法参考，不把与其差异仅归架构/特征网格/头容量；ADE额外密集标签与ImageNet监督预算不同、预训练OVIS/COCO重合未知披露。

主要描述对照同FP32 ADEencoder−MITencoder在原balanced val的raw宏F1（三种子）；同架构/网格/头/采样，比较额外分割预训练完整checkpoint效果，不证明某一标签或层唯一因果。次对照FP32 ADEdecoder−ADEencoder、三个分支BF16−FP32、完整方案对B14，保存balanced与spatial两val集合raw/两种校准的全部正负seed，不以某一个口径最高叙述。

每50步只balanced raw val宏选checkpoint早同分；原balanced40960查询与78空间val40960查询都原样冻结、不按GT新筛点，各组在其各自val集合按原−8..8步.25/同分abs小负优先选三seed共享背景偏移。raw0始终保留，不混两个查询任务或用确认选profile。B14旧balanced−2/spatial+.25与raw输出直接复用并独立重算核验。

源val候选顺序B14旧基线/三FP32/三BF16，按balanced-profile在balanced val的三seed宏均值选，仅探索、不部署；若没有新组优于B14，不为证明新方法而消费预留确认。模型选择只源验证，反复适应限制保留。

先冻结本协议/代码、81资源/隔离环境/两权重、原100查询GT元数据与源RGB、78空间查询与旧B14 raw输出指纹，再提特征/训练。验证258 BF16新缓存每32取一须复现81已核验64点；所有720组feature package的标签/XY/时刻不变，FP32/BF16同权重且精度算术记录。独立首训练562/首验证258手动encoder+ADE多尺度融合重算六分支及两val坐标、共享三采样轨迹核验18头预算/checkpoint、42原始输出逐值/12新共享偏移曲线/主次对照/选择与旧指纹。保存所有负结果，数据/缓存/权重/第三方不入Git；独立核验报告逐轮提交推送后当前运行立即下一必要研究。
