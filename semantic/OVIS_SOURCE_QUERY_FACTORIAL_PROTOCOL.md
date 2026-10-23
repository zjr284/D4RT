# 第105轮：冻结几何主路的源查询×SAM匹配训练

第104预检独立发布后执行。本轮只用原60训练/20验证，607已研究、非新确认；不评分原test20、外部40，不换默认。新结构只适配独立语义decoder，原几何主路完全冻结；不把它称为微调整个几何网络。

全部使用原sample_queries生成的u=x/127、v=y/127、t_src和原t_tgt/t_cam，重新从GT/teacher与seed42+videoID重建，标签/xy必须逐值旧缓存。沿旧99三seed各500步GT64/extraGT32/SAM32日程，点数/类别监督不重选。raw与source两种prefix从同一原JPEG128、同一次BF16视频memory、原冻结前六层生成，唯一查询字段差别为source的t_tgt=t_cam=t_src；raw保留原字段。双方均重新提特征，不混旧raw与新source的数值路径。对旧memory/prelast差只描述并保留指纹，不据数值差挑方案；坐标/类别重建失败则记录停止，不静默放宽。

六臂三种子：raw_GT、source_GT、raw_SAM、source_SAM、source_head、source_extraGT。除source_head仅训练头外，其余独立语义tail的FF rank8/alpha16 LoRA+1280→26头训练。所有臂同seed初始、同500步GT64，头lr1e-3/LoRA1e-4/AdamW .01，eval模式关dropout但允许梯度。GT臂损失CE_GT；SAM臂加.25*CE_SAM；extraGT臂加.25*CE首帧额外FG。所有臂同128候选点前向。几何主路不经语义tail，不添加限制独立语义tail几何的保持损失，所有臂一致；与102比较同时改变了查询结构/几何隔离/约束，不能唯一归因source canonical。

主source_SAM−source_GT；必要source_GT−source_head；canonical对照source_GT−raw_GT、source_SAM−raw_SAM；额外监督source_SAM−source_extraGT；保存SAM×canonical差中差。完整六臂18终点权重冻结后才评价原20验证raw宏/前景F1、逐类/视频、时间变体一致率。raw时间变体按原t_tgt+1/t_cam+1，source分支不读这两字段；保存对应输出，不把高一致率当准确率。

真实接口另验证18权重×固定原val数值ID首两段B1及B2×原/目标相机+1变体，共108例。几何与同输入/批量冻结base逐值相同；B1语义分数须逐值复现本轮缓存评价；B2只与同批手工参照核验，不要求跨批逐值一致。source语义原/时间变体须逐值相同，raw只记录。视频memory仍全片，不称只看源RGB。独立语义tailFP32原冻结参数，base BF16，均BF16 autocast。固定原权重指纹与存储隔离，不改原几何路径。

外测资格只在source_SAM对source_GT与source_extraGT三个seed汇总宏差均正且各自视频等权均差正时满足；这只支持另立新外部协议，不是泛化或部署结论。所有负类/负视频保留，不校准/早停/搜索预算。独立核验GT/query重建、raw/source全部新特征/旧数值差、18初始化/预算及代表梯度/末优化器、18完整验证/108接口与指纹，写自有报告后单独发布。
