# 第76轮：同维度冻结B14最后四层均值特征控制

第75研究98c6d78已核验推送main5466a85b后立即开始；不继续梯度投影参数搜索。此前B14完整方案两个静态COCO面板优于S14，但SAM辅助确认未复现。本轮预定一个表征控制，不按旧确认选择层/分辨率/参数，仍只原80train/20反复val，不新确认/不耗LVVIS40/不改默认部署。

固定官方ViT-B/14源commit7764ea0与同一0b8b82权重、原JPEG448/T1首帧训练/T8验证。旧final归一化patch768及第58 GT32三头/共享偏移−2直接复用。只新mean_last4三个线性头：官方get_intermediate_layers(n=4,norm=True,reshape=False)取最后四block，各沿官方同一个norm；将各输出转FP32后逐token算术平均，无再归一化/拼接/类别/GT裁剪或提示。位置查询仍128像素中心bilinear border/align_corners=False→FP16缓存→FP32头，768维/19994参数均保持。不得根据模型分数选择不同层组；本控制是完整均值特征方案，不能唯一归因某层。

同原GTsource20480点/80视频、每步16FG16BG、2000步64000曝光、AdamW .001/.01、seed0/1/2。新组完整采样索引SHA/类直方图必须同58 GT32。每50步raw val宏选checkpoint早同分，三种子共享偏移旧grid−8..8步.25同分abs小负优先。主要mean_last4−final，全部种子负结果保存。固定合法顺序final/mean_last4按源val均值选，不按COCO排名调/部署；比旧全局auxSAM仅描述，不当新泛化。

先冻结协议/源码/官方vendor/权重/原100组query缓存和全部实际RGB/旧头/源验证报告指纹。每个新cache同时用get_intermediate_layers最后一项采样，必须与历史final缓存逐值相同（所有100组），避免把另一个预处理/批量差误当多层效果；失败保留、不静默放宽或改历史。记录均值特征dtype/提取时间/显存、查询/标签逐值相同。

独立核验100组meta/GTsource仅0、最后层等同证明、首固定训练562/验证258分别用手动forward hooks取最后四block、官方norm后FP32均值+单独插值复算新特征，不能直接调用新缓存函数镜像；三采样轨迹/预算/checkpoint、六val logits逐值/混淆/校准/主差/选择/旧指纹。完成自有代码/协议/报告/文本聚合独立提交推送，数据/特征/权重/第三方代码不提交，当前运行立即下一必要研究。
