# 第68轮：冻结语义分支的COCO静态查询确认

第67全部候选/暂留RGB完成且独立去重/实际盲复核提交推送后才能启动本轮。只用原65 candidate80经盲复核保留者，空图/类不补选；reserve80和LV-VIS40暂留不提取模型特征/预测。COCO元数据及RGB已观察、未根据候选有效mask/预测选择方法；本轮为非Person元数据富集静态图像上的确认，不是全图像像素/官方AP/视频跟踪/几何结论。预训练COCO重合未知。

全部7方案×种子0/1/2权重/源验证偏移沿65：S14_gt256−1.5、S14_gt32−1.25、B14_gt256/B14_gt32/B14_class/B14_class_video−2、S14_native_sam−1.75。主比较源验证最新所选B14_class_video−旧所选S14_gt32；预定未校正次比较B14_gt32−S14_gt32、class_video−B14_gt32、B14_gt256−S14_gt256、S14_native_sam−S14_gt256。不训练/重校准、不按候选改选/部署，SAM对照是原完整传播监督方案，查询分布亦不同，不能纯归因标签噪声。

官方COCO API annToRLE统一解码polygon/未压缩或压缩RLE，GT仅定义评价点/标签，不传给骨干或头。strict沿原任意实例重叠/crowd/未知忽略，为主；consensus沿唯一已知语义/未知不抹已知、异已知类或任意crowd仍忽略，仅敏感性。严格同名OVIS映射，Person/BG不采评价点、26预测全保留。每图每有效非Person类64点，seed20261052+image_id*32+class_id，不足有放回，空项保留；记录不同坐标数，不把重复点当独立样本。

每图仅真实源RGB单帧T1，原JPEG PIL RGB bilinear448/CPU FP32÷255/ImageNet normalization/BF16冻结官方S14/B14、32×32最终normalized patch384/768；128像素中心bilinear border/align_corners=False、FP16缓存→FP32头，禁TF32，无GT裁剪/类别提示/目标/相机输入。不按COCO分数改变T1为T8或其他实现。直接登记实际S14/B14权重及全部vendor源码和运行依赖指纹，先冻结再GT/特征/评分。

分别保存strict/consensus全部21份logits、三种子present-GT类宏F1/每类/每图结果，主判断strict主对照的图像等权（先三种子平均）差及seed20261066/2000图像bootstrap95%区间；另保存汇总查询宏差但不混其区间。consensus与四次对照均完整保存、不按高分挑口径；图片可能相关来源/同摄影事件，有限像素哈希不能证明完全独立。

必要合成检查覆盖polygon、uncompressed crowd RLE、ID0、未知/异类冲突、两口径有效像素保持；独立从原GT重新构建全部标签/查询、两骨干各口径首有效图完整特征、42份输出/指标及10配对bootstrap、全部新缓存/旧模型/骨干/偏移指纹；完成才报告与独立提交推送。失败/负种子/被盲筛排除图保留，不提交图片/GT/特征/权重或第三方代码，部署native_gt不变。
