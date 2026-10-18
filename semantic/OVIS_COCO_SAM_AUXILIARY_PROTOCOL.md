# 第72轮：冻结SAM辅助监督的预留COCO图像确认

第71研究2ee2f73已独立核验并推送GitHub main cf9a6e8a后当前运行立即接续。本轮仅使用第65固定、第67像素/近似重复盲复核保留的COCO reserve80，顺序不变、无补选/排除；这80张元数据/原RGB已见，但从未构建评价查询、提模型特征或评分。旧candidate80已在第68评分，不能再叫未见；本轮不复用其评分/查询来选方法。LVVIS reserve40继续不评价。预训练COCO重合未知、图片可能同来源；是非Person富集静态面板确认，非官方AP、自然像素/视频/全OVIS结论。

在任何本轮GT构建/预测之前，固定7方案×3seed权重及源验证偏移：B14_aux_sam/extra_gt均−2.25（71）、B14_gt32−2（58/71）、B14_weak−2.75（69/71）、B14_gt256−2（58）、B14_class_video−2（61）、S14_gt32−1.25（旧32/68）。source val在预定weak/aux_sam中已选aux_sam，匹配计算/人工GT预算对照为extra_gt；此选择不是按第68旧COCO排名。oracle额外GT方案禁止进入本轮。当前部署native_gt保持，不按新确认最高方案切换。

唯一主要比较B14_aux_sam−B14_extra_gt：均2000步/19994参数/128000抽样曝光/同64000基础GT索引，补充池时间/坐标不同，属于完整追加方案。四个预定未校正次比较：aux_sam−B14_gt32（保留GT锚但计算曝光翻倍）、B14_weak−B14_gt256（原传播recipe512000曝光）、aux_sam−weak（完整recipe不同/预算不同）、B14_class_video−S14_gt32（第68容量完整方案复现）。次比较不据最高结果改选；主对照为较弱追加GT基线的限制必须结合aux_sam−GT32报告。全部三种子/负图保留。

沿第68已核验的官方COCOAPI polygon/RLE、严格同名OVIS映射与strict主/consensus敏感性定义；GT只构造评价点与标签，不输入特征或头。任意实例重叠/未知/crowd在strict忽略；consensus仅唯一已知语义且无crowd有效。每图每有效非Person类64点，seed20261052+image_id*32+class_id，不足有放回、记录不同像素数；空图保留、不换图。Person/BG不采点、全部26预测类别保留，不按GT限预测。

原JPEG单帧T1/PIL RGB bilinear448/CPU FP32÷255/ImageNet归一化/BF16冻结官方S14/B14最终normalized32×32patch384/768，128像素中心bilinear border/align_corners=False、FP16缓存→FP32线性头，TF32关闭，image_batch1；不按确认分数调预处理/头/校准，不输入GT裁剪、类别提示或目标/相机时间。冻结实际骨干、21头、源选择报告、所有vendor源码和相关依赖/图片指纹。

主要判断strict主对照的图像等权（先三种子平均）差与seed20261072/2000次图像bootstrap95%区间；另列汇总查询宏差/每种子，但不能混成汇总宏区间。consensus+四次对照均完整保存且未多重校正；不能凭敏感性最高口径宣称通过。先保存protocol JSON/manifest再GT/特征/评分。

独立重构160张图口径GT/查询、两骨干各口径首有效图完整特征、42份logits/全部指标/10个图像配对bootstrap、旧模型/骨干/偏移与新缓存指纹；不重复训练。完成后代码/协议/报告/文本快照独立提交推送，不提交数据/缓存/权重/第三方源码。失败/负结果如实保留，LVVIS40不消耗，当前运行立即推进下一项必要研究。
