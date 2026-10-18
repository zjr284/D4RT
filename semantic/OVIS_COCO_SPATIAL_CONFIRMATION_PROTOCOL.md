# 第80轮：冻结空间校准在新40COCO图像的首次确认

第79研究45117ee已核验推送main3876bc67后立即执行。仅79已固定合格candidate40，排除旧160，先模型/新旧偏移冻结、合法RGB与重复筛查及单Codex盲复核后首次GT/特征/评价。reserve40只元数据、LVVIS40不动，未按新GT有效/成绩补图。源5000类别元数据与候选RGB已见，模型未评价；预训练COCO重合未知，不能称完全未见标注/自然全分布/官方AP/视频结论。

沿79预注册，七方案21头全部原样复用，参数对每组三seed共享：原source平衡查询旧偏移、78仅OVIS空间val选定新偏移、raw0。主要模型固定B14_auxSAM（源71选定，非COCO挑选）；唯一主要strict图像等权present宏F1差=同头新−旧，先seed0/1/2平均每图差，再seed20261080/2000次图像bootstrap95%区间，汇总查询宏差另列而不混区间。重要预定次同头新−raw0，其余六新−旧/新−raw共12均未多重校正；consensus敏感性不冒充额外确认。BG误报、FG漏成BG/类间错误、accuracy/FG宏/空项/全部负图种子都保留，不按最高方案/口径改选部署或偏移。

每图np.default_rng(20261077+imageID).choice(16384,1024,False)不依GT固定唯一XY，所有点提特征/预测，unknown/歧义/crowd仅评价时沿既有strict主/consensus敏感性忽略；BG0/Person1/同名映射FG保留，26预测类别不缩减。官方COCOAPI polygon/RLE定义沿68/77，严格同名映射不做文本教师或新类别映射。

原JPEG RGB/PIL bilinear448 CPU FP32÷255、ImageNet归一化/BF16官方冻结S14/B14最后32×32normalized patch384/768、像素中心bilinear border align_corners=False、FP16特征→FP32线性头、TF32关闭、sourceT1/image_batch1。无GT裁剪/提示、target/camera或重新训练/校准，不改原D4RT部署。先直接登记21头/两骨干/全部vendor与依赖、79资格/实际图像/源78选择哈希，再GT/预测。

独立80套bitset GT与40组固定XY/支持、两骨干首图完整1024特征、21原始logits/双口径三offset共126指标及各图、14对比×2口径共28bootstrap与新旧参数溯源/全部输入指纹；纯BG/全忽略指标None保持。报告不夸大以旧弱偏移为基线的提升，必须同时评估raw0和FG召回代价。不训练/重评旧确认，不自动改默认部署；保存自有代码/协议/结果/精简文本快照，逐轮独立提交推送，当前运行立即下一必要研究。
