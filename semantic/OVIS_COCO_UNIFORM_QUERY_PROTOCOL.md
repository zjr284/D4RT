# 第77轮：已见COCO图像的均匀空间查询描述性诊断

第76研究82c72ba已核验推送main27afa7fe后立即接续。只旧65/67 candidate80（68已评分），不是新确认，不消耗LVVIS40；不读reserve80新模型、不训练/重校准/选部署。原COCO主要类平衡前景点没有评估BG/Person/空间频率，故本轮补足使用范围。

预先每图np.default_rng(20261077+imageID).choice(16384,1024,replace=False)，转128网格(x,y,t_src0)，不依GT采样/删除点；所有1024点都提特征/预测，包括unknown。每图仍RGB原JPEG448单帧T1，S14/B1432×32normalized patch384/768、像素中心bilinear border align_corners=False/FP16量化/FP32头，不变官方权重/vendor/预处理。

冻结第72七方案21头/源偏移：B14_auxSAM/extraGT−2.25、GT32/GT256/class_video−2、weak−2.75、S14GT32−1.25。不包含75/76新失败头，不按旧确认排名调；七方案选择是72预注册的固定对照全集，不新命名最佳。GT只评价既有strict主/consensus敏感性：BG0、Person1、同名映射FG保留，unknown/歧义/crowd依既有定义为−1，仅算指标时忽略，模型预测仍26类。两口径使用同一XY/特征/logits，不用GT限输出类别。

完整保存每图支持/忽略/空项、全部三seed raw和固定校准的条件有效像素accuracy、present-GT宏F1、FG宏/FG→BG/FG类间错误、BG→FG误判率及预测BG数量。另提供非Person FG子集用于了解类平衡差异，但其分布/支持与旧前景采点也不同，不把F1差纯归因模型退化。配对比较CV−S14GT32、auxSAM−extraGT、auxSAM−GT32、weak−GT256，只描述同旧图集，不做新确认或按结果改阈值。

uniform XY是在128方形变换网格上的抽样，GT有效条件/非Person元数据富集/未知与未注释限制仍存在，因此不是完整自然像素分布、官方AP或泛化性能。骨干预训练COCO重合未知、图片可能相关，三seed/空图/负例全保留，不把raw vs校准差据为重新校准授权。

先冻结协议/源码/原21头源记录/实际图像/两骨干/全部vendor/相关依赖与76核验指纹，再GT/特征/预测；保存本地全部XY/两GT目标/特征/21logits。独立160GT bitset重构/不依GT固定XY与1024唯一点、两骨干首图特征、21logits与两口径raw/cal/误判/所有支持和配对指标/旧指纹，核验通过后自有代码/报告/文本聚合逐轮提交推送，数据/缓存/权重不入Git；当前运行立即下一必要研究。
