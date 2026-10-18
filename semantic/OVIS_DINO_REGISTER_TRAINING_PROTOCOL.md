# 第90轮：普通/register B14 × raw/训练标准化的同预算2×2

第89研究763fbf7已独立核验推送main88b17c76后立即执行。资源通过不代表分类有效，仍只用原80首帧GT/20已反复源val；普通两组三seed6头及偏移复用83，只新训register raw/z各seed0/1/2六头。无默认替换，原几何不变。

先固定本代码/协议/官方骨干权重与源码/80训练160验证源RGB/原查询/原六头、源校准规则。另在任何新骨干特征提取或训练前，按seed20261090固定未来COCO candidate80+reserve80：排除此前模型评分240（原160+79candidate40+79reserve40），5000ID排序后一次排列；14非Person严格共同类按ID两轮取非crowd/area>0且未选首项，然后同排列合格池补足80，reserve同法排除前80。只是已见元数据预留，无新RGB/GT解码/评分；不存在“完全未见标注”声明。

register配方沿89官方B14_reg4：4tokens、官方另权重、antialiasTrue/offset0，BF16冻结；普通B14无register/False/.1。原JPEG RGB直接bilinear448、固定normalized32×32patch768、像素中心/128 bilinear border align_corners=False、FP16原始点cache。每训练video仅源0帧，20val原8帧分别复用旧balanced和78spatial查询，XY/标签不变，不用GT提示/裁剪/目标相机时间。

raw用FP16cache→FP32；z用自己骨干20480原训练特征FP64总体mean/std→FP32、std≤1e−6 scale1，与83相同规则，不再FP16或搜EPS。普通统计直接复用，register统计在训练前保存SHA。各19994参数768头/GT32 16FG16BG/2000步64000曝光/AdamW .001/.01、完全同索引generator/类SHA。每50步balanced raw val选早同分checkpoint；raw0、两val各自共享偏移原grid/tie规则均保留。

主要描述register raw−普通raw的balanced raw宏差；匹配z−z、两骨干内标准化效应及2×2差分、全部三seed/两查询/profile均保留，完整方案变化不唯一归因4token或某个插值算子。候选顺序普通raw、普通z、regraw、regz，按源balanced校准宏均值选。只有所选register组均值超过两普通组且相对同raw/z普通参考三seed校准差都正，才有资格另预注册91RGB准备/92新确认；不按旧COCO240成绩换参考/配方，未达保留负结果不消费新RGB。

独立重建未来metadata候选/排除240、120新特征元数据/训练源0、首train562和val258两个查询集真实register特征手动切分重算、20480训练统计/三完整采样/六头预算checkpoint、24val输出与四新共享校准曲线/主次交互/选择资格/历史指纹。核验后保存自有代码/协议/报告/文本快照独立提交推送，不上传数据/cache/统计向量/权重/第三方，立即下一必要研究。不把反复源val当新泛化，也不因新确认高分自动替换nativeGT默认。
