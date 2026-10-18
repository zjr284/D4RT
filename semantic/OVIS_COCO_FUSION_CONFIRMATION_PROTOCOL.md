# 第86轮：冻结融合方案的新COCO40独立确认

第85官方RGB/39近对盲复核与40资格独立核验提交推送后执行；沿85固定候选、模型/三seed/偏移/统计。唯一主要strict image-equal宏F1：84验证选中b14_ade_decoder−同宽b14_zero，两者source spatial偏移均+1.5。先三seed平均每图差，再seed20261086固定2000图像bootstrap95%区间；汇总query宏与其差另报，不挪用该CI。禁止按确认改选/调参/删负图种子/换默认。

五组15头：旧zB14及zero/MIT/ADEencoder/ADEdecoder拼接。三既定profile均保留：raw0、源balanced偏移、源spatial偏移；主要只spatial。全部次要未校正：decoder−旧zB14；ADEencoder−MIT；三拼接对zero和旧zB14在raw/balanced/spatial下；zero−旧zB14；选中decoder spatial−raw。consensus只敏感性，不作为第二独立确认。相同对照键去重，不能按最高口径报告。

每图1024 GT无关无放回XY seed20261077+imageID，128坐标源0，GT strict/consensus同77/80。BG/Person/14非Person共同词表保留，未知歧义crowd仅评价忽略，全26预测类别不减。GT注释仅生成评价标签，不参与RGB特征/头/拼接。40图元数据已见，预训练重合未知，不是视频跟踪/OVIS AP/完整自然像素。

原JPEG PILRGB bilinear448/FP32除255。B14沿冻结BF16归一化32×32 patch768，MIT/ADE沿82 FP32，encoder256×14×14与ADEdecoder classifier前256×112×112；固定像素中心/128 bilinear border align_corners=False，所有原point feature先FP16缓存，再使用83 train-only均值/scale在FP32标准化并按84拼接。bone eval/冻结、TF32两类均关、每图T1，无GT裁剪/类别提示/目标相机输入。所有旧权重/源码/依赖与统计直接哈希，未训练/重校准。

独立重建40XY/80GT、首图B14官方patch与独立SegFormer多尺度decoder手动重算、160缓存元数据/指纹、15全logits逐值及90策略profile聚合+每图指标、全部预定跨组配对/图像bootstrap/源选择与偏移/输入指纹。保留负结果及BG误报/FG漏报代价；改善只属于整个特征方案，不能独立归因ADE标签、某层或空间分辨率。不上传图像/GT/缓存/统计向量/权重/第三方。核验后独立提交推送、立即下一必要研究。
