# 第75轮：固定GT优先投影的同预算补充监督控制

第74研究fc00ad8已核验推送main4bb05698后立即接续。第74不支持SAM独有/普遍梯度压倒GT，GTextra也有冲突，因此本轮GT补充/SAM补充都做相同实际干预，不能只改SAM。只是原80训练/20反复验证探索，不读取旧COCO/LVVIS确认或消耗LVVIS40，不宣称泛化/改默认native_gt。

灵感来源[Yu等Gradient Surgery论文](https://arxiv.org/abs/2001.06782)及[作者实现说明](https://github.com/tianheyu927/PCGrad)。本轮是单向GT锚优先投影，非原论文随机对称PCGrad，无照抄第三方代码：每步baseGT CE梯度g_b保持，补充g_s改为g_s'=g_s−min(dot(g_s,g_b),0)/(||g_b||²+1e−12)*g_b；送AdamW的合并原始梯度为g_b+g_s'。固定eps/规则，无λ/LR/梯度裁剪搜索。仅原始欧氏补充方向非冲突（浮点尺度2e−5），AdamW动量/二阶预条件/权重衰减不保证实际更新方向与g_b同向，单独记录实际参数下降向量dot(g_b)。

仅新增project_extra_gt/project_sam各seed0/1/2共6头；复用第71 gt32/extra_gt/aux_sam九头。原冻结B14 768维/19994参数、源20480和后帧142080固定特征/标签/查询，base16FG16BG+supp16FG16BG，2000步/总128000曝光/GT基础64000、AdamW .001/.01；源GT补充池vs后帧teacher池仍不同时间/坐标分布。base generator(seed)、supp generator20261071+seed全部2000索引SHA与71相应原组相同，模型初始化同seed。目标不改、无后帧oracle入训练。

每50步raw val宏选择checkpoint同分早、共享背景grid−8..8步.25/同分abs小负优先，旧偏移复用。主比较project_sam−aux_sam；预定次project_extra_gt−extra_gt、project_sam−project_extra_gt、project_sam−gt32。全部seed/负结果保存，不按旧确认调策略；原可行候选顺序gt32/extra_gt/aux_sam/project_extra_gt/project_sam按源val均值选，仅探索、不自动部署。

先投影向量单元（冲突/同向/正交/零base、base不变、非finite拒绝）通过再冻结代码/协议/测试/74验证、71所有输入/旧权重。保存完整2000步pre/post dot/norm/alpha与真实AdamW下降dot、全部采样类/时间直方图/索引SHA、初始化及前三步权重快照供独立短前缀复算；快照/权重/缓存不入Git。单元验证公式及浮点正交，不用模型分数设容差。

独立重放六全采样轨迹/目标预算；用单独autograd+NumPy FP64投影实现复算六个前三步AdamW更新，对快照FP32差阈值2e−6，不重复训练全程。核验每步投影代数/非冲突界/实际更新统计、checkpoint选择/15份val logits逐值/全部校准/四对照/输入指纹；完成代码报告文本快照独立提交推送，当前运行立即下一必要研究。
