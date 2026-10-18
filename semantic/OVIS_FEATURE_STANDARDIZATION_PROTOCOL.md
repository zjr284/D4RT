# 第83轮：固定训练通道标准化的同信息/同预算头控制

第82研究68205c1已核验推送main3fea9209后立即执行。82全部SegFormer新组远低B14、精度差小，因此不消费预留确认。本轮只检验输入参数化/优化/正则化控制，不加新语义信息/外部监督，不改骨干/原缓存/GT/默认部署、不搜超参。

四个固定raw分支：B14_GT32旧768维与82三个SegFormer FP32分支256维；原各三头复用，新z_四分支各seed0/1/2共12头。BF16不按成绩再挑，FP32沿82主要控制。仅原80源GT查询20480点的特征，无目标标签参与，FP64总体mean/std（ddof0）再转FP32；std≤1e−6时scale=1，否则scale=std。固定z=(x−mean)/scale在FP32执行、无再FP16量化/裁剪/删除通道；val和未来test只用这套train统计，禁止用val拟合/选择EPS。

先冻结协议/代码/原输入，然后fit4套统计并在训练前冻结其SHA/样本数/维度/尺度摘要。均值/尺度是本地模型参数，不入Git。此变换可逆但改变初始化/等价参数上的正则化/优化过程，不把收益唯一归于条件数，更不叫新增语义信息。

同每分支原维度/参数（B14 19994，其余6682）、GT32每步16FG16BG/2000步64000曝光/AdamW .001/.01、完全相同seed/逐批索引SHA/类直方图。每50步原balanced raw val宏选checkpoint早同分；原balanced/78spatial两套val查询不变，raw0与各自val共享偏移按原grid与同分规则保留。主描述z_ADEencoder−raw ADEencoder在balanced val的raw宏，次匹配MIT/decoder/B14及全部三seed/两profile。不把已反复验证当新泛化。

固定候选顺序四raw后四z，源balanced-profile均值选，仅探索，不自动部署。若无新SegFormer方案优于B14，不消耗预留确认证明负模型；所有负结果保存。

另外在原源GT训练点和balanced val记录各冻结raw/z头的raw0拟合、共同出现FG类别宏召回（类别交集由GT支持先固定），避免把train/val类别支持不同误作纯转移差。新z头可折回raw特征线性权重W_raw=W_z/scale、b_raw=b_z−W_raw@mean，验证等价logits的FP32数值差/类别变化，实际评分始终用标准化表示，不按折回差调参数。预定容差2e−4+2e−5×logit最大绝对值，只数值代数检查，非成绩选择。

独立train-only统计/零方差边界、标签XY与480原feature package指纹、12头预算/三共享采样/checkpoint、48两val原始输出/8共享偏移网格/主次对照/选择、训练拟合与共同类召回/折回等价及全部旧指纹；不重训完成头。自有代码/报告/精简聚合文本逐轮独立提交推送，数据/缓存/统计向量/权重/第三方不提交，当前运行立即下一必要控制。不读COCO200已见结果调规则，不消费COCO reserve40或LVVIS确认。
