# 第70轮：B14匹配采样的锚曝光/损失归一化/标签oracle控制

第69 d8436ab已推送（远程ea56276d包含），立即继续；仅原80train/20val、冻结全部B14特征/同一162560 queries和原teacher sampler，不能把反复val当新确认。不使用COCO旧确认调参数/评分，不消耗两暂留集。

复用weak=B14_SAM三头和gt32=B14_GT32三头；新增三方法×seed0/1/2=9头，全部768维/19994参数/2000步AdamW .001/.01。每步原teacher标签池FG128+BG128、generator(seed)与第69逐批索引完全相同，保留相同GT锚访问64120/64194/64351，不重新抽点。source_active仅t_src0 CE求和/本批活动锚数；source_total同样仅t_src0 CE求和/固定256；oracle所有query用原GT纠正目标、固定256分母。zero方法非锚权重0，不能把512000被抽索引当有512000非零GT监督。若活动锚为0直接失败不重采样。AdamW下归一化影响不能简单等同固定学习率倍数，两个source方法均保留。

oracle只复用第21已独立核验corrected_labels，全部80原xy/original_labels/帧路径必须与69一一相同，t0零纠正、全目标0..25；直接冻结原GT注释/旧oracle标签指纹。后帧额外GT是诊断，不是首帧弱监督候选。特征仍只来自第69源RGB/坐标，不让GT掩码/类别构造特征、筛点、裁剪或提示模型。oracle实际FG/BG目标不保证各半，分别报所有抽样/非零loss目标类别计数。

所有新方法共享初始化规则和原3种子/每50步raw val最佳checkpoint同分早/原三种子共享BG偏移规则，不能用oracle成绩选部署。固定候选顺序weak/source_active/source_total/gt32（oracle排除），val均值最高为本轮可行临时候选；并与原全局source-val候选class_video .831876比较，无部署自动变化。固定三个主要描述性对照source_active−weak、source_total−source_active、oracle−weak，另列gt32参考（其总64000源曝光、每批16+16与这里随机锚数不同，不能称精确匹配）。

保存索引SHA匹配69、teacher时间×类/实际目标/非零目标计数、活动锚数与分母直方图/加权总曝光、最佳checkpoint与负种子；不追加λ网格。独立80元数据/GT缓存匹配、9采样和归一化统计/轨迹最佳点、15val输出/混淆指标/校准全曲线/可选集合/对照与全部旧指纹，之后自有代码报告/文本快照独立提交推送，立即下一必要研究。oracle不可入全局候选或确认部署，数据/缓存/权重不上传，native_gt默认保持。
