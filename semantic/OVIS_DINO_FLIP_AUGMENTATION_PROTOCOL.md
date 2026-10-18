# 第88轮：源RGB水平翻转、同点监督与同曝光控制

第87研究5ff1df7已核验推送mainee399928后立即执行。源训练拟合/验证转移差仍大，预注册通用左右不变性增强；不按任何COCO弱类别/负图挑增强，不把240已评分图当新确认，不读新外部RGB/评分。

固定DINOv2 B/14官方冻结BF16骨干，原JPEG PILRGB bilinear448/FP32除255；原80视频第一帧各提一套torch.flip(width)后的RGB特征。query保持原GT标签/原XY/t_src=0，实际采样u'=127−u、v'=v，像素中心/128、32×32 normalized patch768、bilinear border align_corners=False、FP16缓存原始点特征。无需GT掩码或类别来构造RGB/特征，虚拟翻转不是新独立视频/标注。

复用第83 B14训练原20480特征统计，在FP32标准化原与翻转cache，不重拟合aug统计、不再量化。只新训flip_mix三个同768维19994参数头seed0/1/2；原zB14三头及源偏移直接复用。GT32每步16FG16BG/2000步64000总曝光/AdamW .001/.01，索引generator仍seed0/1/2逐批匹配旧头。每个被采样查询使用独立CUDA generator(seed20261088+seed)的32个uniform<.5选择翻转，否则原视图；view RNG不扰动索引RNG，保存两者全SHA、逐类翻转曝光数/视图总数，不保证精确一半。总曝光不翻倍，原/翻虚拟视图不增加独立训练视频数。

每50步balanced raw val宏选早同分checkpoint；原balanced/spatial两套val不翻转，raw0与各自共享偏移grid/tie规则不变。主描述flip_mix−旧zB14的balanced raw宏差，次三seed全profile/两val、同现类训练拟合转移可后续另诊断。源balanced校准均值选，仅重复验证探索，不替换nativeGT默认。若对旧zB14三seed校准均提高可另预注册新外部集合；本轮不消费确认，旧COCO绝不能成为确认新增强。

坐标翻转/双翻恢复、连续坐标和border对应的独立合成网格单元检查；80新增包旧metadata/采样XY/GT一致、首训练562真实翻转特征独立重算、view与索引generator全重放、三头预算/checkpoint/12两val输出/两新校准曲线/对照/历史指纹独立核验。DINO骨干不训练，FP32头/TF32两类关。无三seed选择、无按确认调增强/偏移、不上传数据/cache/权重/统计向量/第三方。完成研究白名单独立提交推送后立即下一必要研究。
