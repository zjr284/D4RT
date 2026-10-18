# 第85轮：冻结融合候选、原预留40官方RGB与盲重复筛查

第84研究3f41c61已核验推送main2d2c60c7后立即执行。84唯一满足预定确认资格且按源balanced校准均值选中b14_ade_decoder；本轮先冻结所有五组三头、源balanced/spatial偏移、83统计/骨干/实现，只取得RGB和冻结资格，不解码mask/提特征/评分。原nativeGT默认保持。

候选严格沿第79 seed20261079已经固定的metadata reserve40，原顺序不变，排除已评分200（65/67的160及80的40）；没有按84或外部成绩重抽/补图。此前5000元数据/类别映射已观察，称新模型未评价图像，不称标注未见/预训练不重叠。reserve角色在85正式启用，无其它新增reserve；LVVIS旧保留集合不动。

官方来源/许可和预算沿79：每图保留Flickr许可和源链接，HTTPS路径式S3、无重定向/禁证书/令牌，0HEAD最多160GET每图4段、8MiB/图64MiB总900秒、socket30秒/GET间隔1秒。精确200/206/ContentRange/ETag约束，完整MD5/PIL/尺寸/file+pixelSHA+dHash，短响应续传；连续2零字节或不可恢复HTTP/预算触顶即保存失败停止，不换图/重置预算。

重复筛查全部42149OVIS、79固定的可用LVVIS RGB、旧COCO200和新内部，file/pixel及dHash≤6；实际查看新40RGB及近对contact sheets。精确或同场景近帧排除、不同图保留，记录理由、不补图；一个Codex视觉复核者，不称独立人工。有限hash和单次视觉不保证排除所有衍生重复。资格冻结/独立核验/独立Git提交推送后，另轮86才解码GT并评分。

提前固定86问题：唯一主要strict下选中ADEdecoder拼接−同1024维零通道，以source spatial偏移，每图宏F1先三seed平均再2000次图像等权bootstrap(seed20261086)，报告查询汇总宏但不混CI。五组15头全部输出，raw0、原balanced偏移、spatial偏移全保留；次decoder−旧zB14、ADEencoder−MIT、三拼接对零与旧参考、选中组spatial−raw，均未校正；consensus只敏感性、不冒充第二独立确认。

与77/80相同GT无关1024无放回XY(seed20261077+ID)，保留BG/Person/共同14FG，未知/歧义/crowd只评价忽略，26预测不按GT减类。独立重算所有GT/XY、至少首图两骨干源特征、全部15原输出/所有固定profile指标/配对及区间/输入指纹。不用确认重新选模型/阈值/类别、删负图/种子或默认部署。改善只属于完整融合方案，ADE外部密集监督/架构/空间尺度等不能无对照独立归因。
