# 第79轮：空间校准新COCO候选的冻结与合法RGB准备

第78研究9dcaf3b已核验推送main30dd6a33后立即开始。78新偏移只源OVIS val，77旧COCO不作为调参。本轮仅冻结新集合/模型/新旧偏移、有界官方RGB获取与重复盲复核，不解码候选GT masks、不提模型特征/评分。全部5000 COCO元数据已检查，故称模型未评价新图像，不称标注完全未见；预训练COCO重合未知。

排除第65/67已模型评分160个官方COCO ID。排序5000 ID后seed20261079一次排列；14非Person严格同名类别（已知映射2..11及21..24）以非crowd/area>0元数据建池，按classID顺序两轮取未用第一项，再按同排列补到candidate40；排除既有160+candidate40后按相同两轮/补齐规则固定reserve40。保留原顺序，无按GT有效性/成绩换图；reserve40只元数据，不下载/评分。LVVIS40不消耗。

固定78七方案21头/原旧偏移与新空间val共享偏移，禁止oracle/75投影/76多层新头加入。后续80唯一主：strict下B14_auxSAM新空间偏移−同头旧偏移的图像等权宏F1（先三seed平均），seed20261080/2000图像bootstrap95%区间；汇总query宏另列，不混其CI。重要预定次同头新−raw0，及其余六方案新−旧/新−raw均未校正，consensus仅敏感性，不挑最高口径/模型部署。固定每图1024无放回XY seed20261077+imageID，不依GT、保留BG/Person/共同FG、未知歧义crowd只评价忽略，26预测类别保持。BG误报下降并非全部性能改善保证，FG召回代价和raw基线必须报告。

来源沿64官方COCO实例val2017元数据与原coco_url/path，当前安全TLS路径式https://s3.amazonaws.com/images.cocodataset.org/val2017/...，无禁证书/私有凭据/付费。只candidate40，禁止自动重定向，0新HEAD、最多160GET/每图4段、单图8MiB/总64MiB、socket30秒/总900秒、GET起点间隔至少1秒。首次200 ContentLength/ETag固定，短响应仅续Range，后续必须206/精确ContentRange/同ETag；两次连续零字节连接失败或不可恢复HTTP停止，保留前缀/账本，不无限重试/重置预算。完整ETag单part MD5、PIL JPEG验证/元数据尺寸、file/pixel SHA和dHash核验后才登记。

COCO图像依各Flickr原license条款，保存license_id/许可描述/源链接，不笼统称全部CCBY；注释CCBY4并不等于所有图片许可。数据/图片/GT/缓存/权重/第三方不上传Git。

对照全部42149 OVIS索引帧、当前所有external*/image_records.json可用LVVIS RGB、原COCO160、候选内部：file/pixel精确与dHash<=6近似对生成contact sheets。实际查看近似对，不根据类别/GT/分数淘汰；精确或同场景近帧排除、保留原因、不补图，完全不同图保留。一个Codex视觉复核者，不冒充独立人工；有限hash筛查不保证所有衍生重复/预训练无重合。资格必须先冻结后80才可GT/预测。

独立元数据抽样重建、40完整JPEG/尺寸/MD5/两像素指纹、所有HTTP段账本/预算、参考库与近对/实际复核资格、21头/新旧偏移/输入指纹；完成自有协议/代码/报告/文本快照独立提交推送后当前运行立即80固定评估，不等心跳。不按网络失败/GT无效偷偷补图，失败保存明确负结果/依赖边界后另协议恢复。
