# 第62轮：独立COCO图像复核的官方注释资源准备

第61独立核验并提交推送后立即执行。本轮只获取并核验COCO2017 val实例注释元数据，暂不下载图片、不解码GT掩码、不产生特征/预测，不是新泛化结果。为后续检验冻结语义分支在更广非Person类上的迁移作准备；静态图像不能证明时间/跟踪或D4RT几何性能。

2026-10-18已核对官方资料：[下载](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm)、[条款](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm)、[格式](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/format-data.htm)。下载页列出annotations_trainval2017.zip及官方GCP bucket images.cocodataset.org；使用该bucket公开HTTPS等价入口https://storage.googleapis.com/images.cocodataset.org/annotations/annotations_trainval2017.zip，不装SDK、不需账户、不付费、不绕TLS。注释与网站CC BY4.0，图片不由COCO统一持有版权、须遵循各自Flickr条款；本轮不下载图片，未来保留每图license来源，不对图片统一宣称CC BY。第三方数据不入Git。

单次预算≤64MiB、1次HEAD+最多11次Range GET、总300秒/socket45秒，每个逻辑区间最多4次连续补足，不重复整份下载。HEAD记录总长/ETag/Last-Modified；GET必须206、精确Content-Range、身份一致、无压缩，不接受200大体积回退。所有响应头先保存，实际接收/分片/短读和失败记录持续保存。只读ZIP目录及annotations/instances_val2017.json成员（压缩≤32MiB、解压≤64MiB）；原RangeZIP目录解析器复用、不改历史文件，成员单独严格核对local/central名称/方法/flags、deflate长度/终止与CRC。失败保存具体证据，不另加无界重试或换非官方镜像。

JSON必须5000唯一图像、80唯一类别，所有annotation关联合法，存储原license/图像尺寸/官方coco_url、类别与注释数，严格同名归一化映射OVIS词表、统计各类元数据图像数量，不按新预测或有效mask筛图。本轮仍不选择具体确认集合/评价口径，不拿同名类数当已评价类数。后续另冻结候选/预留划分与查询规则、RGB全部去重盲复核，再模型评分；COCO与骨干预训练重合未知，静态来源/许可证/多目标重叠限制明示。

绑定第61验证选择/偏移/9权重与第58旧S14/B14对照，不新增训练、不按COCO选模型。原LV-VIS40暂留继续不动。独立验证器用只含已下载区间的虚拟seek文件交由Python zipfile独立解析目录/解压/CRC，核验原成员字节/JSON/schema/全部映射统计、区间版本/预算/哈希和旧输入，只有通过才报告资源完成。按轮提交推送后立即后续固定集合准备。
