# 第64轮：COCO val实例元数据小范围获取成功

一次公开DoH返回原官方下载域images.cocodataset.org的CNAME为images.cocodataset.org.s3.amazonaws.com，支持按S3 path-style读取同一桶对象。保留第62 GCP404、第63 canonical HTTPS主机名不匹配失败，没有关闭TLS/HTTP降级。DoH AD=false，来源关系来自公开DNS和正确主机名HTTPS，不是作者签名或DNSSEC认证。

复用第63已固定总长/版本头，本轮0新HEAD、1个DoH GET、7个206 Range GET，实际注释范围接收6660841字节（≤64MiB），没有下载252907541字节全ZIP。目标压缩流首读短30177字节，按已固定连续补足规则第二段完成，记录全部区间。只解压annotations/instances_val2017.json，19987840字节、SHA256 e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f，成员deflate/长度/CRC通过。

元数据5000唯一图像、80类别、36781实例注释（36335 polygon、446 RLE）；全部annotation关联与license ID合法。严格同名OVIS共同15类（含Person），非Person14类：Bird/Cat/Dog/Horse/Sheep/Cow/Elephant/Bear/Zebra/Giraffe/Bicycle/Motorcycle/Airplane/Boat。非crowd正area元数据图像数分别125/184/177/128/65/87/89/49/85/101/149/159/97/121。它们是元数据覆盖，尚未验证128像素/查询支持或模型性能，静态图像不能提供跟踪/几何结论。

独立验证器只用本地已保存区间构造虚拟seek文件，由标准zipfile独立解析全部目录字段、解压目标并验证CRC，与原成员逐字节相同；全部区间身份/版本/字节预算、schema/许可ID/同名类图像统计、冻结模型与旧输入指纹通过。没有下载RGB、解码mask或提取模型特征/预测。图片版权各自Flickr条款，不能将注释CC BY4.0泛化到所有图片；每图license来源后续保留。

提交推送后立即第65固定COCO候选80/暂留80及模型/查询选择，再仅获取这些RGB、与全部OVIS及已观察LV-VIS图像和新集合跨图重复筛查/盲复核。先按元数据固定，不按新预测/有效mask挑图；原LV-VIS40暂留不动，当前部署native_gt保持。
