# 第三十八轮：官方val.zip范围读取与预处理pilot验证

第37核验提交推送后立即启动。来源为作者公开val.zip fileID 1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn；保持匿名公开GET/TLS验证，普通公开确认字段不输出。只接受HTTP206及与请求/总大小匹配Content-Range，若服务器忽略Range即在读取大正文前失败，不降级为完整档案下载。

本轮网络总预算16MiB（含确认HTML/首尾/目录/一个pilot图像），不是完整视频下载。检查ZIP32/ZIP64单盘目录、名字唯一/相对路径无遍历、可支持的stored/deflate成员；保存目录范围/版本标识/哈希与GT文件引用匹配计数，所有新输入来自第37有效COCO oracle而不修复/替换失败VIS源。

固定pilot为JSON images[0]，当前imageID0/file_name00000/00000.jpg。在同一档案版本下读取单个成员，CRC/解压长度/图像尺寸必须与目录与GT图像元数据一致；原帧只用于正确性/图像可用性，无特征或分类头评价。后续正式确认和保留集合排除LV-VIS目录00000，不能按本次pilot或之后分数更换视频。

检查COCO metadata按父目录可分组、帧名可排序、每组至少8图像、同组尺寸/GT共享类是否足够（只元数据，不挑预测）；不得据此宣称重复筛查或正式40已固定。OVIS与外部目录同样数字ID不代表同视频，后续须检查图像内容/来源并披露不能完全排除近重复或预训练重合。

独立ZIP/路径/CRC/范围拒绝单元检查和保存的真实范围/GT引用/pilot再解码核验后，保存自有代码报告文本快照、独立提交推送。失败/不支持保持，不下载完整档案或关闭安全设置。后续先建立COCO适配及固定确认/保留，模型评分前验证query/feature/RGB对应。四组三种子/偏移及部署保持第36/37，不评模型、不训练/校准。

入口`.venv/bin/python -m semantic.external_archive_probe`，产物`semantic_artifacts/external_archive_probe/`。
