# 连续研究当前接续点

用户要求每轮复盘、独立核验、独立Git提交并推送GitHub main后，当前运行立即下一轮。现有分钟心跳仅为空闲/中断恢复。先检查最新聊天消息、semantic最新报告/协议、Git状态及实际进程/结果，避免重复训练、覆盖历史或启动两个研究运行。

第63研究2ca07ec已推送远程8898ec5e，canonical COCO HTTPS hostname证书失败保持、S3候选200记录。第64已成功DoH证实原域CNAME images.cocodataset.org.s3.amazonaws.com，默认TLS的S3 path-style同桶对象限量Range获取val实例JSON；0新HEAD+1DoH GET+7Range，共6660841B（无全241MB），原成员19987840B/SHAe8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f。标准zipfile独立区间目录/解压/CRC/schema/旧指纹通过；5000图/80类/36781anns，15同名共同含Person，非Person14类含Bear/Horse/Sheep等。未RGB/mask/model，报告OVIS_COCO_RANGE_METADATA_RESULTS.md。提交推送后立即第65元数据固定COCO候选80+暂留80：seed20261065，逐非Person类5轮先选未选图、再同一置乱优先其余非Person填80，候选完后同序排除候选再5轮/填80作暂留；保留license和实际类覆盖，不解码mask/按分数换图。冻结第61 val选B14_class_video主对照旧S14_gt32，保留第58四组+class/class_video+S14_nativeSAM共7×3权重/偏移；后续原strict主GT/consensus敏感性，源RGB单帧T1/448、查询64每非Person类，不称视频几何。仅选中160图片正常HTTPS S3桶获取/字节校验/与42149 OVIS+原130组1040LVVIS+新图间全部重复筛查/实际RGB盲复核，独立核验后提交推送才第66评分candidate80，COCOreserve80与LVVISreserve40未评分。静态语义迁移/预训练COCO重合未知/逐图Flickr许可证区别保留。第61 val .831876仅微小混合种子探索，第60新视频区间跨零，部署native_gt不变。

第37有效COCO image oracle：19096图像/79604 RLE注释/1196类，23严格共同词表但第43核验仅14实际注释类，固定8帧Person87.41%，Bear等五类仅1目录；不称完整25类泛化。缺Poultry/Vehical；未知/重叠/crowd GT忽略，始终26预测输出不依GT限类。普通VIS JSON解析失败保留。无官方video_id/track_id/fps，路径目录作组。第41固定48帧24空保留。第42全部42149 OVIS原帧索引/文件SHA与固定48像素核验通过，完全相同组0，但外部近似重复仍须盲复核。pilot00000永久排除，元数据/RGB已观察、辅助SAM标注与预训练重合未知需披露。

OVIS607段均用于研究，无剩余独立视频。第33验证选中GT32的新27段主对照视频区间跨零，一个种子下降；第35GT32实例准确率三种子都低于GT256，宏小幅上升集中少数类别。SAM在第33/35局部正收益与第24负收益都保留，不能一概有效/无用，不按确认最高quarter改选。

外部准备固定GT256/GT32/lambda0/nativeSAM四组三头及源校准偏移−1.5/−1.25/−1.5/−1.75。GT32选择来自此前OVIS验证；目前部署仍第22native_gt（GT256）。不训练/重校准、不把GT类别/掩码输入RGB特征或裁剪、不过滤预测候选至GT共同类别。后续新模型评价前必须另固定真实图像/重复筛查/候选确认及预留集合/查询和完整recipe；元数据已检查、标注工具含SAM、骨干预训练重合未知的限制保留。

标准文件读取工具在当前Linux会话可用；Git写入、网络需按当前权限经批准执行。临时发布脚本/tmp/d4rt_publish_research.py必须先检查，保护远程历史正常合并、不强推，复用Windows Git凭据助手不读取输出令牌。研究白名单自有代码/报告/文本快照；不上传数据、缓存、模型、第三方源码，保持用户README/论文/PPT整理不变。

当前连接未提供automation_update调用，已检索可用工具，不能宣称已更新调度或手写修改应用配置。以本文件和最新semantic结果为接续依据；工具恢复后再通过正式接口更新现有d4rt，勿新建重复调度。用户暂停/停止立即停止实验。无变化时安静，仅有意义结果/完成/失败需用户处理时通知。不付费，不公开发文，不联系他人。
