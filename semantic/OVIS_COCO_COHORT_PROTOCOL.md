# 第65轮：COCO候选80/暂留80固定与RGB盲筛查

第64独立核验提交推送后立即执行。只元数据选图、RGB获取/重复筛查/资格盲复核，不解码新GT、不特征或模型评分。原LV-VIS40暂留不动。COCO5000图元数据已观察，静态来源只能检验源RGB查询语义，不能视频时间/跟踪或几何泛化，预训练COCO重合未知。

seed20261065对按image_id升序的5000图一次置乱；严格同名共同非Person14类（不扩展同义/层级），元数据noncrowd且area>0建池。每类ID升序一轮各取首个尚未选图，重复5轮，然后按同序优先任一非Person合格图填至80为candidate；再排除candidate、沿同一置乱重复5轮+填80为reserve。保留每类实际图像数，不按有效mask/预测替换或补图；同图多类可计多次。图像尺寸/file_name/coco_url/license引用须合法，逐图许可证/来源保留；不将注释CC BY4统一套图片，不上传图片/第三方注释。

冻结7组×3模型：第58 S14_gt256/gt32、B14_gt256/gt32，第61 B14_class/class_video、原S14_native_sam；旧偏移依次−1.5/−1.25/−2/−2/−2/−2/−1.75。主比较最新源验证所选B14_class_video−旧源验证所选S14_gt32；预定未校正次比较B14_gt32−S14_gt32、B14_class_video−B14_gt32、B14_gt256−S14_gt256、S14_native_sam−S14_gt256。未按COCO选模型/偏移。未来candidate评价strict原任意实例重叠忽略为主、consensus唯一已知语义仅敏感性，26预测输出全保留、每非Person类64点/图，有放回/空图不补选；种子20261052+image_id*32+class_id。静态原RGB PIL bilinear448/单帧T1/BF16骨干/384或768/FP16特征→FP32头，源128像素中心查询，不能按测试改为T8取高分。主图像等权先三种子平均、seed20261066/2000图像bootstrap，实际支持/重复像素和来源相关性限制保留。

只正常TLS访问已公开DoH证实的同桶S3 path-style val2017/filename，URL必须来自元数据coco_url对应路径。最多320个GET（每图≤2，短读只续剩余Range、初始0字节HTTP失败可一次重试），总JPEG实际接收≤256MiB、每图≤5MiB、socket30秒、整体900秒（ledger持久化不得重置预算）。禁止HTTP降级/自动重定向/无限重试/整包下载；请求头响应/实际分片先记，Content-Length必符，续段206/Content-Range/ETag/LastModified相同，若ETag为32hex须MD5一致。PIL验证完整解码/注释尺寸与文件/像素SHA、dHash记录；有未完成图则先保存阻碍，不能直接评分或替换图。

与全部42149 OVIS冻结原帧索引、此前130组1040 LV-VIS RGB、新160图内部全部跨图做文件/像素SHA与dHash≤6枚举；全部近似对相关RGB图页实际盲看，不使用GT/预测。同源或无法判断则排除涉及新图（candidate/reserve同样处理）且无替换，精确匹配不可判distinct；没有匹配不伪称人工查看。只有一个代理视觉审阅者非独立人工，哈希非绝对无重复/预训练去重证明。冻结资格前不能评分。

独立复放置乱/双80清单/21权重偏移、160 JPEG/版本字节与MD5/CRC替代性限制（单图对象非ZIP不声称CRC）、完整三源匹配/全部实际RGB复核/无替换资格。文本快照不含图像/GT/RLE/特征/权重/第三方代码，独立提交推送后立即第66固定模型candidate评价；两暂留集合未评分。
