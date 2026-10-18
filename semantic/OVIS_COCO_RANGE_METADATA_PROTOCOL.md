# 第64轮：确认COCO存储别名后有限获取val实例注释

第63提交核验推送后立即接续，所有旧失败/证书错误保留，不修改62/63代码或根。本轮先一次公开DNS-over-HTTPS请求https://dns.google/resolve?name=images.cocodataset.org&type=CNAME （≤64KiB/default TLS），保存完整响应。只有Answer中原官方下载域名的CNAME明确指向images.cocodataset.org.s3.amazonaws.com（或同名桶s3-website-us-east-1.amazonaws.com）才视作path-style访问同一桶的来源证据；否则停止，不下载注释。DNS是传输来源证据，不是作者签名/绝对密码学认证，不依赖本机代理DNS、不绕主机名证书。

通过后仅用第63冻结200响应的S3 path-style URL、总长252907541、ETag f4bbac642086de4f52a3fdda2de5fa2c与Last-Modified2018-07-10，免重复HEAD。本轮最多1个DoH GET+11 Range GET，注释范围接收≤64MiB、DoH≤64KiB、总300秒/socket45秒/每逻辑区间≤4连续分片，所有Range身份必须匹配第63头；拒200大体积回退，不全包下载、不HTTP降级或非官方镜像。

复用第62固定ZIP目录/val成员解析设计（新文件新根）：只annotations/instances_val2017.json，压缩≤32MiB/解压≤64MiB、local/central名称flags方法一致、deflate结束/长度/CRC正确。JSON5000唯一图像/80类/所有annotation关联和license合法、严格同名OVIS元数据映射；不解码mask、不下载RGB/模型评分/按分数筛选。官方条款/格式来源与第62相同，图片许可证不能统一当CC BY，预训练COCO重合未知、静态图像不代表视频跟踪/几何。

绑定第63核验/响应、第61验证候选/旧9模型与全部既定输入，过程中不能按数据改变参数。独立用本地已有区间构造seek文件交给标准zipfile解析/解压/CRC，重算DoH别名条件、请求/预算、元数据/schema/类别与旧指纹。不完整或来源不符保存失败，不偷偷换端点；核验后提交推送，立即另预注册候选/暂留图像集合与RGB筛查。原LV-VIS40暂留、部署native_gt保持。
