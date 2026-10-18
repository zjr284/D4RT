# 第63轮：COCO HTTPS对象入口诊断结果

按预注册只进行了2次HEAD、0 GET/正文下载。官方下载域名images.cocodataset.org的HTTPS请求因证书主机名不匹配失败（CERTIFICATE_VERIFY_FAILED），未关闭验证或降级HTTP。同名S3桶path-style HTTPS端点返回200：252907541字节、AmazonS3、ETag f4bbac642086de4f52a3fdda2de5fa2c，Last-Modified 2018-07-10。自动重定向关闭，完整状态/头或TLS异常已保存。

本地DNS仅给代理地址且没有CNAME，不能由此验证同名S3桶就是作者源对象；仅200不能称官方归属已独立证明，也还没有ZIP/注释CRC核验。独立离线检查请求顺序/方法/预算、TLS失败保留、候选状态与头/旧指纹通过。没有图像/GT/模型评分，部署不变，第62GCP404保持。

下一轮先固定一次公开DNS-over-HTTPS CNAME核对，检查原官方下载域是否实际指向同名S3桶（默认TLS验证、不依赖工具代理DNS），符合后才在已固定S3总长/ETag身份下有限Range取目录和val实例成员；不符则停止获取并保存来源不确定，不把通用存储200当绕过来源核验。新根保留63全部观察，最多64MiB，仍不下全241MB或图片。该迁移入口使用正确主机名TLS，不能用verify=False或自签证书替代。
