# 第63轮：COCO官方对象HTTPS可达性诊断

第62研究2371f90已推送（远程d96601c2包含），其GCP HEAD404失败保持。立即只检查官方注释对象的传输入口，不获取注释/图片或模型评分。冻结两个HTTPS候选：（1）官方下载页原images.cocodataset.org/annotations/annotations_trainval2017.zip；（2）同域名作为公开S3桶名的path-style s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip。第二个仅是同名桶对象假设；若第一入口不能返回可比身份，不能仅凭第二200就称已证明作者对象完全相同，须记录来源推断。

最多2次HEAD、socket30秒、总90秒；禁止自动跟随重定向（响应Location留证）、禁止GET/下载正文/HTTP降级/证书关闭/非官方镜像。公开DNS在当前工具主机仅解析为代理地址198.18.1.129且没有CNAME，不能拿它当AWS归属证明。记录每端点完整URL、HTTP状态/Content-Length/ETag/Last-Modified/Location/服务标识，或TLS/HTTP异常证据；首原域名若200可停止，无须再测试替代。

绑定第62失败与第61验证选择，所有旧模型/数据不变。独立离线检查请求顺序/预算与HEAD限定、响应记录及旧指纹，结果只说明当前端点传输状态，不是annotation CRC、合法性或性能核验。提交推送后立即对可核验端点另固定小范围ZIP目录/成员获取；若全部失败保存阻碍并选择不依赖COCO资源的实际研究，不反复HEAD刷请求。
