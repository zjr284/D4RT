# 第62轮：官方COCO注释入口HEAD失败记录

原定只通过Range读取COCO2017 val实例JSON，不下载整包、图片或解码mask。已固定预算/代码/第61验证选择和9模型指纹。根据COCO官方下载页GCP bucket说明构造的公开HTTPS对象路径，在第一次HEAD即返回HTTP404；没有进入Range GET，没有获得注释/图片，也没有模型评分或新结果。

实际1次HEAD尝试、0次Range GET、0注释字节；原有模型与选择/部署不变。异常类型/消息和回溯保留，HTTP错误响应头未持久化，不能假称已验证Content-Length/ETag或目录/CRC。先前准备的正向标准zipfile验证器没有运行；独立离线失败核验检查回溯中的HEAD位置、缺失head/ranges/member/directory边界及全部旧指纹，通过。未在本轮擅自换镜像或重试。

官方[下载页源码](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm)确实同时列出images.cocodataset.org注释ZIP和GCP bucket说明；当前所试GCP对象404说明该对象路径本次不可用，不能外推COCO数据或所有官方入口不可用。[使用条款](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm)区分注释CC BY4.0和图片各自Flickr权益；没有下载图片或对其统一授权作保证。

提交推送后立即单独预注册官方入口可达性诊断：先对下载页原HTTPS域名和可核对的同一官方对象端点作最多3次HEAD，完整持久化HTTP/TLS失败，不绕证书、不降级HTTP、不重下241MB整包、不自动转非官方镜像。资源可用后另冻结范围获取；失败继续保留具体证据与既有可用LV-VIS/OVIS研究资源，不要求用户代为排查。
