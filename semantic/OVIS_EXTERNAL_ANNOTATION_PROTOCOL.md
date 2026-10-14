# 第三十七轮：公开注释下载确认与真实schema/共同类别核验

第36完成核验提交推送后立即开始。沿第36作者公开的LV-VIS验证/验证image oracle链接，以匿名标准HTTPS GET完成公开大文件下载流程。仅接受指定file ID、drive.usercontent.google.com/download、GET与id/export/confirm/uuid字段；不登录、读取用户cookies/凭据、不处理登录/CAPTCHA/权限请求，不关闭TLS或运行下载文件代码。公共下载确认字段只在内存使用，不输出或提交临时参数。

目标是拿到实际JSON并独立核验VIS/COCO结构、分类表及严格名称映射；不是模型性能实验。12头、四方案、源验证偏移及GT32原验证选择均保持第36冻结。重试最多一次公开页面刷新；拒绝私有访问、异常表单或JSON超限。不自动取得视频档案、不评价新图像。

沿单文件192MiB及总体256MiB元数据预算，计入第36已有97327字节。下载确认页面上限4MiB，实际注释响应按声明大小与流式读取同时限制，失败保留状态/已收到字节/安全原因，部分文件不作数据。只保存源canonical URL、响应类型/字节/哈希、去除临时query的终点，不保存表单值到Git。

解析真实分类表后仍用第36规范：小写/下划线换空格/空白合并、严格名称相同，未共享和重叠区域−1。记录实际可共享类、非共享数、视频/图像/注释计数、id唯一性、annotation引用完整性和RLE/polygon样例类型；目前只实现VIS RLE桥，COCO image结构需要单独明确适配。查看元数据不等于未见标注，之后新复核须披露。

下一正式复核仍须先固定图像来源/重复检查/数据版本/候选40和保留40（不足如实处理）、前景条件查询/预算/seed、完整特征recipe，再评分。GT只采样与评价，不裁剪RGB或提示模型；非穷尽与SAM辅助标注的限制保留。保存自有代码/报告/文本审计快照，独立核验后白名单提交推送并继续图像可用性准备，当前native_gt部署不变。

入口`.venv/bin/python -m semantic.external_annotation_acquisition`，产物`semantic_artifacts/external_annotation_acquisition/`。
