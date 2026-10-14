# 第三十六轮：外部入口可访问，注释尚未解析

OVIS607段已用于研究，本轮为外部数据准备，不产生新模型性能结果。固定GT256/GT32/lambda0/nativeSAM四方案三头及源验证偏移，12权重SHA与第33冻结值一致，当前native_gt部署不变。

七次公开HTTPS请求共接收97327字节，均返回HTTP200：YouTube-VOS首页、VIS数据说明、使用条款、官方2019 CodaLab入口、LV-VIS作者README及两份验证注释请求。HTTP200并不代表获得注释；两份Google Drive响应实际是约2.4KiB的大文件下载确认HTML，JSON解析失败。没有视频档案、外部模型评分、训练或校准。

官方[YouTube-VOS首页](https://youtube-vos.org/)说明验证与测试GT已公开；[VIS入口](https://youtube-vos.org/dataset/vis/)仍链接下载页面，真实CodaLab网页请求成功，后续需要检查网页的实际数据链接。[使用说明](https://youtube-vos.org/dataset/term/)列明注释CC BY4.0及非商业研究用途。

[LV-VIS作者仓库](https://github.com/haochenheheda/LVVIS)提供验证VIS注释和image oracle注释，数据CC BY-NC-SA4.0/非商业研究，格式可与VIS接口衔接。作者同时说明训练注释非穷尽、工具使用SAM/STCN，后续必须披露；尚未下载真实JSON，不能声称共同类别集合、GT完整性或图像可用性已核验。

已实现名称严格匹配（小写/下划线/空白格式规范化），不擅自合并eagle/owl到Bird等类别；未共享对象区域标−1、重叠−1。三项单元检查验证名称匹配、非共享区域不变背景、歧义目标名与错误尺寸失败。后续采用已注释共享前景的条件分类复核，避免把不穷尽标注的未标对象当背景，不能称全数据集AP或自然像素评测。

当前逻辑文件系统可用约934GB，GPU RTX5060 Laptop约8GB，已有torch/PIL/pycocotools；requests/gdown未安装，预检沿标准库完成。这不是视频档案下载授权或主机物理磁盘容量保证，下一阶段仍控制数据量。TLS校验一直开启，没有账号/凭据或付费计算。

独立核验七份响应快照/容量边界/JSON失败、12模型/偏移、零评分/零视频档案与原指纹。启动时一处条件表达式语法错误在联网和协议冻结前发生，修正后py_compile通过；原错误与单行修正记录保留，未产生模型结果或改动旧输入。

下一轮继续公开下载流程，验证返回实际JSON、VIS/COCO结构、共同类别与样本计数。仍不启动评分；图像可用性、与OVIS重复检查、正式固定确认集/预留集和查询预处理核验均待完成。原响应HTML不上传，Git仅自有代码/协议/报告及文本审计快照。
