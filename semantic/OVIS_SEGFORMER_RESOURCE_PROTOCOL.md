# 第81轮：官方SegFormer/MIT-B0隔离资源与源RGB预检

第80研究bb91fb7已核验推送maina603db5e后立即接续。80空间校准优于旧偏移却未确认优于raw，且FG漏BG显著增加；不在确认上继续调阈值。本轮先检查独立分割骨干资源，不训练OVIS头/骨干、不计算GT指标、不读取新确认，不改默认native_gt或现有.venv。

只两种预定官方公开权重：[NVIDIA mit-b0 ImageNet1k](https://huggingface.co/nvidia/mit-b0)与[NVIDIA SegFormer-B0 ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)。[原论文](https://arxiv.org/abs/2105.15203)与[原NV许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)已核对：当前只非商业研究/评价、本地权重不分发；ADE额外密集标签监督/预训练数据与目标重合未知必须披露，不称外部标注预算相同或只架构改变。HF卡是官方namespace转换版（卡由HF团队撰写），记录固定revision/文件SHA而非泛称原作者代码完全相同。

先保存本协议、代码、80核验/原源val258的RGB与查询元数据指纹和原环境关键包/文件快照；元数据/许可/配置读取允许但仅匿名官方HTTPS，不读token/凭据或trust_remote_code。HF API blobs=true先固定每repo40hex commit及LFS size/SHA；优先官方model.safetensors，否则同repo官方pytorch_model.bin，仅weights_only=True加载；每模型≤32MiB、总模型字节≤64MiB。config/preprocessor/README固定commit并检查git-blob SHA1，许可证保存SHA；不从未知镜像/第三方重打包找权重。

模型HTTP含重定向至HF官方域族hf.co/huggingface.co，强制HTTPS/域检查，完整TLS，不记录签名URL query。最多256实际GET（包括重定向）/单模型128段、最小起点间隔1秒、socket30秒/模型网络总900秒、元数据每件≤2MiB/总8MiB；短响应仅按已验证ContentRange起点/总长度续传，200仅可首段，后续必须206；ETag/LastModified稳定，最终LFS SHA256/尺寸必须相同。两次零字节失败或不可恢复HTTP停止，保留前缀/预算/日志，不重置计数无限重试。

新资源环境只在semantic_artifacts/segformer_resource_control/env中创建venv，私有.pth指向原.venv读取其Torch/Numpy/Pillow，不写原环境。固定transformers==4.57.6、huggingface_hub==0.36.0（原2.0.0保持），新环境pip从官方PyPI仅binary、no-cache/isolated、timeout30/retries1/总300秒、保存安装report；只一次安装计划，不自动升级原Torch或全局包。安装前后原关键包版本与文件哈希须一致；模型/依赖/第三方源码不入Git。

加载官方config和严格checkpoint，编码器结构必须同hidden_sizes[32,64,160,256]/depth2×4；ImageNet分类头1000、ADE150保留加载但不映射至OVIS类、不用其GT。源RGB固定已见val258（第一帧T1及原8帧T8），原JPEG448/PIL bilinear/CPU FP32÷255/ImageNet mean/std（要求两官方processor一致）/CUDA BF16，低精度查询缓存FP16。448为同未来DINO对照的预定输入，非ADE卡512默认；原RGB仍真实原JPEG。

只预检三个冻结特征分支：MIT encoder最后归一化stage256（14×14）、ADE encoder同256/14×14、ADE decode classifier之前融合256（112×112）；捕获原模块输出，不暗用GT/类别提示/裁剪。R58同258查询每32取一共64，128像素中心bilinear border align_corners=False，source RGB与xy只提特征；检查张量/有限值、冻结参数、T1-vs-T8首帧差与量化差、时间/显存，不能按GT分数改变batch/尺寸/分支。不是语义性能结论或标准吞吐。

资源成功才另预注册82同原80源GT32/20val/2000步预算的三seed线性头对照：同架构MIT vs ADE encoder控制额外分割预训练，ADE decoder vs encoder完整分支控制，另B14 GT32旧768维头作完整方法基线（参数/网格/外部监督不同须披露）。81不得提前训练/消费COCO40元数据预留或LVVIS40。独立核验官方revision/两个权重与环境不变、严格结构/三64点特征预检及source指纹，保存自有代码/协议/报告/文本快照独立提交推送；失败如实留证后下一有界必要恢复，不把资源准备当实验收益。
