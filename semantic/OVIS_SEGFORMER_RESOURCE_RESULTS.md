# 第81轮：官方MIT/SegFormer-B0资源与隔离RGB预检完成

本轮只资源/已见OVIS源val258 RGB预检，无OVIS头训练/GT指标/新确认。取官方NVIDIA HF转换版[MIT-B0](https://huggingface.co/nvidia/mit-b0)与[ADE20K SegFormer-B0](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)，[原论文](https://arxiv.org/abs/2105.15203)与[非商业研究/评价许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)记录。ADE有额外密集标注监督，不能把未来差异只归架构/层或声称外部标注预算相同。两卡由HF团队转换/撰写，实际严格config/权重验证为准。

|资源|固定revision|字节|LFS SHA256|
|---|---|---:|---|
|mit|`80983a413c30d36a39c20203974ae7807835e2b4`|14380029|`4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc`|
|ade|`489d5cd81a0b59fab9b7ea758d3548ebe99677da`|15036944|`6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`|

两个文件均完整，MIT为官方pytorch_model.bin以weights_only=True加载，ADE为safetensors；实际3次权重GET（合计29,416,973字节），全模型HTTP含元数据/重定向20GET，预算内、全TLS/官方域/匿名、最终SHA与LFS一致。未保存签名URL query、未读token/凭据、无trust_remote_code/第三方镜像。官方config/processor/README git-blob SHA与许可文件哈希核验。

新私有venv路径在semantic_artifacts/segformer_resource_control/env，通过私有.pth复用父Torch/Numpy/Pillow；transformers4.57.6、hub0.36.0、tokenizers0.22.2、safetensors0.8.0等只在新环境。原.venv hub2.0.0/Torch2.14.0+cu130及关键包版本/文件哈希不变；未升级原环境/全局Torch。PyPI binary wheel哈希/安装报告记录，不提交第三方代码/权重。

|冻结分支|T8特征形状|T1 vs T8首帧最大差|FP16查询量化最大差|
|---|---|---:|---:|
|mit_encoder|[8, 256, 14, 14]|0.192311570|0.003901482|
|ade_encoder|[8, 256, 14, 14]|0.132192850|0.003894806|
|ade_decoder|[8, 256, 112, 112]|0.046875000|0.001953125|

三个分支均输出64×256有限点特征：MIT encoder和ADE encoder均14×14、ADE classifier之前融合112×112；不将ADE150类输出映射OVIS或输入GT提示。实际原JPEG448（预定同DINO输入，非ADE卡512默认）/ImageNet归一化/CUDA BF16/FP16点缓存。MIT全模型3,576,392参数（含1000类头）、ADE3,752,694（含150类头）；两encoder状态结构完全相同。

BF16 T1/T8差异最大.192312/.132193/.046875，不声称跨批特征逐值相同，不按GT分数改精度/尺寸。有限T8前向计时MIT .438秒/ADE .052秒受加载/内核热身/顺序影响，不据此称ADE快八倍；峰已分配约304/1152MiB只是预检，不是D4RT整接口吞吐。

独立全部revision/LFS尺寸SHA/配置git-blob/HTTP段/隔离依赖与父环境、两encoder结构、手动重构ADE多尺度linear_c→上采样→融合BN激活→分类前特征，三个64点FP16逐值复算及T1/T8差/量化差通过。verifier首次编译因else后缺空格失败，已在核验前修正并保留启动日志，未修改冻结主资源脚本/模型或重下载。summary SHA256：`bc1fe20848783d6aafeb053c149e88fa671af6d763daec0158b52bef6b12d22c`。

下一轮82将预检数值差作为预定控制因素：MIT encoder/ADE encoder/ADE decoder三个分支×FP32/BF16各三seed，仍同OVIS80源GT32/20val/2000步64000曝光，256维6682参数头（B14旧768/19994仅完整方法基线）；关闭TF32、FP32与BF16模型权重相同，不按确认成绩选精度。主同FP32 ADEencoder−MITencoder，次decoder−encoder与对应精度差；原balanced/空间val两个校准profile分别保存，不混查询任务/调新确认。18头是真实训练控制，不把资源预检当性能收益；另协议先冻结再训练。
