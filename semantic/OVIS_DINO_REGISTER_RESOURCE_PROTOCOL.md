# 第89轮：官方DINOv2 B/14 registers资源与冻结RGB预检

第88研究9f1d321已独立核验推送main970dc9c8后立即开始。88翻转源验证三seed均负不继续概率搜索、不消费确认。独立选择官方同B/14/768维registers变体，先资源可行性，不训练头/不评分/不读新确认。此前未使用registers权重。官方来源[README](https://github.com/facebookresearch/dinov2)、[MODEL_CARD](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)、[论文](https://arxiv.org/abs/2309.16588)与现有固定commit7764ea0f源码核对。

现有源码dinov2_vitb14_reg(pretrained=False)：4register、interpolate_antialias=True、interpolate_offset=0，与旧B14无register/插值不同；权重来自独立预训练/蒸馏，完整方案变化不唯一归因4token或单算子。官方代码/权重Apache2许可，LVD142M训练数据重合未知，不新增ADE密集语义监督；论文效果不是本项目测量。

唯一官方权重HTTPS https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth 。冻结原源码/模型卡/许可/本脚本/源RGB/旧B14指纹后，最多1HEAD+4GET、接收≤400MiB、socket30秒/累计900秒、磁盘至少1GiB；无重定向/禁证书/私有凭据/付费。不完整只在同ETag/LastModified和精确206/Range下续传，保留所有段和失败账本，不新起全量GET/重置预算；两连续零字节/不可恢复HTTP/预算即停止。完整长度/ZIP全部CRC/weights_only严格结构加载；记录实测SHA，不冒充作者签名校验。若ETag为单part MD5则另核验，否则明确不可作MD5。

只用已见源验证首段258原JPEG RGB：T1首帧和T8全部8帧448，原BF16/normalize patch768配方，两类TF32关。固定原258查询每32取一共64点（只读XY，不输入GT类别），T1只对应源0点；记录class/register/patch输出shape、FP32模型参数、finite、FP16点量化/时间/显存及T1/T8差，不能宣称跨批逐值不变或生产吞吐。旧B14 T8同64点要逐值复现旧缓存。独立重组HTTP段/哈希ZIP/源码/权重严格加载、手动最终norm切分和像素中心采样复算register/patch及旧B14，不用分类分数选实现。

仅保存自有代码/协议/资源报告/聚合文本独立提交推送，权重/第三方/图像/点特征不上传，原默认nativeGT/几何不变。资源通过后下一轮独立预注册同80源GT/20验证、raw与train-only标准化的register/普通B14同维同19994头/GT32预算对照，参数仅验证选择，不把COCO240已见图称新确认。
