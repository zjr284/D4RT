# 第89轮：官方register B14资源与RGB预检结果

经[官方README](https://github.com/facebookresearch/dinov2)、[模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)及现有固定源码核对，使用唯一官方dl.fbaipublicfiles.com的dinov2_vitb14_reg4_pretrain.pth；没有安装新依赖或下载第三方代码。最多1HEAD+4GET/400MiB/900秒预算下，实际1HEAD+2GET合计346393545字节，首段不完整后严格206 Range补齐，同ETag/LastModified，未全量重下。完整178个ZIP条目CRC通过。实测SHA256：73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71；这是本地来源记录，不是作者签名；ETag非单part MD5，明确不作MD5认证。

现有官方dinov2_vitb14_reg严格weights_only加载成功，86583552参数，比普通B14多3072；4register、antialias=True、插值offset0，普通B14为0register/False/.1。权重来自另一预训练/蒸馏方案，后续收益不可唯一归因4token或某个算子。官方Apache2许可，预训练LVD142M与OVIS/COCO重合未知。

只用已见源val258的原RGB，T1/T8得到patch1/8×1024×768、register1/8×4×768及class1/8×768，固定原每32个查询取一个共64，仅XY入特征；T1用对应8个源0点。真实数据不含分类GT输入。普通B14 T8同64点逐值复现旧缓存。本次所抽源0点两模型T1/T8量化特征最大差均0，不推广成所有视频/批量不变性。

register T8本次约.0969秒，峰值已分配582766592B，普通T8约.1006秒/583085056B；启动/顺序不同且仅一个样本，不称稳定加速/生产吞吐/整体D4RT成本。磁盘/8GiB设备资源足够此冻结特征实验，CPUFP32权重、BF16骨干/FP16点cache/FP32头既定配方保持。

独立完整HTTP段逐字节重组/长度与SHA/Range预算/ZIP CRC、严格构建两模型；手动prepare_tokens→所有blocks→最终norm→class/register/patch切分与点采样，四套T1/T8完整patch/register/点cache逐值复算通过，旧输入指纹均未变。无头训练、分类评分、新确认或部署变化。权重/原图/cache/第三方源码只在本地，不入Git。

资源已可用，提交推送后立即第90同80train/20val同查询标签/19994参数/GT32预算，普通和register×raw和固定train-only标准化的2×2；只训练register六头，普通六头复用。先固定后续COCO候选身份（排除已评分240）再训练，是否取RGB/确认由事先源选择规则决定；不把旧确认再评分当新结论。
