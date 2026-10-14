# 第二十三轮：原JPEG纯GT语义模型的接口复现与来源验证

第二十二轮按验证选择的native_gt三头及−1.5已接入独立入口。40段已有确认视频、全部2048查询、三个种子和真实双视频批量通过；语义特征、原始分数及类别逐值复现旧缓存，六项几何与同输入原D4RT逐值相同。本轮只验证接口，未训练、校准或消费新的最终40保留视频。

协议见 [OVIS_DINO_NATIVE_INTERFACE_PROTOCOL.md](OVIS_DINO_NATIVE_INTERFACE_PROTOCOL.md)。原JPEG GT被验证集选中；没有因本轮已有确认的SAM分数更高而改选。

## 新输入契约

`NativeDinoSemanticD4RT`接收原video/query，另需`source_frames`：每个视频对应的原RGB文件路径列表。每帧只解码一次，PIL双线性生成128几何与448语义两份RGB，CPU float32除255、连续NCHW。生成的128图像量化为video.dtype后必须逐值等于video，否则拒绝；旧semantic_video字段与此接口混用也拒绝。

来源检查保证两分支使用同一原帧，不能把无对应关系的高分辨率图片塞进语义分支。原batch直接交原D4RT，语义仅源RGB/u/v/t_src，不读GT、掩码、类别提示或目标/相机时间。旧224/128→448入口及实现未改变。

```python
import torch
from semantic.dino_native_interface import load_native_selected_adapter
from semantic.ytvis import load_video

adapter = load_native_selected_adapter(seed=0, device="cuda")
outputs = adapter({
    "video": load_video(clip)[None].cuda().to(torch.bfloat16),
    "source_frames": [clip["frames"]],
    "query": query,
})
xyz = outputs["xyz_3d"]
categories = outputs["semantic_labels"]
```

入口校验第二十二轮报告/冻结选择/三头指纹及验证选中一致性，再复用严格骨干加载器。全部参数冻结，训练模式拒绝，语义头保持float32。

## 完整正确性结果

120个B=1案例（40段×三种子），另固定清单首两段345/574、每视频2048查询，形成3个真实B=2案例。各案比较原查询及`t_tgt+1/t_cam+3`模8之后的六几何，共1476组数组逐值相同。123案语义分数及类别时间不变，全部逐值相同；参照始终是相同批量、相同BF16输入的原D4RT，不要求或声称跨批几何相同。

全部81920个原查询的在线特征与缓存逐值一致，245760次B=1种子×查询类别比较零变化；B=2的12288次也零变化，语义原始分数与对应B=1逐值相同。完整指标复现第二十二轮native_gt：

| 种子 | 宏F1 | 前景宏F1 | 与旧输出差 |
|---|---:|---:|---:|
| 0 | 0.680751 | 0.668986 | 0 |
| 1 | 0.686103 | 0.674816 | 0 |
| 2 | 0.663786 | 0.651555 | 0 |
| 平均 | 0.676880 | 0.665119 | 0 |

这是已观察视频上的接口复现，不能当另一批独立泛化确认。单元检查涵盖同源生成、B=2、单多查询、目标时间不变、源时间影响、来源错序/数量不匹配/旧字段混用拒绝、重载、冻结、输入不变；旧224/448单元回归通过。CPU真实骨干未评价；CPU单元用替身模型。

独立脚本从全部保存分数重算argmax、总体/前景/逐视频混淆指标，核验完整计数和两骨干/官方源码/输入/输出指纹。所有旧指标、模型与最终保留40集合保持，未出现失败或OOM。

## 资源与限制

RTX5060 Laptop、8GB，每案一次顺序计时，有启动和文件缓存影响，不是标准吞吐基准。完整接口**包含原JPEG读取、两次缩放和来源核对**；原几何仅接收已在GPU的视频，二者计时范围不同。

| 批量/每视频查询 | 原几何中位/秒 | 完整接口中位/秒 | 完整接口范围/秒 | 峰值已分配 | 峰值预留 |
|---|---:|---:|---:|---:|---:|
| B=1 / 2048 | 0.0780 | 0.2630 | 0.1544–0.6429 | 2973.3MiB | 3526MiB |
| B=2 / 2048 | 0.3543 | 0.7120 | 0.6925–0.9707 | 3596.2MiB | 4294MiB |

不把顺序记录当可靠速度排名；B=2只覆盖固定两段，不能外推任意长视频或大批量。重复文件读取仍有明显CPU/磁盘成本，后续可对冻结语义特征复用做独立正确性与性能控制。几何保持不等于几何真值精度。

产物`semantic_artifacts/dino_native_interface_probe/`，文本快照`semantic/results/dino_native_interface_probe_summary.json`及`dino_native_interface_probe_verification.json`；不提交数据、模型、特征或在线预测数组。已模型评价仍540段，剩67中40锁为最终复核保留。

本轮提交推送后立即预注册最终保留40的一次独立确认：所有四组旧RGB/原JPEG×GT/SAM权重和偏移冻结，以native_gt为既定验证选择。最终结果不得改变选择或用于阈值调节，也不得按分数换视频。之后继续无需新确认视频的接口效率研究，保留有限数据用于有必要的新方法。
