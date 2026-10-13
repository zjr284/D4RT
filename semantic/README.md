# D4RT 查询点语义扩展

本实验使用**非官方 OpenD4RT 32 帧公开权重**，冻结或轻量微调其查询解码器；并非重新预训练 Google DeepMind 的 D4RT。原有六项几何输出保持不变，新增 `semantic_logits [B,Q,C]`。当前主数据集为 **OVIS**：25 个物体类别加 1 个“非目标类”，所以 `C=26`；原 YouTube-VIS 2019 路径仍可通过 `--dataset ytvis2019` 使用，此时 `C=41`。类别标签由源像素 `(u,v,t_src)` 决定，改变 `t_tgt` 或 `t_cam` 不应改变物体类别。SAM 2 只传播实例掩码，类别 ID 来自数据集真值。

## 数据与依赖

从 [OVIS 作者的数据页](https://songbai.site/datasets/)进入其公开 Google Drive，取得 `annotations_train.json` 和 `train.zip`，分别放到 `semantic_artifacts/ovis/`；解压后帧路径为 `semantic_artifacts/ovis/train/<video_id>/img_*.jpg`。本实验从 OVIS 训练集重新划分 60/20/20 视频，不使用其官方验证/测试集。当前本地文件已校验：标注有 607 段视频、25 类；ZIP 全量校验通过，标注列出的 42,149 帧全部存在。数据、模型权重和实验产物均位于被 Git 忽略的 `semantic_artifacts/`。

依赖安装（使用仓库现有 PyTorch 环境）：

```bash
.venv/bin/python -m pip install -r semantic/requirements.txt
.venv/bin/python -m pip install --no-deps 'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
```

SAM 2.1 tiny 官方权重应位于 `semantic_artifacts/weights/sam2.1_hiera_tiny.pt`；本地校验 SHA-256 为 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。当前 SAM 2 缺少可选 CUDA 后处理扩展，会显示警告；传播可以运行，但正式结果需记录这一环境限制。

## 可重复执行顺序

以下命令均在仓库根目录运行。`manifest` 先用种子 42 随机选 100 个合格视频，再在这 100 个视频内部做确定性交换，保证验证/测试源帧类别都在训练源帧出现；视频不跨集合。

```bash
.venv/bin/python -m semantic.run manifest --dataset ovis --annotation semantic_artifacts/ovis/annotations_train.json --frames-root semantic_artifacts/ovis
.venv/bin/python -m semantic.teacher --dataset ovis --annotation semantic_artifacts/ovis/annotations_train.json --frames-root semantic_artifacts/ovis --manifest semantic_artifacts/manifest.json --out semantic_artifacts/teacher_ovis
.venv/bin/python -m semantic.run cache --dataset ovis --annotation semantic_artifacts/ovis/annotations_train.json --frames-root semantic_artifacts/ovis --manifest semantic_artifacts/manifest.json --teacher-dir semantic_artifacts/teacher_ovis --cache-root semantic_artifacts/features_ovis
.venv/bin/python -m semantic.run train --cache semantic_artifacts/features_ovis/sam --variant gt --out semantic_artifacts/runs_ovis_balanced
.venv/bin/python -m semantic.run train --cache semantic_artifacts/features_ovis/sam --variant sam --out semantic_artifacts/runs_ovis_balanced
.venv/bin/python -m semantic.run train --cache semantic_artifacts/features_ovis/sam --variant rgb --out semantic_artifacts/runs_ovis_balanced
.venv/bin/python -m semantic.run train --cache semantic_artifacts/features_ovis/sam --variant lora --out semantic_artifacts/runs_ovis_balanced
.venv/bin/python -m semantic.run report --dataset ovis --annotation semantic_artifacts/ovis/annotations_train.json --frames-root semantic_artifacts/ovis --manifest semantic_artifacts/manifest.json --teacher-dir semantic_artifacts/teacher_ovis --runs semantic_artifacts/runs_ovis_balanced --out semantic_artifacts/summary_ovis.json
```

先做少量视频的链路检查时，`teacher` 和 `cache` 可加 `--max-videos 2`，`train` 加 `--pilot --steps 10` 并使用独立的缓存/输出目录。`--pilot` 仅跑种子 0，结果明确标记为 pilot，不可当正式对照结论。完整实验须有 60/20/20 缓存，每组跑种子 0/1/2、每种子默认 500 步。

训练只开放每段训练视频首帧的实例真值及类别。SAM 2 从这一帧传播至后 7 帧；低置信、重叠和边界不确定像素标为忽略。非首帧真值仅用于教师 IoU 诊断，不参与伪标签筛选。验证/测试只用原始真值，也不调用 SAM 2。GT 对照只训练首帧查询；SAM 组训练首帧与传播标签；RGB 组使用源帧局部像素；LoRA 组以 SAM 头为起点，仅更新末两层解码器的 rank-8 前馈 LoRA 与语义头。各组共享视频划分、查询缓存和训练步数；每步前景/背景各采样一半，避免 SAM 只提供可靠正例使训练类别先验偏移。原先未平衡的诊断运行保留在 `semantic_artifacts/runs_ovis/`，不计入正式对照。

评价报告准确率、测试集中出现类别的宏 F1、逐类 support/F1、改变目标/相机时间后的标签一致率、LoRA 几何输出平均绝对变化、峰值显存及耗时。类别一致率在模型只预测背景时也可能很高，必须连同逐类 F1 判断。此处不是 OVIS 官方实例分割 AP，也不是 D4RT 原论文的几何基准。

训练完成后，加载 OVIS 语义适配器时需指定 26 类：

```python
import torch
from semantic.model import SemanticD4RT
from tools.load_opend4rt import DEFAULT_VARIANT, load_model

base = load_model(DEFAULT_VARIANT / "opend4rt.ckpt", DEFAULT_VARIANT / "model.yaml")
model = SemanticD4RT(base, variant="sam", num_classes=26)
model.load_adapter("semantic_artifacts/runs_ovis_balanced/sam_seed0.pt")
model = model.to("cuda", dtype=torch.bfloat16).eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model({"video": video, "query": query})
```

## 验证与边界

```bash
.venv/bin/python tests/test_semantic.py
.venv/bin/python tests/test_semantic.py --integration --pipeline
.venv/bin/python tests/test_d4rt.py
```

单元测试覆盖 OVIS/YouTube-VIS 标签、视频隔离、源帧类别覆盖、查询参数不变性与 checkpoint 重载。真实 OVIS 小样本 pilot 仅验证数据解析、SAM 传播、冻结特征、分类头/LoRA 前反向和评价链路；**不能从 pilot 推断 SAM 蒸馏有效**。完整三种子实验的结论须以 `summary_ovis.json`、逐类结果与失败案例为准。

## 后续研究

第二轮背景伪标签控制实验的固定协议、运行入口与结果目录见 [OVIS_BACKGROUND_PROTOCOL.md](OVIS_BACKGROUND_PROTOCOL.md)。

完整三种子结果见 [OVIS_BACKGROUND_RESULTS.md](OVIS_BACKGROUND_RESULTS.md)：背景识别恢复，但前景宏 F1 下降，总体宏 F1 未稳定改善。

第三轮完成源时间与前景类别采样的 2×2、三种子实验，见 [固定协议](OVIS_SAMPLING_PROTOCOL.md)和[结果与机制诊断](OVIS_SAMPLING_RESULTS.md)。两项采样修正均未恢复第一轮前景性能，下一步需区分监督覆盖与冻结表征限制。

第四轮真值诊断已完成，见 [协议](OVIS_ORACLE_PROTOCOL.md)和[结果](OVIS_ORACLE_RESULTS.md)。原查询标签修正及八帧完整真值覆盖均未恢复测试表现；相同类别的训练/测试召回率差距提示优先检查跨视频泛化。

第五轮完成分类头×训练预算诊断，见 [协议](OVIS_HEAD_BUDGET_PROTOCOL.md)和[结果](OVIS_HEAD_BUDGET_RESULTS.md)。线性头延长至2000步在三个种子都改善，跨视频泛化差距仍大；下一轮应在统一预算下回到弱监督方案。

第六轮统一预算弱监督复核完成，见 [协议](OVIS_WEAKLABEL_BUDGET_PROTOCOL.md)和[结果](OVIS_WEAKLABEL_BUDGET_RESULTS.md)。线性2000步同时改善两方案，但背景恢复、前景受损在所有配对种子中仍存在；下一步优先验证背景分数校准。

第七轮背景分数校准完成，见 [协议](OVIS_CALIBRATION_PROTOCOL.md)和[结果](OVIS_CALIBRATION_RESULTS.md)。补背景组验证选−2.0偏移，三个种子总体/前景宏F1均改善；下一轮补纯首帧GT校准基线。

第八轮公平首帧GT基线完成，见 [协议](OVIS_SOURCE_GT_PROTOCOL.md)和[结果](OVIS_SOURCE_GT_RESULTS.md)。相同2000步与验证校准下，SAM前景/背景方案相对纯GT在三个种子均改善。下一轮应固定模型，在原100段之外的未使用视频复核。

第九轮未使用视频确认完成，见 [协议](OVIS_FRESH_CONFIRMATION_PROTOCOL.md)和[结果](OVIS_FRESH_CONFIRMATION_RESULTS.md)。40段新视频上，SAM前景/背景优于只补前景但低于纯GT，此前超越GT的收益没有复现。后续优先分析冻结错误输出并重新设计独立确认。

第十轮冻结错误/覆盖审计完成，见 [协议](OVIS_ERROR_AUDIT_PROTOCOL.md)和[结果](OVIS_ERROR_AUDIT_RESULTS.md)。校准后主要错误是前景类间混淆；SAM没有增加任何类别的独立训练视频，六个仅覆盖单视频的类别在确认集F1均为零。覆盖与性能并非单调关系，后续需新划分训练验证。

第十一轮等视频数覆盖干预完成，见 [协议](OVIS_VIDEO_COVERAGE_PROTOCOL.md)和[结果](OVIS_VIDEO_COVERAGE_RESULTS.md)。全25类至少3段源帧查询覆盖未改善确认宏F1；SAM相对各自GT基线在三种子都改善，但验证到确认的泛化差距仍大。全部查询、预算、模型输出与验证选择独立核验通过；下一步审计冻结类别错误并评估表征控制。

第十二轮冻结跨视频审计完成，见 [协议](OVIS_TRANSFER_AUDIT_PROTOCOL.md)和[结果](OVIS_TRANSFER_AUDIT_RESULTS.md)。14个共同类的训练源帧宏召回约78%–87%，确认源帧约9.5%–14.9%；Bear在覆盖组训练召回超96%但确认仍为零。源时间变化和训练类别缺失不足以解释全部失败，下一步预注册独立语义特征控制；本轮是描述性诊断，未训练或重新校准。

第十三轮独立语义特征对照完成，见 [协议](OVIS_SEMANTIC_FEATURE_PROTOCOL.md)和[结果](OVIS_SEMANTIC_FEATURE_RESULTS.md)。相同源帧GT与预算下，新40段模型确认的宏F1为D4RT 0.0733、DINOv2 0.5565，三种子均改善、39/40视频改善；前景类间错误显著减少，训练缺失的Bear仍零分。下一步在固定DINOv2特征下重检视频覆盖，再验证语义分支接入查询接口。

第十四轮特征×视频覆盖对照完成，见 [协议](OVIS_FEATURE_COVERAGE_PROTOCOL.md)和[结果](OVIS_FEATURE_COVERAGE_RESULTS.md)。覆盖DINOv2宏F1从随机组0.5056提高到0.5613，Bear从零提高到0.7718；视频等权仅19/40改善，覆盖收益有限定。下一轮固定DINOv2比较有/无SAM传播，复用GT头并冻结新的确认视频。

第十五轮DINOv2下SAM对照完成，见 [协议](OVIS_DINO_SAM_PROTOCOL.md)和[结果](OVIS_DINO_SAM_RESULTS.md)。随机组GT/SAM宏F1为0.5798/0.6238，覆盖组0.6155/0.6322，两组三个种子均改善；视频bootstrap区间均跨零，保留负视频。SAM在此特征下有局部辅助收益，下一步验证语义分支接入D4RT的几何保持和时间一致性。

第十六轮真实推理接口验证完成，见 [协议](OVIS_DINO_INTERFACE_PROTOCOL.md)和[结果](OVIS_DINO_INTERFACE_RESULTS.md)。三个种子在固定3视频上的六项几何保持、目标/相机时间语义一致性均逐值通过，576次旧缓存类别比较无变化。RGB预先BF16量化会改变少量类别；下一步扩大完整查询与真实批量验证，并固定几何/语义的双RGB精度接口。本轮未训练或产生新泛化确认。

第十七轮完整双精度接口验证完成，见 [协议](OVIS_DINO_FULL_INTERFACE_PROTOCOL.md)和[结果](OVIS_DINO_FULL_INTERFACE_RESULTS.md)。几何保留BF16、语义保留对应float32图像，在40段已有视频的全部2048查询及真实B=2上，1476组几何数组逐值保持，245760次B=1类别比较无变化，既有宏F1完整复现。下一轮固定骨干比较224/448语义特征网格密度；接口验证不算新泛化确认。

第十八轮224/448分辨率对照完成，见 [协议](OVIS_DINO_RESOLUTION_PROTOCOL.md)和[结果](OVIS_DINO_RESOLUTION_RESULTS.md)。固定骨干、SAM标签、查询及线性预算，新40段确认宏F1从0.6763提高到0.7399，三个种子均提高，37/40视频改善，视频等权差95%区间[0.0301,0.0663]。按验证选定448、偏移−1.5；保留3段负视频及类别/错误类型差异。下一轮验证448分支接入完整D4RT接口，旧224入口继续兼容。

第十九轮448完整接口验证完成，见 [协议](OVIS_DINO_448_INTERFACE_PROTOCOL.md)和[结果](OVIS_DINO_448_INTERFACE_RESULTS.md)。40段全部查询、三种子、真实B=2与旧224兼容共126案例通过，1512组几何数组逐值保持，在线语义分数/类别完全复现；全接口峰值已分配约2960/3568MiB。新增显式448验证选择加载入口，默认224不变。下一轮补448纯GT头，结合冻结模型开展分辨率×SAM监督对照。

第二十轮分辨率×监督2×2完成，见 [协议](OVIS_DINO_RESOLUTION_SAM_PROTOCOL.md)和[结果](OVIS_DINO_RESOLUTION_SAM_RESULTS.md)。新40段448 GT/SAM宏F1为0.6685/0.7071，三种子均提高，28/40视频改善、视频区间高于零；224也有相近SAM收益，分辨率×SAM交互区间跨零。验证仍选448 SAM；下一轮立即进行固定查询标签纠错oracle诊断，不把额外GT方案当可部署弱监督。

第二十一轮标签纠错oracle完成，见 [协议](OVIS_DINO_LABEL_ORACLE_PROTOCOL.md)和[结果](OVIS_DINO_LABEL_ORACLE_RESULTS.md)。固定SAM查询与采样算法，162560个训练标签仅771改变；额外GT纠错后已观察40段宏F1从0.7071到0.6991，两种子下降。保留负结果，不将oracle当弱监督部署候选或新确认。下一轮立即检验保留原JPEG信息的RGB预处理控制。
