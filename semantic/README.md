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
