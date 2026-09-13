# D4RT 论文复现与本科科研起点

本仓库是依据 [CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html)及其附录构建的**非官方、缩小版研究原型**。它实现了视频编码、五元组查询、独立 cross-attention 解码、多任务预测和基于预测点的相机配准，并用可控合成场景验证端到端流程。

它**不是论文最终指标的复现**：本地 tiny 模型约 8.4M 参数，从随机初始化训练 300 步；论文主模型约 1B 参数，使用大规模数据和预训练权重。合成评测也不能与论文的真实数据表格直接比较。代码中的未公开设计选择见下文。

## 运行

在仓库根目录执行（需要 Python、PyTorch；本地已有 `.venv`）。本次验证环境为 Python 3.12、PyTorch 2.14.0+cu130、NumPy 2.5.2、RTX 5060 Laptop 8GB；`demo.py` 还需 NumPy，真实视频输入另需 torchvision 与 PyAV，`inspect_model.py` 需 Matplotlib：

```bash
.venv/bin/python tests/test_d4rt.py
.venv/bin/python train.py --config tiny --low-memory --steps 300 --seed 0 --out checkpoints/repro_safe_seed0 --save-initial
.venv/bin/python eval.py --ckpt checkpoints/repro_safe_seed0/tiny_last.pt --samples 3 --seed 42 --grid 24 --tracks 16 --json-out checkpoints/repro_safe_seed0/eval_seed42_baseline.json
.venv/bin/python eval.py --ckpt checkpoints/repro_safe_seed0/tiny_last.pt --samples 3 --seed 42 --grid 24 --tracks 16 --dynamic-aware --json-out checkpoints/repro_safe_seed0/eval_seed42_dynamic.json
.venv/bin/python demo.py --ckpt checkpoints/repro_safe_seed0/tiny_last.pt --out /tmp/d4rt_safe_demo
.venv/bin/python research/dynamic_pose_benchmark.py --seeds 5
```

`--low-memory` 使用 96×96、8 帧、128×128 合成渲染和每 clip 512 查询，并将检查点间隔限制为 25 步。此次 WSL 只有约 7.4 GiB RAM/2 GiB swap；直接跑常规 160×160、12 帧、256×256 渲染和 2048 查询时，内核日志曾记录 OOM 并杀死 Python。脚本会在小于 10 GiB RAM 的环境阻止此高内存组合，除非显式传 `--force-high-memory`。短程低内存试验的最大常驻内存约 2.3 GiB。

`train.py` 自动选择 CUDA；没有 CUDA 时回退 CPU。`--save-initial` 保存训练前权重。训练保存配置、逐步 JSONL 日志与原子替换的完整检查点；断连后用**相同参数**加 `--resume checkpoints/repro_safe_seed0/tiny_last.pt` 继续。`eval.py --seed` 固定场景，三次训练种子的数值与限制见[复现实验报告](docs/reproduction_report.md)。不要覆盖旧实验目录。

## 数据流与代码

```text
视频 V ──> 编码器 E ──> 全局表示 F
  │                         │
  └─ 9×9 查询处 RGB patch     │
          + (u,v,源/目标/相机时刻)
                    └──> 独立 Cross-Attention 解码 ──> 3D、2D、可见性、位移、法线、置信度
```

| 文件 | 功能 |
|---|---|
| `models/d4rt.py` | 时空管状 patch、局部/全局编码、查询嵌入、纯 cross-attention 解码器与输出头 |
| `models/loss.py` | 归一化 3D、置信度、2D、可见性、位移和法线损失 |
| `data/synthetic.py` | 可解析相机与运动物体的合成视频、任意时空查询 GT |
| `utils/augment.py` | 时空采样、裁剪及与 `align_corners=False` 一致的像素坐标映射 |
| `utils/query_factory.py` | 深度、点云、跟踪和相机查询构造 |
| `utils/camera.py` | 刚体/相似变换 Umeyama、焦距估计、可选动态感知鲁棒配准 |
| `tracking/dense_tracker.py` | 占位栅格稠密跟踪 |
| `train.py`, `eval.py`, `demo.py` | 训练、合成评测、推理演示 |
| `tests/test_d4rt.py` | 数值、几何、梯度及端到端冒烟测试 |

论文查询 `q=(u,v,t_src,t_tgt,t_cam)` 的核心是“**同一个源像素对应的物理点**，在目标时刻处于哪里，并用哪个时刻的相机坐标系表达”。深度使用 `t_src=t_tgt=t_cam` 并取预测 3D 点的 Z；相机外参对同一点切换 `t_cam`，再做刚体配准。`encode` 只需对视频运行一次，多个查询共享场景表示。

## 与论文一致的部分及明确偏离

论文描述的 2×16×16 视频 patch、交替帧内/全局注意力、宽高比 token、9×9 查询 RGB patch、三个独立时间嵌入、无查询间 self-attention 的解码器、13 维多任务输出，以及 AdamW/余弦调度等均有相应实现。可选 `vit-b/l/h/g` 用于结构和参数量检查；**未在本机训练或验证这些大模型**。`tiny` 是本项目自定义配置，不是论文配置。

以下差异会直接限制“严格复现”的结论：

- **数据与初始化**：使用在线解析合成场景，不是论文训练数据；随机初始化，没有论文使用的预训练权重和训练预算。合成物体、纹理、遮挡分布也不代表真实视频。
- **未公开细节**：论文附录写有独立的 `L_conf`，但没有给出可核对的定义；默认损失只实现已明确的 `c·L_3D − λ_conf·log(c)`，`use_lconf=True` 是本地实验选项。傅里叶频率、解码器 FFN 宽度和部分增强幅度同样是合理实现假设，不能视为作者原代码。
- **评测**：`eval.py` 只在本地合成场景计算深度 AbsRel、位姿、点云和跟踪的**代理指标**；其中 APD3D/AJ/OA 并非官方 TAPVid-3D 评测器，位姿 AUC 也只是本地定义。不能用这些数值声称论文公开基准成绩。
- **尺度**：3D 损失对预测与 GT 各自按平均深度归一化，不能保证绝对尺度。无尺度对齐的相机平移、3D 点跟踪和焦距估计可能显著失真；需要分别报告尺度不变与绝对指标。
- **裁剪与内参**：已知裁剪框和原图尺寸时，焦距估计先把查询坐标映回原图；对未知真实裁剪视频，原图内参不可仅凭当前图像唯一恢复。

## 已完成的关键一致性检查

测试覆盖：全局注意力确实更新视频 token、宽高比到达视频 token、默认查询采样 RNG 前进、动态球体渲染与 GT 运动时间戳一致、裁剪后 2D GT 与输入视频坐标一致、刚体配准不误用尺度、Sim(3) 反射修正、裁剪后内参坐标恢复、空有效掩码损失、梯度流和密集跟踪。当前 26 项测试全部通过，结果保存在 `checkpoints/repro_safe_seed*/`。旧的 `checkpoints/run1/` 是修复前权重，**不应作为当前代码的基线**。

## 下一步科研问题

仓库提供普通 Umeyama 基线，以及**非论文原方法**的动态感知加权 Umeyama。后者使用预测置信度、可见性、目标时刻的运动幅度与 IRLS 权重。本次三种子实测**没有稳定改善旋转或 ATE**；不能声称研究假设已成立。`research/dynamic_pose_benchmark.py` 使用**真值运动作为 oracle**，只能检查“有可靠运动线索时鲁棒估计器能否工作”，不能证明 D4RT 预测足够准确。下一步需要按动态物体占比和线索质量做分组压力测试与消融。

详细学习路线见用户提出的 10–12 周计划：先理解几何与论文公式，再完成 tiny 基线，最后围绕一个可检验问题做控制实验。原论文 PDF、附录版 PDF 与中文笔记均保留在仓库中。
