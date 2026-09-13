"""模型输入/输出查看脚本: 沿数据流打印每一步 tensor 的 shape 与数值范围, 并可视化

运行: .venv/bin/python inspect_model.py
输出: 终端打印 (输入 → F → 查询 token → 13 维输出), 图片存到 inspect_out.png

数据流 (论文 §2.1):
  video (1,T,3,S,S) → encode → F (场景表示, 算一次)
  queries {uv, t_src, t_tgt, t_cam} → query_embedder → q_tokens
  q_tokens × F → decoder → 13 维输出 {p_3d, p_2d, vis, disp, normal, conf}
"""
import matplotlib
matplotlib.use("Agg")  # 无图形界面也能存图
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F

from data.synthetic import SyntheticDataset
from models import D4RTModel
from utils.query_factory import D4RTQueryFactory

torch.manual_seed(0)


def show(name, t):
    """打印一个 tensor 的形状和数值范围"""
    print(f"  {name:12s} shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"范围=[{t.float().min():.3f}, {t.float().max():.3f}]")


# ---------------------------------------------------------------------------
# [1] 输入: 一个与训练同分布的合成样本
# ---------------------------------------------------------------------------
ds = SyntheticDataset(num_frames_out=12, num_frames_full=48,
                      render_res=256, out_res=160, seed=7)
sample = ds.make_sample()

print("[1] 输入 (来自 SyntheticDataset.make_sample)")
show("video", sample["video"])          # (T,3,S,S) 像素 [0,1], 模型真正吃的东西
print(f"  aspect={sample['aspect']:.3f} (原始宽高比 W/H, 编码器的额外 token)")
show("depth(附GT)", sample["depth"])    # 模型看不到, 只用来算损失/评测
show("normal(附GT)", sample["normal_world"])
show("边界掩码", sample["boundary_mask"])

# ---------------------------------------------------------------------------
# [2] 模型 + 编码器输出 F
# ---------------------------------------------------------------------------
CKPT = "checkpoints/run1/tiny_last.pt"
if os.path.exists(CKPT):
    # 有训练好的权重就加载 (与 demo.py 相同方式)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = D4RTModel(variant=cfg["config"], image_size=cfg["res"],
                      num_frames=cfg["frames"], max_frames=cfg["frames"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[模型] 已加载训练权重 {CKPT} (step={ckpt.get('step')})")
else:
    model = D4RTModel(variant="tiny", image_size=160, num_frames=12,
                      max_frames=12).eval()
    print("[模型] 未找到检查点, 用随机初始化 (输出会是噪声)")

with torch.no_grad():
    F_ = model.encode(sample["video"].unsqueeze(0), sample["aspect"])

print("\n[2] 编码器输出 (场景表示 F, 之后所有查询都查它)")
show("F", F_)  # (1, 1 + Tt*S, d): 601 个 token (1 宽高比 + 6 组*100 帧内), 256 维

# ---------------------------------------------------------------------------
# [3] 查询 token (论文 Fig.7 的 Query Addition 结果)
# ---------------------------------------------------------------------------
# 深度查询 (Tab.1 Depth Map 行): 每帧 16×16 网格, tsrc=ttgt=tcam=t
q_list = D4RTQueryFactory.build_depth_queries(T=12, H=16, W=16)
queries = {
    "uv": torch.cat([d["uv"] for d in q_list]),
    "t_src": torch.cat([d["t_src"] for d in q_list]),
    "t_tgt": torch.cat([d["t_tgt"] for d in q_list]),
    "t_cam": torch.cat([d["t_cam"] for d in q_list]),
}
print(f"\n[3] 查询 (共 {len(queries['uv'])} 个 = 12 帧 × 16×16 网格)")
show("uv", queries["uv"])
show("t_src=t_tgt=t_cam", queries["t_src"])

with torch.no_grad():
    q_tokens = model.query_embedder(
        sample["video"].unsqueeze(0), queries["uv"], queries["t_src"],
        queries["t_tgt"], queries["t_cam"])
show("q_tokens", q_tokens)  # (N, d)

# ---------------------------------------------------------------------------
# [4] 13 维输出 (解码器 6 个头)
# ---------------------------------------------------------------------------
# 教学范式: F 只算一次, 之后任意查询都走 decode(q_tokens, F)
def decode(model, video, F_, q):
    with torch.no_grad():
        tok = model.query_embedder(video, q["uv"], q["t_src"], q["t_tgt"], q["t_cam"])
        return model.decoder(tok, F_)


print("\n[4] 解码器输出 (论文 Fig.7 的 13 维)")
depth_all = []
for qi in q_list:  # 逐帧解码 (小模型上无所谓; 大批量时应一次全喂)
    preds = decode(model, sample["video"].unsqueeze(0), F_, qi)
    depth_all.append(preds["p_3d"][:, 2].reshape(16, 16))  # 深度 = p_3d 的 Z 分量
depth_pred = torch.stack(depth_all)  # (T,16,16)

preds = decode(model, sample["video"].unsqueeze(0), F_, queries)
for k, v in preds.items():
    show(k, v)

print("\n  前 3 个查询的原始数值:")
for k, v in preds.items():
    print(f"    {k:7s} = {v[:3].flatten().tolist()}")

# ---------------------------------------------------------------------------
# [5] 可视化: 输入帧 / GT 深度 / 预测深度
# ---------------------------------------------------------------------------
gt = sample["depth"]                                   # (T,S,S)
up = F.interpolate(depth_pred.unsqueeze(1).float(),    # (T,16,16)→(T,160,160)
                   size=(160, 160), mode="bilinear", align_corners=False).squeeze(1)

print(f"\n[5] GT 深度范围 [{gt.min():.2f},{gt.max():.2f}] vs "
      f"预测深度范围 [{depth_pred.min():.2f},{depth_pred.max():.2f}]")
print("   数值范围不同是正常的: 论文 L3D 按均值深度归一化, 绝对尺度不受监督")

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
axes[0, 0].imshow(sample["video"][0].permute(1, 2, 0))
axes[0, 0].set_title("Input frame 0")
axes[0, 1].imshow(gt[0], cmap="viridis")
axes[0, 1].set_title("GT depth frame 0")
axes[1, 0].imshow(up[0], cmap="viridis")
axes[1, 0].set_title("Pred depth frame 0 (upsampled)")
axes[1, 1].imshow(up[6], cmap="viridis")
axes[1, 1].set_title("Pred depth frame 6 (upsampled)")
for a in axes.flat:
    a.axis("off")
fig.tight_layout()
fig.savefig("inspect_out.png", dpi=120)
print("   图片已保存: inspect_out.png")
