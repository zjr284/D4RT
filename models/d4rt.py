"""D4RT 核心模型实现
论文: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time" (arXiv 2512.08924)

架构总览 (论文 §2.2 / Fig.2 / Fig.7):
  1. D4RTEncoder: 基于 ViT 的视频编码器
     - 时空管状 patch 2×16×16 (论文 §4 "spatio-temporal patch size of 2×16×16")
     - 帧内自注意力与全局自注意力交替 (VGGT [48] 风格)
     - 原始宽高比 W/H 编码为单独 token (论文 §2.2)
     - 输入 resize 到固定正方形分辨率, 输出全局场景表示 F = E(V) ∈ R^{N×C}
  2. QueryEmbedder: 把查询 q = (u, v, t_src, t_tgt, t_cam) 编码为独立 token
     - (u, v) 傅里叶特征编码
     - t_src / t_tgt / t_cam 三个独立的离散时间步嵌入
     - 以 (u, v) 为中心的 9×9 局部 RGB patch 嵌入 (论文 §2.2, 附录 Fig.10 消融确认 9 最优)
  3. D4RTDecoder: 轻量 cross-attention 解码器
     - 查询之间完全独立解码, 不做 self-attention (论文 §2.2 明确: 加了反而掉点)
     - 主配置 8 层 ≈ 144M 参数
     - 输出 13 维: 3D 位置(3) + 2D 坐标(2) + 可见性(1) + 位移(3) + 法线(3) + 置信度(1)

ViT 变体 (论文 Tab.9 消融中的 backbone 规模):
  vit-b: 768 维 / 12 层 / 12 头   (~90M)
  vit-l: 1024 维 / 24 层 / 16 头
  vit-h: 1280 维 / 32 层 / 16 头
  vit-g: 1408 维 / 40 层 / 16 头   (~1B, 论文主模型)
  tiny:  256 维 / 6 层 / 4 头      (本地 8GB 显存验证专用, 非论文配置)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 编码器 (论文 §2.2 "Our encoder E is based on the Vision Transformer [9] with
# interleaved local frame-wise, and global self-attention layers [48]")
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """交替的帧内 (local) / 全局 (global) 自注意力块, ViT pre-norm 结构

    宽高比 token 只在全局层参与注意力 (帧内层按帧分组, 无法容纳额外 token),
    帧内层中原样穿过。
    """

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, local: bool = False):
        super().__init__()
        self.local = local
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, grid):
        # x: (B, 1 + Tt*S, d), 首 token 为宽高比 token; grid = (Tt, S)
        if self.local:
            ar, p = x[:, :1], x[:, 1:]
            B, N, d = p.shape
            Tt, S = grid
            h = self.norm1(p).view(B, Tt, S, d).reshape(B * Tt, S, d)
            h = self.attn(h, h, h, need_weights=False)[0]
            p = p + h.reshape(B, N, d)
            x = torch.cat([ar, p], dim=1)
        else:
            normed = self.norm1(x)
            h = self.attn(normed, normed, normed, need_weights=False)[0]
            x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class D4RTEncoder(nn.Module):
    """时空 ViT 编码器: video → 全局场景表示 F (论文 Fig.2)"""

    def __init__(self, dim: int = 1408, depth: int = 40, nhead: int = 16,
                 mlp_ratio: float = 4.0, patch=(2, 16, 16),
                 image_size: int = 256, num_frames: int = 48):
        super().__init__()
        self.dim = dim
        # 时空管状 patch: 把 2 帧×16×16 像素折叠为一个 token (论文 §4)
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=patch, stride=patch)
        Tt = num_frames // patch[0]
        S = (image_size // patch[1]) * (image_size // patch[2])
        self.grid = (Tt, S)
        # 可学习位置编码 (论文 Fig.7 的 PE)
        self.pos_embed = nn.Parameter(torch.randn(1, Tt * S, dim) * 0.02)
        # 原始宽高比 W/H → 单独 token (论文 §2.2)
        self.ar_proj = nn.Linear(1, dim)
        # 交替: 偶数层帧内自注意力, 奇数层全局自注意力 (论文 §2.2 [48])
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, nhead, mlp_ratio, local=(i % 2 == 0)) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, video, aspect_ratio=None):
        # video: (B, T, 3, H, W), 已 resize 为正方形; aspect_ratio: (B,) 原始 W/H
        B, T, C, H, W = video.shape
        x = self.patch_embed(video.permute(0, 2, 1, 3, 4))  # (B, d, Tt, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, Tt*h*w, d)
        x = x + self.pos_embed[:, : x.shape[1]]
        if aspect_ratio is None:
            aspect_ratio = torch.full((B,), W / H, device=video.device, dtype=video.dtype)
        elif isinstance(aspect_ratio, (int, float)):
            # 兼容 python 标量 (推理时常见): 扩展为 (B,) tensor
            aspect_ratio = torch.full((B,), float(aspect_ratio),
                                      device=video.device, dtype=video.dtype)
        ar = self.ar_proj(aspect_ratio.to(video.dtype).unsqueeze(-1)).unsqueeze(1)  # (B,1,d)
        x = torch.cat([ar, x], dim=1)
        for blk in self.blocks:
            x = blk(x, self.grid)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 查询构建 (论文 §2.2 "A query token is first constructed by adding the Fourier
# feature embedding of the 2D coordinates (u, v) to the learned discrete
# timestep embeddings ... augmenting the query with an embedding of the local
# 9×9 pixel RGB patch centered at (u, v)")
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """连续 2D 坐标 (u, v) 的傅里叶特征编码 [47]

    频率 π·2^i (i=0..F-1), 取 sin 与 cos, 并保留原始坐标以保证亚像素精度
    (论文附录 C 讨论子像素解码)。
    """

    def __init__(self, in_dim: int = 2, num_frequencies: int = 16):
        super().__init__()
        self.frequencies = nn.Parameter(
            2 ** torch.arange(num_frequencies).float() * torch.pi, requires_grad=False
        )

    def forward(self, x):
        # x: (N, 2) in [0, 1] → (N, 2 + 4*F)
        x_proj = x.unsqueeze(-1) * self.frequencies.to(x.device)  # (N, 2, F)
        return torch.cat(
            [x, torch.sin(x_proj).flatten(1), torch.cos(x_proj).flatten(1)], dim=-1
        )


class QueryPatchEmbed(nn.Module):
    """以 (u, v) 为中心在源帧 t_src 上采样 9×9 RGB patch 并线性投影"""

    def __init__(self, dim: int, patch_size: int = 9):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * 3, dim)

    def extract(self, video, t_src, uv):
        # video: (1, T, 3, H, W); t_src: (N,) long; uv: (N, 2) in [0, 1]
        N = uv.shape[0]
        H, W = video.shape[-2], video.shape[-1]
        frames = video[0, t_src]  # (N, 3, H, W), 可微的逐查询帧索引
        # 与视频裁剪/缩放一致地使用 align_corners=False:
        # 像素中心 j 对应 u=(j+0.5)/W，相邻像素的 grid 距离为 2/W。
        cx = uv[:, 0] * 2 - 1
        cy = uv[:, 1] * 2 - 1
        p = self.patch_size // 2
        k = torch.arange(-p, p + 1, device=uv.device, dtype=uv.dtype)
        ox = k * (2.0 / W)
        oy = k * (2.0 / H)
        gx = (cx.view(N, 1, 1) + ox.view(1, 1, -1)).expand(N, p * 2 + 1, p * 2 + 1)
        gy = (cy.view(N, 1, 1) + oy.view(1, -1, 1)).expand(N, p * 2 + 1, p * 2 + 1)
        grid = torch.stack([gx, gy], dim=-1)  # (N, 9, 9, 2)
        patches = F.grid_sample(
            frames, grid, mode="bilinear", padding_mode="border", align_corners=False
        )  # (N, 3, 9, 9)
        return patches.flatten(1)  # (N, 243)

    def forward(self, video, t_src, uv):
        return self.proj(self.extract(video, t_src, uv))


class QueryEmbedder(nn.Module):
    """构建独立 Query Token q (论文 Fig.7 的 "Query Addition")"""

    def __init__(self, dim: int = 1280, max_frames: int = 48, patch_size: int = 9):
        super().__init__()
        self.fourier_embed = FourierEmbedding(2, 16)
        self.uv_proj = nn.Linear(2 + 4 * 16, dim)  # 傅里叶特征 (含原始坐标)
        self.patch_embed = QueryPatchEmbed(dim, patch_size)
        # 三个独立的离散时间步嵌入 (论文 §2.2 "learned discrete timestep embeddings")
        self.embed_t_src = nn.Embedding(max_frames, dim)
        self.embed_t_tgt = nn.Embedding(max_frames, dim)
        self.embed_t_cam = nn.Embedding(max_frames, dim)

    def forward(self, video, uv, t_src, t_tgt, t_cam, video_hr=None):
        # video_hr: 可选, 原始分辨率视频 — patch 从高分辨率帧提取 (论文附录 C Config 4)
        src_video = video if video_hr is None else video_hr
        return (
            self.uv_proj(self.fourier_embed(uv))
            + self.patch_embed(src_video, t_src, uv)
            + self.embed_t_src(t_src)
            + self.embed_t_tgt(t_tgt)
            + self.embed_t_cam(t_cam)
        )


# ---------------------------------------------------------------------------
# 解码器 (论文 §2.2 "The decoder D is a small cross-attention transformer.
# Each query is decoded independently through cross-attention into the Global
# Scene Representation F, ensuring that queries do not interact.")
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    """纯 Cross-Attention + FFN 块: 查询之间零交互 (论文的刻意设计)"""

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q, kv):
        # q: (1, N, d) 查询; kv: (1, Ntok, d) 场景表示 F
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


class D4RTDecoder(nn.Module):
    """8 层 cross-attention 解码器 + 6 个输出头 (共 13 维, 论文 Fig.7)

    主配置: d=1280, 16 头, FFN 3.5×, 8 层 ≈ 144M 参数。FFN 比例由论文
    §4 "our ... cross-attention decoder contain 1 B and 144 M parameters"
    反推: d=1280 下 4× 为 158M, 3.5× 恰为 144M (见 README)。
    """

    def __init__(self, dim: int = 1280, nhead: int = 16, num_layers: int = 8,
                 mlp_ratio: float = 3.5, enc_dim: int = 1408):
        super().__init__()
        # 编码器维度 → 解码器维度的桥接 (论文未指定, 主配置 1408→1280)
        self.bridge = nn.Linear(enc_dim, dim)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(dim, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        # 13 维输出: 3D 位置 / 2D 坐标 / 可见性 / 位移 / 法线 / 置信度
        self.head_3d = nn.Linear(dim, 3)
        self.head_2d = nn.Linear(dim, 2)
        self.head_vis = nn.Linear(dim, 1)
        self.head_disp = nn.Linear(dim, 3)
        self.head_normal = nn.Linear(dim, 3)
        self.head_conf = nn.Linear(dim, 1)

    def forward(self, queries, latent_F):
        # queries: (N, d); latent_F: (1, Ntok, enc_dim)
        kv = self.bridge(latent_F)
        q = queries.unsqueeze(0)  # (1, N, d)
        for blk in self.blocks:
            q = blk(q, kv)
        out = q.squeeze(0)  # (N, d)
        return {
            "p_3d": self.head_3d(out),                    # 3D 位置 P (3)
            "p_2d": self.head_2d(out),                    # 2D 图像坐标 (2)
            "vis": torch.sigmoid(self.head_vis(out)),     # 可见性 (1)
            "disp": self.head_disp(out),                  # 点位移/运动向量 (3)
            "normal": F.normalize(self.head_normal(out), dim=-1),  # 表面法线 (3)
            "conf": torch.sigmoid(self.head_conf(out)) + 1e-4,     # 置信度 c (1)
        }


# ---------------------------------------------------------------------------
# 整体模型
# ---------------------------------------------------------------------------

VARIANTS = {
    # 论文 Tab.9 的 backbone 规模; 解码器按论文主配置 1280/8 层/16 头
    "vit-b": dict(enc_dim=768, enc_depth=12, enc_heads=12, dec_dim=1280, dec_layers=8, dec_heads=16),
    "vit-l": dict(enc_dim=1024, enc_depth=24, enc_heads=16, dec_dim=1280, dec_layers=8, dec_heads=16),
    "vit-h": dict(enc_dim=1280, enc_depth=32, enc_heads=16, dec_dim=1280, dec_layers=8, dec_heads=16),
    "vit-g": dict(enc_dim=1408, enc_depth=40, enc_heads=16, dec_dim=1280, dec_layers=8, dec_heads=16),
    # 本地 8GB 显存验证专用 (非论文配置, 见 README)
    "tiny": dict(enc_dim=256, enc_depth=6, enc_heads=4, dec_dim=256, dec_layers=4, dec_heads=4),
}


class D4RTModel(nn.Module):
    """D4RT 整体架构: 视频编码器 E + Query 构建器 + Cross-Attention 解码器 D

    推理范式 (论文 §2.1): F = E(V) 只算一次; 之后任意查询 P = D(q, F) 并行解码。
    """

    def __init__(self, variant: str = "tiny", image_size: int = 256,
                 num_frames: int = 48, max_frames: int = 48, patch_size: int = 9):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.encoder = D4RTEncoder(cfg["enc_dim"], cfg["enc_depth"], cfg["enc_heads"],
                                   image_size=image_size, num_frames=num_frames)
        self.query_embedder = QueryEmbedder(cfg["dec_dim"], max_frames, patch_size)
        self.decoder = D4RTDecoder(cfg["dec_dim"], cfg["dec_heads"], cfg["dec_layers"],
                                   enc_dim=cfg["enc_dim"])

    def encode(self, video, aspect_ratio=None):
        # video: (1, T, 3, H, W) → F: (1, 1 + Tt*S, d)
        return self.encoder(video, aspect_ratio)

    def decode(self, q_tokens, latent_F):
        return self.decoder(q_tokens, latent_F)

    def decode_queries(self, video, queries, latent_F):
        """复用一次编码得到的场景表示，按需独立解码查询。"""
        q_tokens = self.query_embedder(
            video, queries["uv"], queries["t_src"], queries["t_tgt"], queries["t_cam"])
        return self.decoder(q_tokens, latent_F)

    def forward(self, video, queries, video_hr=None, aspect_ratio=None):
        """queries: dict 含 uv (N,2) in [0,1]、t_src/t_tgt/t_cam (N,) long"""
        latent_F = self.encode(video, aspect_ratio)
        if video_hr is None:
            return self.decode_queries(video, queries, latent_F)
        q_tokens = self.query_embedder(
            video, queries["uv"], queries["t_src"], queries["t_tgt"], queries["t_cam"], video_hr)
        return self.decoder(q_tokens, latent_F)

    def param_counts(self):
        """返回编码器 / (解码器+查询) 参数量, 用于对照论文的 1B / 144M"""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters()) + sum(
            p.numel() for p in self.query_embedder.parameters()
        )
        return {"encoder": enc, "decoder+query": dec}
