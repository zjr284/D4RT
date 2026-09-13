"""D4RT 损失函数 (论文 §2.3 与附录 A 公式)

L = (1/N) Σᵢ [ c·λ3D·L3D − λconf·log c + λ2D·L2D + λvis·Lvis
               + λdisp·Ldisp + λconf·Lconf + λnormal·Lnormal ]ᵢ

论文权重: λ3D=1.0, λ2D=0.1, λvis=0.1, λdisp=0.1, λconf=0.2, λnormal=0.5

各项定义 (论文 §2.3):
  - L3D: 预测与 GT 点集分别按各自均值深度归一化 [51], 再经变换
    sign(x)·log(1+|x|) 压制远点对损失的主导, 然后取 L1
  - L2D: 图像空间 2D 坐标的 L1
  - Lvis: 可见性的二分类交叉熵 (BCE)
  - Ldisp: 点运动向量的 L1
  - Lnormal: 3D 表面法线的余弦相似度
  - c·L3D − λconf·log c: 置信度加权 + 惩罚项 (Kendall & Gal [25] 风格)
  - 所有项仅在 GT 可用处施加 (论文 §2.3 "applied only where ground truth
    supervision is available"), 通过 mask 实现

【未公开细节】论文附录公式出现 λconf·Lconf，却没有定义 Lconf。
默认仅实现可核对的 c·L3D−λconf·log(c)，不擅自把未知项当成论文损失。
可通过 use_lconf=True 启用实验性的置信度校准项；该配置不是论文复现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class D4RTLoss(nn.Module):
    """论文附录 A 的加权复合损失"""

    def __init__(self, lambda_3d: float = 1.0, lambda_2d: float = 0.1,
                 lambda_vis: float = 0.1, lambda_disp: float = 0.1,
                 lambda_conf: float = 0.2, lambda_normal: float = 0.5,
                 use_lconf: bool = False):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_2d = lambda_2d
        self.lambda_vis = lambda_vis
        self.lambda_disp = lambda_disp
        self.lambda_conf = lambda_conf
        self.lambda_normal = lambda_normal
        self.use_lconf = use_lconf

    @staticmethod
    def log_transform(x):
        """sign(x)·log(1+|x|) — 论文 §2.3"""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def _masked_mean(x, mask):
        # x: (N, ...) 逐点损失; mask: (N, 1) bool
        m = mask.expand_as(x)
        return x[m].mean() if m.any() else x.sum() * 0.0

    def forward(self, preds, targets, mask=None):
        # mask: (N, 1) bool, 指示 GT 有效 (论文: 仅在 GT 可用处施加); None = 全部有效
        p_pred, p_gt = preds["p_3d"], targets["p_3d"]
        N = p_pred.shape[0]
        if mask is None:
            mask = torch.ones(N, 1, dtype=torch.bool, device=p_pred.device)
        if not mask.any():
            zero = p_pred.sum() * 0.0
            terms = {key: zero for key in
                     ("total", "loss_3d", "loss_conf_penalty", "loss_conf",
                      "loss_2d", "loss_vis", "loss_disp", "loss_normal")}
            return zero, terms

        # ---- L3D: 均值深度归一化 + log 变换 + 置信度加权 L1 ----
        # 预测与 GT 各自归一化 (论文 §2.3 "normalized by their respective mean depths")
        valid_p, valid_g = p_pred[mask[:, 0]], p_gt[mask[:, 0]]
        mean_z_pred = valid_p[:, 2].mean().clamp(min=1e-5)
        mean_z_gt = valid_g[:, 2].mean().clamp(min=1e-5)
        err3d = torch.abs(
            self.log_transform(p_pred / mean_z_pred) - self.log_transform(p_gt / mean_z_gt)
        ).sum(dim=-1, keepdim=True)  # (N, 1) 逐点 L1 (对 xyz 求和)
        loss_3d = self._masked_mean(err3d * preds["conf"] * self.lambda_3d, mask)

        # ---- −λconf·log c: 置信度惩罚 (论文公式第二项) ----
        loss_conf_penalty = self._masked_mean(
            -torch.log(preds["conf"]) * self.lambda_conf, mask
        )

        # ---- 实验性 Lconf：误差越小，目标置信度越高；默认不启用 ----
        if self.use_lconf:
            target_conf = torch.exp(-err3d.detach() / max(self.lambda_conf, 1e-6))
            loss_conf = self._masked_mean(
                torch.abs(preds["conf"] - target_conf) * self.lambda_conf, mask
            )
        else:
            loss_conf = torch.zeros((), device=p_pred.device)

        # ---- 辅助损失 (论文 §2.3, 仅在 GT 可用处施加) ----
        loss_2d = self._masked_mean(
            torch.abs(preds["p_2d"] - targets["p_2d"]).sum(-1, keepdim=True)
            * self.lambda_2d, mask
        )
        # BCE 在 autocast 下被禁止 (须 fp32), 且 vis 头已含 sigmoid
        with torch.autocast(device_type="cuda", enabled=False):
            bce = F.binary_cross_entropy(preds["vis"].float(), targets["vis"].float(),
                                         reduction="none")
        loss_vis = self._masked_mean(bce * self.lambda_vis, mask)
        loss_disp = self._masked_mean(
            torch.abs(preds["disp"] - targets["disp"]).sum(-1, keepdim=True)
            * self.lambda_disp, mask
        )
        loss_normal = self._masked_mean(
            (1.0 - F.cosine_similarity(preds["normal"], targets["normal"], dim=-1,
                                       eps=1e-6).unsqueeze(-1))
            * self.lambda_normal, mask
        )

        total = loss_3d + loss_conf_penalty + loss_conf + loss_2d + loss_vis + loss_disp + loss_normal
        terms = {
            "total": total, "loss_3d": loss_3d, "loss_conf_penalty": loss_conf_penalty,
            "loss_conf": loss_conf, "loss_2d": loss_2d, "loss_vis": loss_vis,
            "loss_disp": loss_disp, "loss_normal": loss_normal,
        }
        return total, terms
