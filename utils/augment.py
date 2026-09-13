"""训练数据增强 (论文附录 A: Data augmentation)

- 时序一致色彩抖动: 同一视频的所有帧共享同一组 brightness/saturation/
  contrast/hue 参数 (保证时序一致性)
- 随机色彩丢弃 p=0.2, 高斯模糊 p=0.4
- 随机裁剪: 尺度比 0.3–1.0, 宽高比在对数域均匀采样 (宽/窄等概率),
  裁剪后 resize 回训练分辨率; 另以 p=0.05 随机缩放 (zoom)
- 时序子采样: 从长视频中按随机步长抽帧

坐标约定: uv ∈ [0,1] 归一化坐标 (增强后图像空间)。
裁剪会返回 uv 映射参数, 供数据集把查询映射回原始渲染像素 (GT 提取用)。
"""

import math

import torch
import torch.nn.functional as F


def temporal_subsample(num_frames_full: int, num_frames_out: int,
                       stride_range=(1, 2), rng=None):
    """从 T_full 帧中按随机步长抽 T 帧, 返回帧索引映射 t_map (T,)"""
    if rng is None:
        rng = torch.Generator()
    s = int(torch.randint(stride_range[0], stride_range[1] + 1, (1,), generator=rng))
    max_start = num_frames_full - 1 - s * (num_frames_out - 1)
    start = int(torch.randint(0, max(0, max_start) + 1, (1,), generator=rng))
    return torch.arange(start, start + s * num_frames_out, s, dtype=torch.long)


def sample_crop_params(H: int, W: int, scale_range=(0.3, 1.0),
                       log_aspect_range=(math.log(0.75), math.log(4.0 / 3.0)),
                       zoom_p: float = 0.05, rng=None):
    """采样随机裁剪参数, 返回 (x0, y0, cw, ch) (原始像素坐标)

    论文附录 A: 尺度比 0.3–1.0; 宽高比对数均匀; p=0.05 额外随机缩放。
    """
    if rng is None:
        rng = torch.Generator()

    def uni(lo, hi):
        return float(lo + (hi - lo) * torch.rand(1, generator=rng))

    s = uni(*scale_range)
    if float(torch.rand(1, generator=rng)) < zoom_p:
        s = s / uni(1.0, 1.2)  # zoom: 裁更小再放大
        s = min(s, 1.0)
    # 宽高比在对数域均匀采样, 宽/窄等概率
    aspect = math.exp(uni(*log_aspect_range))
    ch = max(16, min(H, int(s * H)))
    cw = max(16, min(W, int(aspect * ch)))
    x0 = int(torch.randint(0, W - cw + 1, (1,), generator=rng))
    y0 = int(torch.randint(0, H - ch + 1, (1,), generator=rng))
    return x0, y0, cw, ch


def apply_crop(video: torch.Tensor, x0, y0, cw, ch, out_size: int):
    """裁剪并 resize 回 out_size; video: (T, 3, H, W) → (T, 3, out, out)"""
    v = video[:, :, y0 : y0 + ch, x0 : x0 + cw]
    return F.interpolate(v, size=(out_size, out_size), mode="bilinear", align_corners=False)


def uv_to_orig_pixel(uv, x0, y0, cw, ch):
    """增强后归一化坐标 → 原始渲染张量索引坐标。

    align_corners=False 的 resize 将裁剪图中心 u 映到 x0 + u*cw - 0.5，
    其中原图像素中心在整数张量索引上。少掉 0.5 会产生系统性 GT 偏移。
    """
    px = x0 + uv[..., 0] * cw - 0.5
    py = y0 + uv[..., 1] * ch - 0.5
    return px, py


def color_jitter_temporal(video: torch.Tensor, rng=None):
    """时序一致色彩抖动: 全视频共享一组参数 (论文附录 A)

    video: (T, 3, H, W), 像素值范围任意 (0..1 或 0..255 均适用)。
    """
    if rng is None:
        rng = torch.Generator(device=video.device)

    def uni(lo, hi):
        return float(lo + (hi - lo) * torch.rand(1, generator=rng))

    b = uni(0.7, 1.3)
    c = uni(0.7, 1.3)
    s = uni(0.7, 1.3)
    h = uni(-0.1, 0.1)

    x = video
    # 亮度
    x = x + b - 1.0
    # 对比度: 以整图均值为锚点
    mean = x.mean(dim=(-1, -2), keepdim=True)
    x = (x - mean) * c + mean
    # 饱和度: 与灰度混合
    gray = x.mean(dim=1, keepdim=True)
    x = gray * (1 - s) + x * s
    # 色相: 绕灰度轴旋转 (torchvision 标准做法)
    cos_h, sin_h = math.cos(h), math.sin(h)
    const = torch.tensor(
        [[0.213, 0.715, 0.072]] * 3, device=video.device, dtype=video.dtype
    )
    rot = torch.tensor(
        [[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]],
        device=video.device, dtype=video.dtype,
    )
    transform = const + rot @ (torch.eye(3, device=video.device, dtype=video.dtype) - const)
    # (T, 3, H, W) → 按通道线性组合
    x = torch.einsum("tchw,cd->tdhw", x, transform)
    return x.clamp(0.0, 1.0)


def color_drop(video: torch.Tensor):
    """随机色彩丢弃 (p=0.2 由调用方控制): 转灰度"""
    gray = video.mean(dim=1, keepdim=True)
    return gray.expand_as(video)


def gaussian_blur(video: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5):
    """对每帧做高斯模糊; video: (T, 3, H, W)"""
    k = torch.arange(kernel_size, device=video.device, dtype=video.dtype) - kernel_size // 2
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    T, C, H, W = video.shape
    x = video.reshape(T * C, 1, H, W)  # 逐通道独立卷积 (RGB 不混合)
    x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), kernel)
    return x.view(T, C, H, W)


def apply_color_augment(video: torch.Tensor, blur_p: float = 0.4,
                        color_drop_p: float = 0.2, rng=None):
    """组合色彩增强 (论文附录 A 概率): 抖动 + p=0.4 模糊 + p=0.2 色彩丢弃"""
    if rng is None:
        rng = torch.Generator(device=video.device)
    video = color_jitter_temporal(video, rng)
    if float(torch.rand(1, generator=rng)) < blur_p:
        video = gaussian_blur(video)
    if float(torch.rand(1, generator=rng)) < color_drop_p:
        video = color_drop(video)
    return video


def depth_sobel_boundary_mask(depth_maps: torch.Tensor, quantile: float = 0.9):
    """Sobel 梯度检测深度不连续/运动边界 (论文附录 A 的困难采样掩码)

    depth_maps: (T, H, W)。返回 (T, H, W) bool: 梯度幅值前 (1-quantile) 的像素。
    """
    T, H, W = depth_maps.shape
    # 未命中几何体的像素深度为 inf (合成渲染), 先置 0 避免 Sobel 产生 NaN
    depth_maps = torch.where(torch.isfinite(depth_maps), depth_maps, torch.zeros_like(depth_maps))
    sobel_x = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=depth_maps.device)
    sobel_y = sobel_x.T
    d = depth_maps.view(T, 1, H, W)
    gx = F.conv2d(d, sobel_x.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(d, sobel_y.view(1, 1, 3, 3), padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt()[:, 0]  # (T, H, W)
    # 每帧取梯度幅值前 (1-quantile) 的像素作为边界候选。
    # 注意: 平滑区域 (如纯色背景平面) 的梯度幅值会大面积并列,
    # 纯 quantile 阈值 `flat > q` 在并列时会选空, 故用随机 tie-break
    # 的 top-k 保证每帧恰好选出 (1-quantile) 比例的候选。
    flat = mag.flatten(1)
    k = max(1, int((1 - quantile) * H * W))
    jitter = torch.rand_like(flat) * 1e-6
    top = torch.argsort(flat + jitter, dim=1, descending=True)[:, :k]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, top, True)
    return mask.view(T, H, W)
