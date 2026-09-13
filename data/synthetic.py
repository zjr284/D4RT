"""合成动态场景数据集 (验证/冒烟训练用)

真实数据集 (PointOdyssey 数百 GB、TartanAir 等) 在 8GB 显存与国内网络下
不现实, 因此用解析式渲染器生成可控动态场景: 静态纹理背景平面 + 若干运动
刚体 (立方体 = 6 个平面, 或球面), 相机沿平滑路径运动, 全部 GT (3D 位置、
2D 坐标、可见性、位移、法线) 由已知几何解析计算, 无噪声。

渲染方式: 逐像素光线与平面/球面求交 + z-buffer 最近表面选择, 全程张量
向量化, 无需光栅化器。

GT 语义约定 (README 记录):
  p_3d / p_2d / disp / normal 均为"目标时刻 ttgt 的状态、表达在目标相机
  tcam 坐标系"; vis = 点在 ttgt 帧是否可见 (z-buffer 遮挡测试)。
"""

import math

import torch
import torch.nn.functional as F

from utils.augment import (
    temporal_subsample,
    sample_crop_params,
    apply_crop,
    uv_to_orig_pixel,
    apply_color_augment,
    depth_sobel_boundary_mask,
)


# ---------------------------------------------------------------------------
# 解析式场景渲染
# ---------------------------------------------------------------------------

def look_at(C, target, up=(0.0, 1.0, 0.0)):
    """从相机中心 C 看向 target, 返回 c2w 旋转矩阵 (列 = 相机轴在世界系下)

    相机空间 +z 为视线方向 (见 build_rays), 故相机 z 轴 = target - C。
    """
    z = target - C
    z = z / z.norm()
    x = torch.linalg.cross(torch.tensor(up, dtype=z.dtype), z)
    x = x / (x.norm() + 1e-8)
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z], dim=1)  # (3, 3)


def build_rays(H, W, f_norm, C, R_c2w):
    """逐像素世界系光线方向 d (H, W, 3); 主点 (0.5, 0.5), 归一化焦距 f_norm"""
    u = (torch.arange(W) + 0.5) / W
    v = (torch.arange(H) + 0.5) / H
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    dir_cam = torch.stack([(gu - 0.5) / f_norm, (gv - 0.5) / f_norm,
                           torch.ones_like(gu)], dim=-1)  # (H, W, 3)
    dir_cam = dir_cam / dir_cam.norm(dim=-1, keepdim=True)
    return torch.einsum("hwc,dc->hwd", dir_cam, R_c2w)  # 世界系方向 = R @ dir_cam


class PlaneSurface:
    """平面: 世界系点 O、法线 n、平面基 (e1, e2), 解析纹理"""

    def __init__(self, O, n, e1, e2, color, texture):
        self.O = O
        self.n = n / n.norm()
        self.e1 = e1 / e1.norm()
        self.e2 = e2 / e2.norm()
        self.color = color  # (3,) 基色
        self.texture = texture  # dict: kind/freq/amp/phase

    @staticmethod
    def procedural(u, v, color, texture):
        # u, v: 平面坐标 (H, W); 返回 RGB (H, W, 3)
        k = texture.get("kind", "waves")
        amp = texture.get("amp", 0.3)
        f1 = texture.get("f1", 0.7)
        f2 = texture.get("f2", 0.5)
        p1 = texture.get("p1", 0.0)
        p2 = texture.get("p2", 0.0)
        if k == "checker":
            c = torch.sin(f1 * u + p1) * torch.sin(f2 * v + p2)
            pattern = torch.stack([c] * 3, dim=-1)
        else:  # waves: 三个通道不同频率的条纹, 便于建立对应关系
            r = torch.sin(f1 * u + p1)
            g = torch.sin(f2 * v + p2)
            b = torch.sin(0.5 * (u + v) + p1 + p2)
            pattern = torch.stack([r, g, b], dim=-1)
        return (color.view(1, 1, 3) + amp * pattern).clamp(0.05, 0.95)

    def intersect(self, C, rays):
        """光线-平面求交; 返回 (t_hit, valid, local_u, local_v)

        与平面夹角过小的掠射光线 (|denom| ≤ 0.05) 交点极远, 视为无效
        (避免无限平面 + 掠射角产生 1e7 量级的伪深度污染 GT)。
        """
        denom = (rays * self.n).sum(-1)  # (H, W)
        numer = ((self.O - C) * self.n).sum(-1)
        with torch.no_grad():
            valid = denom < -0.05
        t_hit = torch.where(valid, numer / denom.clamp(max=-0.05), torch.inf)
        valid = valid & (t_hit > 1e-3)
        X = C + t_hit.unsqueeze(-1) * rays  # (H, W, 3)
        rel = X - self.O
        return t_hit, valid, (rel * self.e1).sum(-1), (rel * self.e2).sum(-1)

    def render(self, C, rays):
        t_hit, valid, u, v = self.intersect(C, rays)
        rgb = self.procedural(u, v, self.color, self.texture)
        return {"depth": t_hit, "valid": valid, "rgb": rgb,
                "normal": self.n.expand_as(rgb)}


class SphereSurface:
    """球面: 中心 S, 半径 r"""

    def __init__(self, S, r, color, texture):
        self.S = S
        self.r = r
        self.color = color
        self.texture = texture

    def render(self, C, rays, O, R):
        # 球心和纹理都随刚体位姿移动；否则图像与解析轨迹 GT 不一致。
        b = (rays * (C - O)).sum(-1)
        c = ((C - O) ** 2).sum(-1) - self.r ** 2
        disc = b ** 2 - c
        valid = (disc > 0)
        t_hit = torch.where(valid, -b - torch.sqrt(disc.clamp(min=0)), torch.inf)
        valid = valid & (t_hit > 1e-3)
        X = C + t_hit.unsqueeze(-1) * rays
        normal = X - O
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)
        # 用球面局部坐标做纹理: 经纬
        local = torch.einsum("hwc,cd->hwd", X - O, R)
        u = torch.atan2(local[..., 0], local[..., 2])
        # 浮点误差可能使 |X-S|/r 略超 1, asin 会出 NaN → 夹紧到 [-1, 1]
        v = torch.asin((local[..., 1] / self.r).clamp(-1, 1))
        rgb = PlaneSurface.procedural(u, v, self.color, self.texture)
        return {"depth": t_hit, "valid": valid, "rgb": rgb, "normal": normal}


class BoxSurface:
    """运动立方体: 6 个矩形面 (解析裁剪), 刚体位姿 O(t), R(t)"""

    def __init__(self, half_extents, color, texture):
        self.he = torch.tensor(half_extents)  # (3,)
        self.color = color
        self.texture = texture
        # 6 个面: 局部法线 + 局部中心 + 局部平面基 (u, v 轴)
        n_local = torch.tensor([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                                [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
        c_local = n_local * self.he  # 面中心
        self.n_local = n_local
        self.c_local = c_local
        # 面内基: 对每个面取两条与法线正交的局部轴
        u_axes, v_axes = [], []
        for n in n_local:
            ua = torch.zeros(3)
            idx = torch.argmax(n.abs()).item()
            ua[(idx + 1) % 3] = 1.0
            ua = torch.linalg.cross(n, ua)
            ua = ua / ua.norm()
            va = torch.linalg.cross(n, ua)
            u_axes.append(ua)
            v_axes.append(va)
        self.u_axes = torch.stack(u_axes)  # (6, 3)
        self.v_axes = torch.stack(v_axes)

    def faces_world(self, O, R):
        """当前位姿下 6 个面的世界系几何: 中心 (6,3), 法线 (6,3), 基 (6,3)x2"""
        centers = O + (R @ self.c_local.T).T
        normals = (R @ self.n_local.T).T
        us = (R @ self.u_axes.T).T
        vs = (R @ self.v_axes.T).T
        return centers, normals, us, vs

    def render(self, C, rays, O, R):
        centers, normals, us, vs = self.faces_world(O, R)
        H, W = rays.shape[:2]
        best_t = torch.full((H, W), torch.inf)
        best_rgb = torch.zeros(H, W, 3)
        best_normal = torch.zeros(H, W, 3)
        any_hit = torch.zeros(H, W, dtype=torch.bool)
        for k in range(6):
            n, ctr, ua, va = normals[k], centers[k], us[k], vs[k]
            denom = (rays * n).sum(-1)
            numer = ((ctr - C) * n).sum(-1)
            hit = denom.abs() > 1e-6
            safe_denom = torch.where(hit, denom, torch.ones_like(denom))
            t_hit = torch.where(hit, numer / safe_denom, torch.inf)
            X = C + t_hit.unsqueeze(-1) * rays
            rel = X - ctr
            lu = (rel * ua).sum(-1)  # 面内局部坐标
            lv = (rel * va).sum(-1)
            # 矩形裁剪: 局部坐标须在面半宽内 (另两轴)
            lims = self.he[torch.tensor([(k // 2 + 1) % 3, (k // 2 + 2) % 3])]
            hit = hit & (t_hit > 1e-3) & (lu.abs() <= lims[0]) & (lv.abs() <= lims[1])
            rgb = PlaneSurface.procedural(lu, lv, self.color, self.texture)
            closer = hit & (t_hit < best_t)
            best_t = torch.where(closer, t_hit, best_t)
            best_rgb = torch.where(closer.unsqueeze(-1), rgb, best_rgb)
            best_normal = torch.where(closer.unsqueeze(-1), n.expand_as(rgb), best_normal)
            any_hit = any_hit | closer
        return {"depth": best_t, "valid": any_hit, "rgb": best_rgb,
                "normal": best_normal}


class SyntheticScene:
    """一个合成动态场景: 相机路径 + 背景平面 + 运动刚体"""

    def __init__(self, H=256, W=256, f_norm=0.8, rng=None):
        if rng is None:
            rng = torch.Generator()
        self.H, self.W = H, W
        self.f_norm = f_norm
        g = lambda lo, hi: lo + (hi - lo) * torch.rand(1, generator=rng)

        # ---- 相机路径: 平滑 Lissajous 轨迹, 始终看向原点附近 ----
        self.C0 = torch.tensor([0.0, 0.0, 0.5])
        self.amp = torch.stack([g(0.3, 1.2), g(0.15, 0.6), g(0.2, 1.0)]).flatten()
        self.w = torch.stack([g(0.02, 0.10), g(0.02, 0.10), g(0.02, 0.10)]).flatten()
        self.phi = torch.stack([g(0, 6.28), g(0, 6.28), g(0, 6.28)]).flatten()
        self.look_target = torch.tensor([g(-0.5, 0.5), g(-0.5, 0.5), -2.0]).flatten()

        # ---- 背景平面 (静态, 纹理) ----
        self.bg = PlaneSurface(
            O=torch.tensor([0.0, 0.0, -12.0]),
            n=torch.tensor([0.0, 0.0, 1.0]),
            e1=torch.tensor([1.0, 0.0, 0.0]),
            e2=torch.tensor([0.0, 1.0, 0.0]),
            color=torch.stack([g(0.2, 0.6), g(0.2, 0.6), g(0.2, 0.6)]).flatten(),
            texture=dict(kind="checker", amp=float(g(0.15, 0.35)), f1=float(g(0.3, 0.9)),
                         f2=float(g(0.3, 0.9)), p1=float(g(0, 6.28)), p2=float(g(0, 6.28))),
        )

        # ---- 1~3 个运动刚体 ----
        n_obj = int(torch.randint(1, 4, (1,), generator=rng))
        self.objects = []  # (kind, O0, vel, osc_amp, osc_w, osc_phi, rot_axis, ang_vel, surface)
        for i in range(n_obj):
            kind = "sphere" if torch.rand(1, generator=rng) < 0.4 else "box"
            color = torch.stack([g(0.2, 0.8), g(0.2, 0.8), g(0.2, 0.8)]).flatten()
            tex = dict(kind="waves", amp=float(g(0.15, 0.4)), f1=float(g(2.0, 6.0)),
                       f2=float(g(2.0, 6.0)), p1=float(g(0, 6.28)), p2=float(g(0, 6.28)))
            O0 = torch.tensor([float(g(-2.5, 2.5)), float(g(-1.5, 1.5)), float(g(-7.0, -3.5))])
            vel = torch.tensor([float(g(-0.02, 0.02)), float(g(-0.01, 0.01)), float(g(-0.01, 0.01))])
            osc_amp = torch.tensor([float(g(0.0, 0.6)), float(g(0.0, 0.3)), float(g(0.0, 0.3))])
            osc_w = torch.tensor([float(g(0.01, 0.08)), float(g(0.01, 0.08)), float(g(0.01, 0.08))])
            osc_phi = torch.tensor([float(g(0, 6.28)), float(g(0, 6.28)), float(g(0, 6.28))])
            axis = torch.tensor([float(g(-1, 1)), float(g(-1, 1)), float(g(-1, 1))])
            axis = axis / (axis.norm() + 1e-8)
            ang_vel = float(g(-0.03, 0.03))
            if kind == "box":
                surf = BoxSurface(half_extents=[float(g(0.3, 0.8)), float(g(0.3, 0.8)), float(g(0.3, 0.8))],
                                  color=color, texture=tex)
            else:
                surf = SphereSurface(S=O0.clone(), r=float(g(0.25, 0.6)), color=color, texture=tex)
            self.objects.append(dict(kind=kind, O0=O0, vel=vel, osc_amp=osc_amp, osc_w=osc_w,
                                     osc_phi=osc_phi, axis=axis, ang_vel=ang_vel, surface=surf))

    # ---- 运动学 ----
    def cam_pose(self, t):
        """t (标量): 返回 C (3,), R_c2w (3,3)"""
        C = self.C0 + self.amp * torch.sin(self.w * t + self.phi)
        return C, look_at(C, self.look_target)

    def object_pose(self, obj, t):
        """刚体位姿 O(t) (3,), R(t) (3,3)"""
        O = obj["O0"] + obj["vel"] * t + obj["osc_amp"] * torch.sin(obj["osc_w"] * t + obj["osc_phi"])
        ang = obj["ang_vel"] * t
        R = _rot_matrix(obj["axis"], ang)
        return O, R

    def object_velocity(self, obj, X, t, eps=0.5):
        """X (K,3) 为刚体上 t 时刻的点, 求世界系速度 (中心差分)"""
        O, R = self.object_pose(obj, t)
        X_local = (R.T @ (X - O).T).T  # (K, 3)
        Op, Rp = self.object_pose(obj, t + eps)
        Om, Rm = self.object_pose(obj, t - eps)
        Xp = Op + (Rp @ X_local.T).T
        Xm = Om + (Rm @ X_local.T).T
        return (Xp - Xm) / (2 * eps)

    # ---- 渲染 ----
    def render_frame(self, t):
        """渲染第 t 帧: 返回 dict (均为 (H,W) 或 (H,W,3) 张量)"""
        C, R_c2w = self.cam_pose(t)
        rays = build_rays(self.H, self.W, self.f_norm, C, R_c2w)
        best_depth = torch.full((self.H, self.W), torch.inf)
        best_rgb = torch.zeros(self.H, self.W, 3)
        best_normal = torch.zeros(self.H, self.W, 3)
        best_obj = torch.full((self.H, self.W), -1, dtype=torch.long)

        res = self.bg.render(C, rays)
        ok = res["valid"] & (res["depth"] < best_depth)
        best_depth = torch.where(ok, res["depth"], best_depth)
        best_rgb = torch.where(ok.unsqueeze(-1), res["rgb"], best_rgb)
        best_normal = torch.where(ok.unsqueeze(-1), res["normal"], best_normal)
        best_obj = torch.where(ok, 0, best_obj)  # 0 = 背景

        for i, obj in enumerate(self.objects):
            O, R = self.object_pose(obj, t)
            if obj["kind"] == "box":
                res = obj["surface"].render(C, rays, O, R)
            else:
                res = obj["surface"].render(C, rays, O, R)
            ok = res["valid"] & (res["depth"] < best_depth)
            best_depth = torch.where(ok, res["depth"], best_depth)
            best_rgb = torch.where(ok.unsqueeze(-1), res["rgb"], best_rgb)
            best_normal = torch.where(ok.unsqueeze(-1), res["normal"], best_normal)
            best_obj = torch.where(ok, i + 1, best_obj)

        X = C + best_depth.unsqueeze(-1) * rays  # (H, W, 3)
        return {"rgb": best_rgb, "depth": best_depth, "X_world": X,
                "normal_world": best_normal, "obj_id": best_obj,
                "C": C, "R_c2w": R_c2w}

    def render_all(self, T):
        """渲染 T 帧, 堆叠为 dict; 并计算每帧每像素世界系速度"""
        frames = [self.render_frame(t) for t in range(T)]
        out = {k: torch.stack([f[k] for f in frames]) for k in
               ["rgb", "depth", "X_world", "normal_world", "obj_id"]}
        out["C"] = torch.stack([f["C"] for f in frames])
        out["R_c2w"] = torch.stack([f["R_c2w"] for f in frames])
        # 导出运动学参数 (查询 GT 需要跨帧变换刚体上的点)
        out["obj_params"] = [
            {k: torch.as_tensor(obj[k]).clone() for k in
             ["O0", "vel", "osc_amp", "osc_w", "osc_phi", "axis", "ang_vel"]}
            for obj in self.objects
        ]
        # 速度: 静态背景为 0, 运动刚体中心差分
        vel = torch.zeros(T, self.H, self.W, 3)
        for t in range(T):
            for i, obj in enumerate(self.objects):
                mask = out["obj_id"][t] == i + 1
                X = out["X_world"][t][mask]
                if X.shape[0] > 0:
                    vel[t][mask] = self.object_velocity(obj, X, float(t))
        out["vel_world"] = vel
        return out


def _rot_matrix(axis, ang):
    """绕单位轴 axis 旋转 ang 弧度的 3×3 矩阵 (Rodrigues); ang 为标量 float"""
    a = axis / (axis.norm() + 1e-8)
    K = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return torch.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


# ---------------------------------------------------------------------------
# 数据集: 渲染 + 增强 + 查询 GT
# ---------------------------------------------------------------------------

class SyntheticDataset:
    """无限合成数据流: 每步生成一个新场景样本 (论文附录 A 的增强管线)"""

    def __init__(self, num_frames_out: int = 12, num_frames_full: int = 24,
                 render_res: int = 256, out_res: int = 160,
                 f_norm_range=(0.8, 1.2), seed: int = 0, device="cpu"):
        self.T = num_frames_out
        self.T_full = num_frames_full
        self.render_res = render_res
        self.out_res = out_res
        self.f_norm_range = f_norm_range
        self.device = device
        self.rng = torch.Generator().manual_seed(seed)

    def make_sample(self, max_tries: int = 8):
        """生成一个样本 dict (所有 GT 保持在原始渲染分辨率)

        渲染器在极端配置下偶发数值溢出 (如球面 silhouette 处浮点误差),
        这里防御性检查 RGB 有限性, 非有限则重试 (概率极低)。
        """
        for _ in range(max_tries):
            sample = self._make_sample_once()
            if torch.isfinite(sample["video"]).all():
                return sample
        raise RuntimeError(f"连续 {max_tries} 次生成样本含非有限值, 检查渲染器数值稳定性")

    def _make_sample_once(self):
        """生成一个样本 dict (所有 GT 保持在原始渲染分辨率)"""
        g = self.rng
        f_norm = float(self.f_norm_range[0] + (self.f_norm_range[1] - self.f_norm_range[0])
                       * torch.rand(1, generator=g))
        scene = SyntheticScene(self.render_res, self.render_res, f_norm, rng=g)
        data = scene.render_all(self.T_full)

        # 1. 时序子采样 (论文附录 A: 随机步长)
        t_map = temporal_subsample(self.T_full, self.T, rng=g)
        for k in ["rgb", "depth", "X_world", "normal_world", "obj_id", "vel_world", "C", "R_c2w"]:
            data[k] = data[k][t_map]

        # 2. 随机裁剪 (尺度 0.3–1.0, 对数均匀宽高比, p=0.05 zoom)
        x0, y0, cw, ch = sample_crop_params(self.render_res, self.render_res, rng=g)
        video = apply_crop(data["rgb"].permute(0, 3, 1, 2), x0, y0, cw, ch, self.out_res)

        # 3. 时序一致色彩增强 (仅作用于 RGB)
        video = apply_color_augment(video, rng=g)

        # 4. 边界掩码 (深度 + 运动 Sobel, 论文附录 A 的困难采样) — 原分辨率计算后映射到增强空间
        mask_depth = depth_sobel_boundary_mask(data["depth"])
        mask_motion = depth_sobel_boundary_mask(data["vel_world"].norm(dim=-1))
        mask_orig = (mask_depth | mask_motion).float()  # (T, H0, W0)
        mask_crop = mask_orig[:, y0 : y0 + ch, x0 : x0 + cw]
        mask_aug = F.interpolate(mask_crop.unsqueeze(1), size=(self.out_res, self.out_res),
                                 mode="nearest").squeeze(1) > 0.5  # (T, S, S)

        return {
            "video": video,                      # (T, 3, S, S) 增强后
            "t_map": t_map,                      # (T,)
            "crop": (x0, y0, cw, ch),            # 原始像素坐标裁剪框
            "aspect": cw / ch,                   # 裁剪宽高比 (宽高比 token 输入)
            "boundary_mask": mask_aug,           # (T, S, S) bool
            # GT 地图 (原始渲染分辨率, 已时序子采样):
            "depth": data["depth"],              # (T, H0, W0) 光线深度
            "X_world": data["X_world"],          # (T, H0, W0, 3)
            "normal_world": data["normal_world"],
            "vel_world": data["vel_world"],
            "obj_id": data["obj_id"],            # (T, H0, W0) long, 0=背景
            "C": data["C"],                      # (T, 3) 相机中心
            "R_c2w": data["R_c2w"],              # (T, 3, 3)
            "f_norm": f_norm,
            "obj_params": data["obj_params"],  # 运动学参数 (查询 GT 刚体变换用)
        }

    # ---- 查询与 GT ----

    def sample_queries(self, sample, N: int, boundary_ratio: float = 0.3,
                       p_tcam_eq: float = 0.4, rng=None):
        """论文附录 A 的查询采样: N 个查询, 30% 在边界附近, ttgt=tcam 概率 0.4

        返回 (queries, targets, mask): queries 为模型输入 dict,
        targets 为监督 dict (GT 语义约定见模块 docstring)。
        """
        if rng is None:
            rng = self.rng
        T, S = sample["video"].shape[0], sample["video"].shape[-1]

        t_src = torch.randint(0, T, (N,), generator=rng)
        t_tgt = torch.randint(0, T, (N,), generator=rng)
        t_cam = torch.randint(0, T, (N,), generator=rng)
        # 40% 强制 ttgt = tcam (论文附录 A)
        force = torch.rand(N, generator=rng) < p_tcam_eq
        t_cam = torch.where(force, t_tgt, t_cam)

        # 30% 查询采样在深度/运动边界附近
        uv = torch.rand(N, 2, generator=rng)
        use_boundary = torch.rand(N, generator=rng) < boundary_ratio
        for t in range(T):
            idx = torch.nonzero(use_boundary & (t_src == t)).flatten()
            if idx.numel() == 0:
                continue
            cand = torch.nonzero(sample["boundary_mask"][t])  # (K, 2) [y, x]
            if cand.numel() == 0:
                continue
            pick = cand[torch.randint(0, cand.shape[0], (idx.numel(),), generator=rng)]
            uv[idx] = torch.stack([(pick[:, 1] + 0.5) / S, (pick[:, 0] + 0.5) / S], dim=-1)

        queries = {"uv": uv, "t_src": t_src, "t_tgt": t_tgt, "t_cam": t_cam}
        targets, mask = self.compute_gt(sample, queries)
        return queries, targets, mask

    def compute_gt(self, sample, queries):
        """对任意查询集计算解析 GT (训练采样与评测共用)

        语义约定: p_3d/p_2d/disp/normal = 目标时刻 ttgt 的状态、
        表达在目标相机 tcam 坐标系; vis = 点在 ttgt 帧是否可见 (z-buffer)。
        返回 (targets, mask), mask=False 表示 GT 无效 (光线未击中几何体)。
        """
        uv, t_src = queries["uv"], queries["t_src"]
        t_tgt, t_cam = queries["t_tgt"], queries["t_cam"]
        H0, W0 = self.render_res, self.render_res

        # 映射回原始渲染像素, 双线性采样原始分辨率 GT 地图
        px, py = uv_to_orig_pixel(uv, *sample["crop"])
        g = self._bilinear_gather(
            sample["X_world"], sample["normal_world"], sample["obj_id"],
            t_src, px, py, H0, W0,
        )
        X_s, n_s = g["X"], g["normal"]
        obj_s = g["obj"].long()

        # 物体运动使用原视频中的时间索引，避免随机步长子采样后 GT 与渲染错位。
        actual_src = sample["t_map"][t_src]
        actual_tgt = sample["t_map"][t_tgt]
        # 物体运动: X 从 t_src 时刻变换到 t_tgt 时刻 (刚体: 局部坐标不变)
        X_tgt = X_s.clone()
        n_tgt = n_s.clone()
        for i in range(len(sample["obj_params"])):
            m = obj_s == i + 1
            if not m.any():
                continue
            O_s, R_s = obj_pose_fn(sample, i, actual_src[m])
            O_t, R_t = obj_pose_fn(sample, i, actual_tgt[m])
            X_local = torch.bmm(R_s.transpose(1, 2), (X_s[m] - O_s).unsqueeze(-1))
            X_tgt[m] = O_t + torch.bmm(R_t, X_local).squeeze(-1)
            n_tgt[m] = torch.bmm(R_t @ R_s.transpose(1, 2), n_s[m].unsqueeze(-1)).squeeze(-1)

        # 转换到目标相机 t_cam 坐标系
        R_w2c = sample["R_c2w"].transpose(1, 2)  # (T, 3, 3)
        X_tgt_cam = torch.bmm(R_w2c[t_cam], (X_tgt - sample["C"][t_cam]).unsqueeze(-1)).squeeze(-1)
        n_tgt_cam = torch.bmm(R_w2c[t_cam], n_tgt.unsqueeze(-1)).squeeze(-1)
        disp_cam = torch.bmm(R_w2c[t_cam], (X_tgt - X_s).unsqueeze(-1)).squeeze(-1)

        # p_2d: 先投影到原图，再映到实际输入视频的裁剪归一化坐标。
        # 查询 uv 本来就是裁剪后的坐标，两者必须使用同一坐标系。
        f = sample["f_norm"]
        p_2d_orig = torch.stack([
            0.5 + f * X_tgt_cam[:, 0] / X_tgt_cam[:, 2],
            0.5 + f * X_tgt_cam[:, 1] / X_tgt_cam[:, 2],
        ], dim=-1)
        x0, y0, cw, ch = sample["crop"]
        p_2d = torch.stack([(p_2d_orig[:, 0] * W0 - x0) / cw,
                            (p_2d_orig[:, 1] * H0 - y0) / ch], dim=-1)

        # vis: X_tgt 在 ttgt 帧的遮挡测试 (z-buffer)
        # 注意深度图存的是"光线深度" (t 参数), 故 d_pred 也取光线深度 (到相机中心的欧氏距离)
        d_pred = (X_tgt - sample["C"][t_tgt]).norm(dim=-1)
        X_cam_ttgt = torch.bmm(
            R_w2c[t_tgt], (X_tgt - sample["C"][t_tgt]).unsqueeze(-1)
        ).squeeze(-1)
        u2 = 0.5 + f * X_cam_ttgt[:, 0] / X_cam_ttgt[:, 2]
        v2 = 0.5 + f * X_cam_ttgt[:, 1] / X_cam_ttgt[:, 2]
        u2_crop = (u2 * W0 - x0) / cw
        v2_crop = (v2 * H0 - y0) / ch
        in_view = ((X_cam_ttgt[:, 2] > 1e-3) & (u2_crop >= 0) & (u2_crop <= 1)
                   & (v2_crop >= 0) & (v2_crop <= 1))
        u2 = u2.clamp(0, 1)
        v2 = v2.clamp(0, 1)
        pxi = (u2 * W0 - 0.5).round().long().clamp(0, W0 - 1)
        pyi = (v2 * H0 - 0.5).round().long().clamp(0, H0 - 1)
        d_stored = sample["depth"][t_tgt, pyi, pxi]
        vis = (in_view & ((d_stored - d_pred).abs() < (0.05 * d_stored + 0.01))).float().unsqueeze(-1)

        # 无效查询 (光线未击中任何几何体, GT 为 inf/NaN) 用 mask 标记,
        # 损失通过 _masked_mean 跳过这些项
        valid = (torch.isfinite(X_tgt_cam).all(-1) & torch.isfinite(p_2d).all(-1)
                 & torch.isfinite(disp_cam).all(-1) & torch.isfinite(n_tgt_cam).all(-1)
                 & torch.isfinite(d_pred))

        def _clean(t):
            return torch.where(torch.isfinite(t), t, torch.zeros_like(t))

        targets = {"p_3d": _clean(X_tgt_cam), "p_2d": _clean(p_2d),
                   "vis": torch.where(valid.unsqueeze(1), vis, torch.zeros_like(vis)),
                   "disp": _clean(disp_cam), "normal": _clean(n_tgt_cam)}
        return targets, valid.unsqueeze(1)

    @staticmethod
    def _bilinear_gather(Xmap, Nmap, Objmap, t, px, py, H0, W0):
        """在原始分辨率地图上按 (t, px, py) 双线性采样; 返回 dict"""
        px = px.clamp(0, W0 - 1)
        py = py.clamp(0, H0 - 1)
        x0 = px.floor().long()
        y0 = py.floor().long()
        x1 = (x0 + 1).clamp(max=W0 - 1)
        y1 = (y0 + 1).clamp(max=H0 - 1)
        wx = px - x0.float()
        wy = py - y0.float()

        def interp(M):
            s = M[t, y0, x0]
            # 权重升维与特征维对齐: (N,) → (N, 1[, 1]) 视地图是否带特征维
            shape = wx.shape + (1,) * (s.dim() - 1)
            wxx, wyy = wx.reshape(shape), wy.reshape(shape)
            return (M[t, y0, x0] * (1 - wxx) * (1 - wyy) + M[t, y0, x1] * wxx * (1 - wyy)
                    + M[t, y1, x0] * (1 - wxx) * wyy + M[t, y1, x1] * wxx * wyy)

        xn = px.round().long()
        yn = py.round().long()
        obj = Objmap[t, yn, xn]
        same_surface = ((Objmap[t, y0, x0] == obj) & (Objmap[t, y0, x1] == obj)
                        & (Objmap[t, y1, x0] == obj) & (Objmap[t, y1, x1] == obj))
        # 深度边界不能把前景和背景的 3D 点混合成不存在的点。
        return {"X": torch.where(same_surface[:, None], interp(Xmap), Xmap[t, yn, xn]),
                "normal": torch.where(same_surface[:, None], interp(Nmap), Nmap[t, yn, xn]),
                "obj": obj}


def obj_pose_fn(sample, obj_idx, t):
    """由存储的刚体运动参数重建位姿 (与渲染时一致)

    t: (K,) 帧索引张量。返回 O (K,3), R (K,3,3)。
    """
    p = sample["obj_params"][obj_idx]
    O0, vel, osc_amp, osc_w, osc_phi = p["O0"], p["vel"], p["osc_amp"], p["osc_w"], p["osc_phi"]
    axis, ang_vel = p["axis"], p["ang_vel"]
    tf = t.float()
    O = O0 + vel * tf.unsqueeze(-1) + osc_amp * torch.sin(osc_w * tf.unsqueeze(-1) + osc_phi)
    R = _rot_batched(axis, ang_vel * tf)
    return O, R


def _rot_batched(axis, ang):
    """批量 Rodrigues: axis (3,), ang (K,) → R (K,3,3)"""
    K = ang.shape[0]
    a = axis / (axis.norm() + 1e-8)
    Kmat = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    I = torch.eye(3)
    c, s = torch.cos(ang), torch.sin(ang)
    R = I.unsqueeze(0) + s.view(-1, 1, 1) * Kmat.unsqueeze(0) + \
        (1 - c).view(-1, 1, 1) * (Kmat @ Kmat).unsqueeze(0)
    return R
