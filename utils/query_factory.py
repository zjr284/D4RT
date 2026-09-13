"""下游任务查询构造器 (论文 Tab.1: Unified decoding)

同一个解码接口, 不同任务的差异只在于查询参数 (u, v, t_src, t_tgt, t_cam) 的组合:
  Point Track : u,v,tsrc 固定, ttgt = tcam = 1..T
  Point Cloud : u=1..W, v=1..H, tsrc=1..T, ttgt 固定 (=tsrc), tcam 固定 (参考帧)
  Depth Map   : u=1..W, v=1..H, tsrc = ttgt = tcam = 1..T
  Extrinsics  : 粗 (h,w) 网格, 两套参考帧查询 q_i=(u,v,i,i,i), q_j=(u,v,i,i,j)
  Intrinsics  : 粗 (h,w) 网格, tsrc = ttgt = tcam = i
"""

import torch


class D4RTQueryFactory:
    """自动化构建不同下游任务的 Query 输入矩阵 (参考 Table 1)"""

    @staticmethod
    def build_depth_queries(T: int, H: int, W: int, device="cpu"):
        """深度图任务: u=1..W, v=1..H, tsrc = ttgt = tcam = 1...T

        返回每帧一个查询 dict; 解码 p_3d 的 Z 分量即该帧深度图。
        """
        v = (torch.arange(H, device=device) + 0.5) / H
        u = (torch.arange(W, device=device) + 0.5) / W
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        uv_base = torch.stack([gu.flatten(), gv.flatten()], dim=-1)

        queries_list = []
        for t in range(T):
            N = uv_base.shape[0]
            t_seq = torch.full((N,), t, dtype=torch.long, device=device)
            queries_list.append(
                {"uv": uv_base, "t_src": t_seq, "t_tgt": t_seq, "t_cam": t_seq}
            )
        return queries_list

    @staticmethod
    def build_point_cloud_queries(T: int, H: int, W: int, ref_cam: int = 0, device="cpu"):
        """点云重建任务: tsrc = ttgt = 1...T, tcam 统一固定在参考帧视角

        所有像素的 3D 位置直接预测在共享参考帧 tcam 下 (论文 §2.1:
        免去显式坐标变换, 避免噪声相机估计带来的误差传播)。
        """
        v = (torch.arange(H, device=device) + 0.5) / H
        u = (torch.arange(W, device=device) + 0.5) / W
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        uv_base = torch.stack([gu.flatten(), gv.flatten()], dim=-1)

        queries_list = []
        for t in range(T):
            N = uv_base.shape[0]
            t_seq = torch.full((N,), t, dtype=torch.long, device=device)
            t_cam = torch.full((N,), ref_cam, dtype=torch.long, device=device)
            queries_list.append(
                {"uv": uv_base, "t_src": t_seq, "t_tgt": t_seq, "t_cam": t_cam}
            )
        return queries_list

    @staticmethod
    def build_point_track_queries(uv_pt: torch.Tensor, t_src_idx: int, T: int, device="cpu"):
        """单点轨迹追踪: 固定 (u, v) 及 tsrc, 令 ttgt = tcam = 1...T"""
        uv = uv_pt.unsqueeze(0).repeat(T, 1)  # (T, 2)
        t_src = torch.full((T,), t_src_idx, dtype=torch.long, device=device)
        t_tgt_cam = torch.arange(T, dtype=torch.long, device=device)

        return {
            "uv": uv,
            "t_src": t_src,
            "t_tgt": t_tgt_cam,
            "t_cam": t_tgt_cam,
        }

    @staticmethod
    def build_extrinsics_queries(frame_i: int, frame_j: int, grid_size=(16, 16), device="cpu"):
        """外参任务 (论文 §2.1): 返回 (q_i, q_j)

        q_i = (u,v,i,i,i) 帧 i 坐标系; q_j = (u,v,i,i,j) 帧 j 坐标系。
        两套解码结果用 Umeyama 求刚性变换 (见 utils.camera)。
        """
        h, w = grid_size
        v = torch.linspace(0.05, 0.95, h, device=device)
        u = torch.linspace(0.05, 0.95, w, device=device)
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
        K = uv.shape[0]

        t_i = torch.full((K,), frame_i, dtype=torch.long, device=device)
        t_j = torch.full((K,), frame_j, dtype=torch.long, device=device)
        q_i = {"uv": uv, "t_src": t_i, "t_tgt": t_i, "t_cam": t_i}
        q_j = {"uv": uv, "t_src": t_i, "t_tgt": t_i, "t_cam": t_j}
        return q_i, q_j

    @staticmethod
    def build_intrinsics_queries(frame_i: int, grid_size=(16, 16), device="cpu"):
        """内参任务 (论文 §2.1): tsrc = ttgt = tcam = i 的粗网格查询"""
        h, w = grid_size
        v = torch.linspace(0.1, 0.9, h, device=device)
        u = torch.linspace(0.1, 0.9, w, device=device)
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
        K = uv.shape[0]

        t_seq = torch.full((K,), frame_i, dtype=torch.long, device=device)
        return {"uv": uv, "t_src": t_seq, "t_tgt": t_seq, "t_cam": t_seq}
