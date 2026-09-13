"""D4RT 训练脚本 (论文 §4 训练细节 + 附录 A 数据增强)

训练管线 (与论文逐项对应):
  - 查询采样: 每 clip 2048 个查询, 30% 采样在 Sobel 深度/运动边界附近,
    ttgt=tcam 以概率 0.4 强制 (实现于 data/synthetic.py 的 sample_queries)
  - 优化器: AdamW, weight_decay=0.03
  - 学习率: 2500 步 warmup 至峰值 1e-4 → cosine 退火至 1e-6
  - 梯度裁剪: 全局 L2 范数 10
  - 精度: bf16 autocast (GPU; CPU 回退 fp32)
  - 数据: SyntheticDataset 在线渲染合成动态场景 (8GB 显存/国内网络下的
    真实数据集替代品; 替换真实数据 loader 时保持同一接口即可)

用法:
  python train.py --config tiny --steps 500           # 本地 8GB 快速验证
  python train.py --config vit-g --steps 100000      # 论文主配置 (需大显存)
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data.synthetic import SyntheticDataset
from models import D4RTModel, D4RTLoss

# 各变体的默认训练分辨率/帧数 (tiny 为本地验证配置, 非论文)
DEFAULT_SHAPE = {
    "tiny": (160, 12),
    "vit-b": (256, 48),
    "vit-l": (256, 48),
    "vit-h": (256, 48),
    "vit-g": (256, 48),
}


def build_lr_schedule(optimizer, peak_lr, min_lr, warmup_steps, total_steps):
    """warmup 线性升至峰值 → cosine 退火至 min_lr (论文 §4)"""

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr / peak_lr + 0.5 * (1 - min_lr / peak_lr) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def save_training_state(path, step, model, optimizer, scheduler, dataset,
                        config, loss, best_loss, device):
    """先写临时文件再原子替换，WSL/终端断连时保留上一份完整检查点。"""
    state = {"step": step, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "data_rng": dataset.rng.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
             "config": config, "loss": loss, "best_loss": best_loss}
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(path)


def main():
    ap = argparse.ArgumentParser(description="D4RT 训练")
    ap.add_argument("--config", default="tiny", choices=list(DEFAULT_SHAPE),
                    help="ViT 变体 (tiny 为本地验证专用, 其余为论文 Tab.9 规模)")
    ap.add_argument("--steps", type=int, default=500, help="总训练步数")
    ap.add_argument("--queries", type=int, default=None, help="每 clip 查询数 (常规 2048; 低内存 512)")
    ap.add_argument("--frames", type=int, default=None, help="clip 帧数 (默认取配置值)")
    ap.add_argument("--res", type=int, default=None, help="训练分辨率 (默认取配置值)")
    ap.add_argument("--render-res", type=int, default=None, help="合成渲染分辨率")
    ap.add_argument("--lr-peak", type=float, default=1e-4, help="峰值学习率 (论文: 1e-4)")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="退火终值 (论文: 1e-6)")
    ap.add_argument("--warmup", type=int, default=2500, help="warmup 步数 (论文: 2500)")
    ap.add_argument("--wd", type=float, default=0.03, help="AdamW weight_decay (论文: 0.03)")
    ap.add_argument("--grad-clip", type=float, default=10.0, help="梯度 L2 裁剪 (论文: 10)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--out", default="checkpoints", help="检查点目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-initial", action="store_true",
                    help="保存 step=0 权重，用于同一初始化的训练前后评测")
    ap.add_argument("--low-memory", action="store_true",
                    help="WSL 低内存预设: tiny 96px/8帧/128px渲染/512查询")
    ap.add_argument("--force-high-memory", action="store_true",
                    help="在不足 10GiB RAM 的 WSL 中显式允许高内存配置")
    ap.add_argument("--cpu-threads", type=int, default=4,
                    help="限制 CPU 线程数，降低 WSL 渲染峰值内存 (默认 4)")
    ap.add_argument("--resume", default=None, help="从完整的 *_last.pt 检查点继续训练")
    args = ap.parse_args()

    if args.low_memory:
        if args.config != "tiny":
            ap.error("--low-memory 仅支持 tiny 配置")
        args.res = 96 if args.res is None else args.res
        args.frames = 8 if args.frames is None else args.frames
        args.ckpt_every = min(args.ckpt_every, 25)
    if args.render_res is None:
        args.render_res = 128 if args.low_memory else 256
    if args.queries is None:
        args.queries = 512 if args.low_memory else 2048
    if min(args.steps, args.queries, args.render_res, args.cpu_threads,
           args.ckpt_every, args.log_every) < 1:
        ap.error("步数、查询数、分辨率、线程数、日志和检查点间隔必须为正整数")
    torch.set_num_threads(args.cpu_threads)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: 未检测到 CUDA, 使用 CPU (bf16 退化为 fp32)")

    res, frames = DEFAULT_SHAPE[args.config]
    if args.res is not None:
        res = args.res
    if args.frames is not None:
        frames = args.frames
    if frames < 2 or frames % 2 or res < 16 or res % 16:
        ap.error("帧数须为不小于 2 的偶数，训练分辨率须为 16 的整数倍")
    if args.render_res < res:
        ap.error("合成渲染分辨率不能小于训练分辨率")
    mem_total_kib = 0
    try:
        mem_total_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemTotal:")))
    except (OSError, StopIteration, ValueError):
        pass
    high_memory = frames >= 12 and res >= 160 and args.render_res >= 256 and args.queries >= 2048
    if mem_total_kib and mem_total_kib < 10 * 1024 * 1024 and high_memory and not args.force_high_memory:
        ap.error("当前 WSL RAM 不足 10GiB；此配置曾触发 OOM。请加 --low-memory，"
                 "或确认已增大 WSL 内存后显式传 --force-high-memory")
    # 本地短跑时把 warmup 限制在总步数的 1/4 内, 保证 LR 能爬到峰值
    warmup = min(args.warmup, args.steps // 4)

    model = D4RTModel(variant=args.config, image_size=res,
                      num_frames=frames, max_frames=frames).to(device)
    criterion = D4RTLoss()
    ds = SyntheticDataset(num_frames_out=frames, num_frames_full=max(2 * frames, 16),
                          render_res=args.render_res, out_res=res, seed=args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, weight_decay=args.wd)
    scheduler = build_lr_schedule(optimizer, args.lr_peak, args.lr_min, warmup, args.steps)

    counts = model.param_counts()
    cfg = {"config": args.config, "res": res, "frames": frames,
           "steps": args.steps, "queries": args.queries,
           "lr_peak": args.lr_peak, "lr_min": args.lr_min, "warmup": warmup,
           "wd": args.wd, "grad_clip": args.grad_clip, "seed": args.seed,
           "render_res": args.render_res, "torch_version": torch.__version__,
           "device": str(device), "low_memory": args.low_memory,
           "cpu_threads": args.cpu_threads}
    print(f"[D4RT] 配置 {args.config}: 编码器 {counts['encoder']/1e6:.1f}M, "
          f"解码器+查询 {counts['decoder+query']/1e6:.1f}M | 分辨率 {res}, 帧数 {frames}, "
          f"设备 {device}")
    print(f"[D4RT] 调度: warmup {warmup} 步 → {args.lr_peak} → cosine → {args.lr_min}; "
          f"AdamW wd={args.wd}, 梯度裁剪 {args.grad_clip}")

    use_bf16 = device == "cuda"
    best_loss = float("inf")
    nan_steps = 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"
    start_step = 0
    last_loss = float("nan")
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.parent.resolve() != out_dir.resolve():
            ap.error("--resume 检查点必须位于 --out 目录中")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        keys = ("config", "res", "frames", "steps", "queries", "lr_peak", "lr_min",
                "warmup", "wd", "grad_clip", "seed", "render_res")
        different = [key for key in keys if checkpoint["config"].get(key) != cfg.get(key)]
        if different:
            ap.error(f"续训配置与检查点不一致: {', '.join(different)}")
        required = ("optimizer", "scheduler", "data_rng", "torch_rng")
        if any(key not in checkpoint for key in required):
            ap.error("此检查点缺少完整训练状态，不能精确续训")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ds.rng.set_state(checkpoint["data_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_step = int(checkpoint["step"])
        best_loss = float(checkpoint.get("best_loss", checkpoint["loss"]))
        last_loss = float(checkpoint["loss"])
        del checkpoint
        if not metrics_path.exists():
            ap.error("续训所需 train_metrics.jsonl 不存在")
        previous = [line for line in metrics_path.read_text().splitlines()
                    if int(json.loads(line)["step"]) <= start_step]
        metrics_path.write_text("\n".join(previous) + ("\n" if previous else ""))
        print(f"[resume] 已恢复 step={start_step}，后续日志追加到 {metrics_path}")
    else:
        (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        if args.save_initial:
            torch.save({"step": 0, "model": model.state_dict(), "config": cfg},
                       out_dir / f"{args.config}_initial.pt")

    model.train()
    with metrics_path.open("a" if args.resume else "w") as metrics_file:
      for step in range(start_step, args.steps):
        t0 = time.time()
        # 在线渲染一个增强样本 + 采样 2048 个查询的完整 GT (含遮挡可见性)
        sample = ds.make_sample()
        queries, targets, mask = ds.sample_queries(sample, N=args.queries)
        video = sample["video"].unsqueeze(0).to(device)
        queries = {k: v.to(device) for k, v in queries.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            preds = model(video, queries, aspect_ratio=sample["aspect"])
        # 损失在 autocast 外以 fp32 计算 (BCE 等算子不允许 autocast)
        loss, terms = criterion(preds, targets, mask)
        # 防御: 单个坏样本 (渲染偶发溢出/数值病态) 不应杀死整轮训练
        if not torch.isfinite(loss):
            nan_steps += 1
            if nan_steps > 20:
                raise RuntimeError("连续 20 步损失非有限, 训练发散 (非数据问题), 终止")
            if step == 0 or nan_steps % 5 == 1:
                print(f"step {step+1}: 损失非有限 ({loss.item():.3f}), 跳过本步 (累计 {nan_steps})")
            continue
        nan_steps = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_v = loss.item()
        last_loss = loss_v
        best_loss = min(best_loss, loss_v)
        record = {"step": step + 1, "loss": loss_v,
                  "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.time() - t0,
                  **{k: float(v.detach()) for k, v in terms.items()}}
        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_file.flush()
        if (step + 1) % args.log_every == 0 or step == 0:
            lr = optimizer.param_groups[0]["lr"]
            t_str = " ".join(f"{k.replace('loss_', '')}={v:.3f}" for k, v in terms.items())
            print(f"step {step+1}/{args.steps} | lr={lr:.2e} | total={loss_v:.3f} "
                  f"| {t_str} | {time.time()-t0:.1f}s")

        if (step + 1) % args.ckpt_every == 0:
            save_training_state(out_dir / f"{args.config}_last.pt", step + 1,
                                model, optimizer, scheduler, ds, cfg,
                                loss_v, best_loss, device)
            if loss_v == best_loss:
                torch.save({"step": step + 1, "model": model.state_dict(), "config": cfg,
                            "loss": loss_v},
                           out_dir / f"{args.config}_best.pt")
            print(f"[ckpt] step {step+1} 已保存 → {args.out}/")

    if not math.isfinite(last_loss):
        raise RuntimeError("没有完成任何有效训练步，未写入最终检查点")
    save_training_state(out_dir / f"{args.config}_last.pt", args.steps,
                        model, optimizer, scheduler, ds, cfg,
                        last_loss, best_loss, device)
    print(f"[D4RT] 训练完成: 最终 loss={last_loss:.3f}, 历史最优={best_loss:.3f}")


if __name__ == "__main__":
    main()
