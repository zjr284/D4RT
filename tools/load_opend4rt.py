"""Load and verify an OpenD4RT checkpoint with limited host memory.

The model is constructed on the meta device and its parameters are assigned
directly from a memory-mapped checkpoint. This avoids allocating a second
full-size copy of the ViT-g weights in RAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_D4RT_ROOT = PROJECT_ROOT / "external" / "open-d4rt"
DEFAULT_VARIANT = OPEN_D4RT_ROOT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG"


def load_model(checkpoint: Path, config: Path) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Model config not found: {config}")

    sys.path.insert(0, str(OPEN_D4RT_ROOT))
    from src.core.config import load_yaml_config
    from src.model import build_model

    cfg = load_yaml_config(config)
    with torch.device("meta"):
        model = build_model(cfg["model"])

    # weights_only avoids executing arbitrary pickle objects in the checkpoint.
    # The released training checkpoint includes optimizer state (~14 GB).
    # MAP_PRIVATE reserves too much commit space on memory-limited WSL VMs;
    # MAP_SHARED is read-only in practice here and does not reserve that copy.
    with torch.serialization.set_default_mmap_options(torch.serialization.MAP_SHARED):
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary checkpoint")
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint contains no model state dictionary")

    # strict=True catches architecture/config mismatches instead of silently
    # leaving parts of the model randomly initialized.
    model.load_state_dict(state, strict=True, assign=True)
    del state, payload
    # Fourier frequencies are a non-persistent buffer, so state_dict does not
    # replace their meta placeholder. Recreate them on CPU before device moves.
    uv_encoder = model.query_embedder.uv_encoder
    uv_encoder.frequencies = (2.0 ** torch.arange(uv_encoder.num_bands)) * torch.pi
    model.eval()
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Some model tensors were not materialized")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_VARIANT / "opend4rt.ckpt")
    parser.add_argument("--config", type=Path, default=DEFAULT_VARIANT / "model.yaml")
    parser.add_argument("--smoke-forward", action="store_true", help="Run one tiny BF16 CUDA query")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)
    first_name, first_parameter = next(model.named_parameters())
    # Reading one value verifies that the assigned tensor really refers to
    # checkpoint data and not only to metadata on the meta device.
    sample = float(first_parameter.detach().reshape(-1)[0])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded: {args.checkpoint}")
    print(f"Parameters: {parameter_count:,}")
    print(f"First parameter: {first_name} = {sample:.6g}")
    print("All checkpoint keys and tensor shapes match the model config.")

    if args.smoke_forward:
        if not torch.cuda.is_available():
            raise RuntimeError("--smoke-forward requires CUDA")
        model = model.to(device="cuda", dtype=torch.bfloat16)
        video = torch.zeros((1, 2, 3, 32, 32), device="cuda", dtype=torch.bfloat16)
        uv = torch.full((1, 1), 0.5, device="cuda", dtype=torch.bfloat16)
        t = torch.zeros((1, 1), device="cuda", dtype=torch.long)
        query = {"u": uv, "v": uv, "t_src": t, "t_tgt": t, "t_cam": t}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model({"video": video, "query": query})
        shapes = {name: tuple(value.shape) for name, value in output.items()}
        if not all(torch.isfinite(value).all().item() for value in output.values()):
            raise RuntimeError("Smoke forward produced non-finite values")
        print(f"Tiny CUDA forward passed: {shapes}")


if __name__ == "__main__":
    main()
