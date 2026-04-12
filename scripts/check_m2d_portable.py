from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
M2D_ROOT = ROOT / "external" / "m2d"
WEIGHT_ROOT = ROOT / "weights" / "m2d"


def find_first_checkpoint(weight_root: Path) -> Path:
    ckpts = sorted(weight_root.rglob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(
            f"No .pth checkpoint found under {weight_root}. "
            "Download and unzip the M2D-AS encoder-only weight first."
        )
    return ckpts[0]


def main() -> None:
    if not M2D_ROOT.exists():
        raise FileNotFoundError(f"M2D submodule not found: {M2D_ROOT}")

    sys.path.insert(0, str(M2D_ROOT))

    # Imported after sys.path update
    from examples.portable_m2d import PortableM2D  # noqa: WPS433

    ckpt = find_first_checkpoint(WEIGHT_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] m2d_root   = {M2D_ROOT}")
    print(f"[INFO] checkpoint = {ckpt}")
    print(f"[INFO] device     = {device}")

    model = PortableM2D(str(ckpt))
    model = model.to(device)
    model.eval()

    # README example style: 10-second waveforms at 16 kHz, values in [-1, 1]
    batch_audio = 2.0 * torch.rand((2, 10 * 16000), device=device) - 1.0

    with torch.no_grad():
        frame_level = model(batch_audio)
        clip_level = frame_level.mean(dim=1)

    print(f"[INFO] input shape       = {tuple(batch_audio.shape)}")
    print(f"[INFO] frame-level shape = {tuple(frame_level.shape)}")
    print(f"[INFO] clip-level shape  = {tuple(clip_level.shape)}")
    print(f"[INFO] dtype             = {frame_level.dtype}")


if __name__ == "__main__":
    main()
