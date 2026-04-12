from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "external" / "m2d" / "examples" / "portable_m2d.py"

OLD = "from timm.layers import trunc_normal_"
NEW = """try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_"""


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if NEW in text:
        print("[INFO] patch already applied")
        return

    if OLD not in text:
        raise RuntimeError(
            "Expected import line not found. "
            "portable_m2d.py may have changed upstream."
        )

    text = text.replace(OLD, NEW)
    TARGET.write_text(text, encoding="utf-8")
    print(f"[INFO] patch applied to {TARGET}")


if __name__ == "__main__":
    main()
