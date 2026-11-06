"""
Packs rendered frames (build/frames/<theme>) into src/animations/<theme>.anim with the firmware's own encoder.

Needs the firmware checkout for scripts/seq2anim.py and its Python (it has colorlog):
  <firmware>/toolchain/current/python/python.exe tools/pack.py [theme ...]
FIRMWARE points at the firmware checkout; by default busybar-firmware next to this repository.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = Path(os.environ.get("FIRMWARE", ROOT.parent / "busybar-firmware"))
sys.path.insert(0, str(FIRMWARE / "scripts"))

from seq2anim import BSBAnimConverter  # noqa: E402

FRAMES = ROOT / "build" / "frames"
OUT = ROOT / "src" / "animations"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
    for name in names:
        target = OUT / f"{name}.anim"
        info = BSBAnimConverter().convert_dir(FRAMES / name, target)
        print(f"{target.name}: {target.stat().st_size // 1024} KiB, {info.frame_cnt} file frames, "
              f"x{info.overall_compression_ratio:.1f} compression")


if __name__ == "__main__":
    main()
