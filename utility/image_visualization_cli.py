#!/usr/bin/env python3
"""Overlay stixel labels from a .stx1 file onto an image.

The script loads a JPEG image and a corresponding StixelWorld (.stx1)
file and visualizes the stixels on top of the image. If an output path
is provided the resulting visualization is written to that file,
otherwise the image is displayed in a window.
"""
import argparse
from pathlib import Path
from PIL import Image
try:  # pyStixel-lib provides the stixel package
    import stixel  # type: ignore
    from stixel.utils import draw_stixels_on_image
except ImportError as exc:  # pragma: no cover - library not installed
    raise SystemExit(
        "pyStixel-lib is required for visualizing .stx1 files."
        " Install it with 'pip install pyStixel-lib'."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a StixelWorld on a given image")
    parser.add_argument("--image", required=True, help="Input JPEG image")
    parser.add_argument("--stx", required=True, help="StixelWorld .stx1 file")
    parser.add_argument("--out", help="Path to save the overlay image")
    parser.add_argument("--instances", action="store_true",
                        help="Color by cluster labels instead of depth")
    args = parser.parse_args()

    img = Image.open(args.image)
    stxl_world = stixel.read(args.stx)
    result = draw_stixels_on_image(stxl_world, img, instances=args.instances)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.save(args.out)
        print(f"Saved visualization to {args.out}")
    else:
        result.show()


if __name__ == "__main__":
    main()