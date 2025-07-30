#!/usr/bin/env python3
"""Overlay stixel labels from a .stx1 file onto an image.

The script loads a JPEG image and a corresponding StixelWorld (.stx1)
file and visualizes the stixels on top of the image. If an output path
is provided the resulting visualization is written to that file,
otherwise the image is displayed in a window.
"""
import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import importlib.util
import types

import numpy as np
from PIL import Image

try:  # pyStixel-lib provides the stixel package
    import stixel  # type: ignore
except ImportError as exc:  # pragma: no cover - library not installed
    raise SystemExit(
        "pyStixel-lib is required for visualizing .stx1 files."
        " Install it with 'pip install pyStixel-lib'."
    ) from exc

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _prepare_libraries(root: Path):
    """Load visualization helpers from ``root``."""
    sys.path.append(str(root))
    stixel_mod = _load_module("libraries.Stixel", root / "libraries" / "Stixel.py")

    dummy = types.ModuleType("libraries")
    dummy.__path__ = [str(root / "libraries")]
    dummy.StixelClass = stixel_mod.StixelClass
    dummy.Stixel = stixel_mod.Stixel
    sys.modules.setdefault("libraries", dummy)
    sys.modules.setdefault("libraries.Stixel", stixel_mod)

    viz_mod = _load_module("libraries.visualization", root / "libraries" / "visualization.py")

    return stixel_mod, viz_mod


def _to_stixels(
    world: "stixel.StixelWorld", img_size: dict, stixel_mod
) -> Tuple[List[object], int]:
    """Convert a ``stixel.StixelWorld`` to a list of local ``Stixel`` objects.

    Parameters
    ----------
    world:
        Parsed ``stixel.StixelWorld`` protobuf.
    img_size:
        ``{"width": int, "height": int}`` dictionary describing the size of the
        image used for visualization. The dimensions inside the stixel file may
        be invalid, so use the actual image size instead.
    stixel_mod:
        Loaded ``libraries.Stixel`` module providing the ``Stixel`` class.
    """
    Stixel = stixel_mod.Stixel
    StixelClass = stixel_mod.StixelClass
    point_dtype = stixel_mod.point_dtype
    stixels: List[object] = []
    width = world.stixel[0].width if world.stixel else 8
    for stx in world.stixel:
        top_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vT, stx.d, stx.label), dtype=point_dtype)
        bot_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vB, stx.d, stx.label), dtype=point_dtype)
        stixels.append(
            Stixel(top_pt, bot_pt, StixelClass.OBJECT, img_size, grid_step=stx.width)
        )
    return stixels, width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a StixelWorld on a given image"
    )
    parser.add_argument("--image", required=True, help="Input JPEG image")
    parser.add_argument("--stx", required=True, help="StixelWorld .stx1 file")
    parser.add_argument("--out", help="Path to save the overlay image")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Path to repository root containing the 'libraries' directory",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    stixel_mod, viz_mod = _prepare_libraries(root)

    img = Image.open(args.image)
    stxl_world = stixel.read(args.stx)
    img_size = {"width": img.width, "height": img.height}
    stixels, width = _to_stixels(stxl_world, img_size, stixel_mod)

    result = viz_mod.draw_stixels_on_image(
        np.array(img), stixels, stixel_width=width, draw_grid=False
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.save(args.out)
        print(f"Saved visualization to {args.out}")
    else:
        result.show()


if __name__ == "__main__":
    main()

