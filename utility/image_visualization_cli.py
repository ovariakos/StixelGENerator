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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

try:  # pyStixel-lib provides the stixel package
    import stixel  # type: ignore
except ImportError as exc:  # pragma: no cover - library not installed
    raise SystemExit(
        "pyStixel-lib is required for visualizing .stx1 files."
        " Install it with 'pip install pyStixel-lib'."
    ) from exc

Stixel_mod = _load_module("libraries.Stixel", ROOT / "libraries" / "Stixel.py")

dummy = types.ModuleType("libraries")
dummy.StixelClass = Stixel_mod.StixelClass
dummy.Stixel = Stixel_mod.Stixel
sys.modules.setdefault("libraries", dummy)
sys.modules.setdefault("libraries.Stixel", Stixel_mod)

Viz_mod = _load_module("libraries.visualization", ROOT / "libraries" / "visualization.py")

Stixel = Stixel_mod.Stixel
StixelClass = Stixel_mod.StixelClass
point_dtype = Stixel_mod.point_dtype
draw_stixels_on_image = Viz_mod.draw_stixels_on_image


def _to_stixels(world: 'stixel.StixelWorld') -> Tuple[List[Stixel], int]:
    """Convert a ``stixel.StixelWorld`` to a list of local ``Stixel`` objects.

    Parameters
    ----------
    world:
        Parsed ``StixelWorld`` protobuf.

    Returns
    -------
    tuple
        The list of converted stixels and the grid width to use when drawing.
    """
    img_size = {
        'width': world.context.calibration.width,
        'height': world.context.calibration.height,
    }
    stixels: List[Stixel] = []
    width = world.stixel[0].width if world.stixel else 8
    for stx in world.stixel:
        top_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vT, stx.d, stx.label),
                          dtype=point_dtype)
        bot_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vB, stx.d, stx.label),
                          dtype=point_dtype)
        stixels.append(
            Stixel(top_pt, bot_pt, StixelClass.OBJECT, img_size, grid_step=stx.width)
        )
    return stixels, width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a StixelWorld on a given image")
    parser.add_argument("--image", required=True, help="Input JPEG image")
    parser.add_argument("--stx", required=True, help="StixelWorld .stx1 file")
    parser.add_argument("--out", help="Path to save the overlay image")
    args = parser.parse_args()

    img = Image.open(args.image)
    stxl_world = stixel.read(args.stx)
    stixels, width = _to_stixels(stxl_world)
    result = draw_stixels_on_image(
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