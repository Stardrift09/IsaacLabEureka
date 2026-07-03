#!/usr/bin/env python
"""Tile image frames into a single grid (reading order), optionally numbered.

No Isaac Sim needed -- just PIL/numpy/imageio. Example:

    python scripts/make_grid.py \
        --frames a.png b.png c.png d.png \
        --rows 2 --cols 2 --out grid_2x2.png

By default each cell is labeled 1..N in its top-left corner (white text, black
outline). Pass --no-label to omit, or --labels "..." to set custom labels.
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def label_cell(im: Image.Image, text: str) -> None:
    w, h = im.size
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(h * 0.11))
    except Exception:
        font = ImageFont.load_default()
    dr = ImageDraw.Draw(im)
    x, y = int(w * 0.04), int(h * 0.03)
    for dx in (-2, -1, 0, 1, 2):           # black outline for contrast
        for dy in (-2, -1, 0, 1, 2):
            dr.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    dr.text((x, y), text, font=font, fill=(255, 255, 255))


def main() -> None:
    p = argparse.ArgumentParser(description="Tile frames into a grid.")
    p.add_argument("--frames", nargs="+", required=True, help="Image paths, in reading order.")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument("--no-label", action="store_true", help="Do not number the cells.")
    p.add_argument("--labels", nargs="+", default=None, help="Custom per-cell labels.")
    args = p.parse_args()

    n = args.rows * args.cols
    if len(args.frames) != n:
        raise SystemExit(f"got {len(args.frames)} frames but rows*cols = {n}")

    imgs = [Image.open(f).convert("RGB") for f in args.frames]
    w, h = imgs[0].size
    imgs = [im if im.size == (w, h) else im.resize((w, h)) for im in imgs]

    if not args.no_label:
        labels = args.labels if args.labels is not None else [str(i) for i in range(1, n + 1)]
        for im, lab in zip(imgs, labels):
            label_cell(im, lab)

    arr = [np.array(im) for im in imgs]
    rows = [np.concatenate(arr[r * args.cols:(r + 1) * args.cols], axis=1) for r in range(args.rows)]
    grid = np.concatenate(rows, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.imwrite(args.out, grid)
    print(f"wrote {args.out}  {grid.shape}")


if __name__ == "__main__":
    main()
