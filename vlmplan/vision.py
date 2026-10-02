"""Rendering (the camera) and perception (image -> scene graph).

The camera is a top-down RGB view. Perception segments colours, classifies shapes from
their geometry, reads stack levels from the size of the drawn shadow ring, and assigns
objects to bins by containment. It works only from pixels, so its errors are real.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .world import BIN_HALF, COLORS, World

PX = 400  # image size; 1 m table -> 400 px


def _px(v: float) -> int:
    return int(round(v * PX))


def render(world: World, noise: float = 6.0, seed: int = 0) -> np.ndarray:
    img = Image.new("RGB", (PX, PX), (235, 235, 230))
    d = ImageDraw.Draw(img)
    for b in world.bins():
        x, y = b.xy
        d.rectangle([_px(x - BIN_HALF[0]), _px(y - BIN_HALF[1]), _px(x + BIN_HALF[0]), _px(y + BIN_HALF[1])],
                    outline=COLORS[b.color], width=6)
    # draw bottom-up so stacked items are visible on top of their base
    order = sorted(world.items(), key=lambda o: world.height(o.id))
    for o in order:
        x, y = world.objects[o.id].xy  # items in bins are given slot positions by the executor
        r = 0.035
        h = world.height(o.id)
        # stack level is drawn as a dark ring whose thickness grows with height
        d.ellipse([_px(x - r - 0.006 * h), _px(y - r - 0.006 * h), _px(x + r + 0.006 * h), _px(y + r + 0.006 * h)],
                  fill=(60, 60, 60))
        box = [_px(x - r), _px(y - r), _px(x + r), _px(y + r)]
        if o.kind == "cube":
            d.rectangle(box, fill=COLORS[o.color])
        else:
            d.ellipse(box, fill=COLORS[o.color])
    arr = np.asarray(img).astype(float)
    arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class Detection:
    kind: str
    color: str
    xy: tuple[float, float]
    level: int = 1
    in_bin: str | None = None  # colour of the containing bin


def perceive(img: np.ndarray) -> tuple[list[Detection], list[Detection]]:
    """Return (items, bins) detected in a rendered image."""
    f = img.astype(float)
    items, bins = [], []
    for cname, rgb in COLORS.items():
        mask = np.linalg.norm(f - np.array(rgb), axis=2) < 60
        mask = ndimage.binary_opening(mask, iterations=1)
        lab, n = ndimage.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 40:
                continue
            w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
            fill = len(xs) / (w * h)
            cx, cy = xs.mean() / PX, ys.mean() / PX
            if w > 70:  # bin outline: large and hollow
                bins.append(Detection("bin", cname, (cx, cy)))
            else:
                kind = "cube" if fill > 0.88 else "ball"  # square fills its box; a disc fills ~π/4
                items.append(Detection(kind, cname, (cx, cy)))
    dark = np.all(f < 100, axis=2)
    for it in items:  # level from the shadow ring thickness around the object
        x, y = _px(it.xy[0]), _px(it.xy[1])
        r = _px(0.035)
        ring = 0
        for k in range(1, 12):
            xx = min(PX - 1, x + r + k)
            if dark[y, xx]:
                ring = k
            else:
                break
        it.level = max(1, int(round(ring / (0.006 * PX))))
    for it in items:
        for b in bins:
            if abs(it.xy[0] - b.xy[0]) < BIN_HALF[0] and abs(it.xy[1] - b.xy[1]) < BIN_HALF[1]:
                it.in_bin = b.color
    return items, bins
