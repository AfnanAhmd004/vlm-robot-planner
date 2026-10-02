"""The robot's belief: a scene graph built from perception (or, for ablations, from ground truth)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vision import perceive
from .world import BASE, REACH, STACK_LIMIT, STACK_OFFSET, World


@dataclass
class Item:
    id: str
    kind: str  # cube | ball
    color: str
    xy: tuple[float, float]
    on: str | None = None  # item id below it
    in_bin: str | None = None  # bin id

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind}"


@dataclass
class Scene:
    items: dict[str, Item]
    bins: dict[str, str]  # bin id -> colour
    bin_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    capacity: int = 4
    holding: str | None = None

    def above(self, iid: str) -> list[str]:
        return [i.id for i in self.items.values() if i.on == iid]

    def height(self, iid: str) -> int:
        h, cur = 1, self.items[iid]
        while cur.on is not None:
            h, cur = h + 1, self.items[cur.on]
        return h

    def reachable(self, iid: str) -> bool:
        x, y = self.items[iid].xy
        return float(np.hypot(x - BASE[0], y - BASE[1])) <= REACH

    def bin_count(self, bid: str) -> int:
        return sum(i.in_bin == bid for i in self.items.values())

    def find(self, color: str | None = None, kind: str | None = None) -> list[str]:
        return [i.id for i in self.items.values() if (color is None or i.color == color) and (kind is None or i.kind == kind)]

    def bin_by_color(self, color: str) -> str | None:
        return next((b for b, c in self.bins.items() if c == color), None)

    def describe(self) -> str:
        lines = [f"{i.id}: {i.name} at ({i.xy[0]:.2f}, {i.xy[1]:.2f})"
                 + (f", on {i.on}" if i.on else "") + (f", in {i.in_bin}" if i.in_bin else "")
                 + ("" if self.reachable(i.id) else ", OUT OF REACH") for i in self.items.values()]
        lines += [f"{b}: {c} bin ({self.bin_count(b)}/{self.capacity} used)" for b, c in self.bins.items()]
        return "\n".join(lines)


def scene_from_image(img: np.ndarray) -> Scene:
    dets, bins = perceive(img)
    bin_ids = {f"bin_{b.color}": b.color for b in bins}
    bin_xy = {f"bin_{b.color}": b.xy for b in bins}
    items = {}
    for k, d in enumerate(sorted(dets, key=lambda d: (d.xy[1], d.xy[0]))):
        items[f"d{k}"] = Item(f"d{k}", d.kind, d.color, d.xy, in_bin=f"bin_{d.in_bin}" if d.in_bin else None)
    levels = {f"d{k}": d.level for k, d in enumerate(sorted(dets, key=lambda d: (d.xy[1], d.xy[0])))}
    for iid, lvl in levels.items():  # a level-L item sits on the level-(L-1) item at the stacking offset
        if lvl > 1 and items[iid].in_bin is None:
            x, y = items[iid].xy
            cand = [j for j, l2 in levels.items() if l2 == lvl - 1 and j != iid]
            if cand:
                j = min(cand, key=lambda j: np.hypot(items[j].xy[0] + STACK_OFFSET[0] - x,
                                                     items[j].xy[1] + STACK_OFFSET[1] - y))
                items[iid].on = j
    return Scene(items, bin_ids, bin_xy)


def scene_from_world(w: World) -> Scene:
    """Ground-truth belief (perfect perception), for ablations."""
    items = {}
    for o in w.items():
        on = o.on if o.on is not None and w.objects[o.on].kind != "bin" else None
        items[o.id] = Item(o.id, o.kind, o.color, o.xy, on=on, in_bin=w.in_bin(o.id))
    return Scene(items, {b.id: b.color for b in w.bins()}, {b.id: b.xy for b in w.bins()})


__all__ = ["Item", "Scene", "scene_from_image", "scene_from_world", "STACK_LIMIT"]
