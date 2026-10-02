"""A tabletop world: coloured blocks and balls, bins, stacking, and a single-arm gripper.

The *true* state lives here. The planner never reads it directly: it sees a rendered
image (through perception) or, for ablations, a ground-truth scene description.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

COLORS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 90, 220), "yellow": (235, 200, 30)}
SHAPES = ("cube", "ball")
TABLE = (0.0, 1.0)  # square table, metres
REACH = 0.62  # arm base at the table centre-left; objects beyond this radius are out of reach
BASE = (0.05, 0.5)
STACK_LIMIT = 3
STACK_OFFSET = (0.08, 0.08)  # the camera sees a stacked item drawn diagonally above its base
BIN_HALF = (0.17, 0.09)


@dataclass
class Obj:
    id: str
    kind: str  # cube | ball | bin
    color: str
    xy: tuple[float, float]
    on: str | None = None  # id of the object or bin it rests on / in; None = table
    capacity: int = 0  # bins only

    @property
    def graspable(self) -> bool:
        return self.kind in SHAPES

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind}"


@dataclass
class World:
    objects: dict[str, Obj]
    holding: str | None = None
    log: list[str] = field(default_factory=list)

    def copy(self) -> "World":
        return copy.deepcopy(self)

    # ------------------------------------------------------------- relations
    def above(self, oid: str) -> list[str]:
        return [o.id for o in self.objects.values() if o.on == oid]

    def is_clear(self, oid: str) -> bool:
        return not self.above(oid)

    def height(self, oid: str) -> int:
        h, cur = 1, self.objects[oid]
        while cur.on is not None and self.objects[cur.on].kind != "bin":
            h += 1
            cur = self.objects[cur.on]
        return h

    def in_bin(self, oid: str) -> str | None:
        cur = self.objects[oid]
        while cur.on is not None:
            nxt = self.objects[cur.on]
            if nxt.kind == "bin":
                return nxt.id
            cur = nxt
        return None

    def reachable(self, oid: str) -> bool:
        x, y = self.objects[oid].xy
        return float(np.hypot(x - BASE[0], y - BASE[1])) <= REACH

    def items(self) -> list[Obj]:
        return [o for o in self.objects.values() if o.kind != "bin"]

    def bins(self) -> list[Obj]:
        return [o for o in self.objects.values() if o.kind == "bin"]


def random_world(rng: np.random.Generator, n_items: int = 6, n_bins: int = 2, far_prob: float = 0.1) -> World:
    objs: dict[str, Obj] = {}
    bin_colors = rng.choice(list(COLORS), size=n_bins, replace=False)
    for i, c in enumerate(bin_colors):
        objs[f"bin{i}"] = Obj(f"bin{i}", "bin", str(c), (0.22 + 0.36 * i, 0.18), capacity=4)
    taken: list[tuple[float, float]] = [o.xy for o in objs.values()]
    k = 0
    while k < n_items:
        far = rng.random() < far_prob
        xy = (float(rng.uniform(0.5, 0.95)), float(rng.uniform(0.05, 0.95))) if far else \
             (float(rng.uniform(0.15, 0.6)), float(rng.uniform(0.36, 0.9)))
        if any(np.hypot(xy[0] - a, xy[1] - b) < 0.16 for a, b in taken):
            continue
        taken.append(xy)
        objs[f"o{k}"] = Obj(f"o{k}", str(rng.choice(SHAPES, p=[0.65, 0.35])), str(rng.choice(list(COLORS))), xy)
        k += 1
    w = World(objs)
    # sometimes start with a small stack
    cubes = [o for o in w.items() if o.kind == "cube"]
    if len(cubes) >= 2 and rng.random() < 0.5:
        top, base = cubes[0], cubes[1]
        top.on, top.xy = base.id, (base.xy[0] + STACK_OFFSET[0], base.xy[1] + STACK_OFFSET[1])
    return w
