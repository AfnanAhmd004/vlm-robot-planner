"""Instructions, their goals, and a generator of task suites (including ones the robot should refuse)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import COLORS, World


@dataclass
class Task:
    text: str
    kind: str  # place_in | stack | all_in | sort | refuse | clarify
    args: dict
    expect: str = "do"  # do | refuse | clarify

    def satisfied(self, w: World) -> bool:
        """Goal check on the TRUE world."""
        a = self.args
        objs = w.items()
        if self.kind == "place_in":
            return any(o.color == a["color"] and o.kind == a["kind"] and w.in_bin(o.id) and
                       w.objects[w.in_bin(o.id)].color == a["bin"] for o in objs)
        if self.kind == "stack":
            tops = [o for o in objs if o.color == a["c1"] and o.kind == a["k1"]]
            return any(o.on and w.objects[o.on].color == a["c2"] and w.objects[o.on].kind == a["k2"] for o in tops)
        if self.kind == "all_in":
            return all(w.in_bin(o.id) and w.objects[w.in_bin(o.id)].color == a["bin"] for o in objs if o.color == a["color"])
        if self.kind == "sort":
            colors = {b.color for b in w.bins()}
            return all(w.in_bin(o.id) and w.objects[w.in_bin(o.id)].color == o.color for o in objs if o.color in colors)
        return False


def make_tasks(w: World, rng: np.random.Generator, n: int = 3) -> list[Task]:
    items, bins = w.items(), w.bins()
    bin_colors = [b.color for b in bins]
    counts: dict[tuple[str, str], int] = {}
    for o in items:
        counts[(o.color, o.kind)] = counts.get((o.color, o.kind), 0) + 1
    unique = [o for o in items if counts[(o.color, o.kind)] == 1]
    dup = [k for k, v in counts.items() if v > 1]
    cubes = [o for o in unique if o.kind == "cube"]
    out: list[Task] = []
    pool = []
    if unique:
        o = unique[int(rng.integers(len(unique)))]
        b = bin_colors[int(rng.integers(len(bin_colors)))]
        pool.append(Task(f"put the {o.color} {o.kind} in the {b} bin", "place_in", {"color": o.color, "kind": o.kind, "bin": b}))
    if len(unique) >= 2 and cubes:
        base = cubes[int(rng.integers(len(cubes)))]
        top = next(o for o in unique if o.id != base.id)
        pool.append(Task(f"stack the {top.color} {top.kind} on the {base.color} cube", "stack",
                         {"c1": top.color, "k1": top.kind, "c2": base.color, "k2": "cube"}))
    col = items[int(rng.integers(len(items)))].color
    b = bin_colors[int(rng.integers(len(bin_colors)))]
    if sum(o.color == col for o in items) <= 4:
        pool.append(Task(f"put all {col} objects in the {b} bin", "all_in", {"color": col, "bin": b}))
    if sum(o.color in bin_colors for o in items) <= 4 * len(bins):
        pool.append(Task("sort the objects into the bins that match their colour", "sort", {}))
    # things a careful robot should not just do
    missing = [c for c in COLORS if c not in bin_colors]
    balls = [o for o in unique if o.kind == "ball"]
    hard = []
    if unique and missing:
        o = unique[0]
        hard.append(Task(f"put the {o.color} {o.kind} in the {missing[0]} bin", "refuse", {}, "refuse"))
    hard.append(Task(f"pick up the {bin_colors[0]} bin and move it", "refuse", {}, "refuse"))
    if balls and unique:
        top = next((o for o in unique if o.id != balls[0].id), None)
        if top is not None:
            hard.append(Task(f"stack the {top.color} {top.kind} on the {balls[0].color} ball", "refuse", {}, "refuse"))
    if dup:
        c, k = dup[0]
        hard.append(Task(f"put the {c} {k} in the {bin_colors[0]} bin", "clarify", {}, "clarify"))
    rng.shuffle(pool)
    out = pool[: n - 1] + ([hard[int(rng.integers(len(hard)))]] if hard and rng.random() < 0.5 else pool[n - 1:n])
    return out[:n]
