"""Evaluate the planning stack on a generated task suite.

    python examples/evaluate.py             # ~7 min, writes docs/scene.png
    python examples/evaluate.py --claude <model-id>   # use Claude as the planner (ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np

from vlmplan import (ClaudePlanner, NoisyPlanner, RulePlanner, make_tasks, perceive, random_world, render, run_task,
                     scene_from_image)

N_SCENES = 150


def suite():
    out = []
    for s in range(N_SCENES):
        rng = np.random.default_rng(s)
        w = random_world(rng)
        out += [(s, t) for t in make_tasks(w, rng)]
    return out


def evaluate(name, make_planner, **kw):
    planner = make_planner()
    rows, unsafe = [], collections.Counter()
    for i, (s, task) in enumerate(SUITE):
        out = run_task(task, random_world(np.random.default_rng(s)), planner, seed=i, **kw)
        rows.append((task, out))
        unsafe.update(out.unsafe)
    do = [o for t, o in rows if t.expect == "do"]
    other = [o for t, o in rows if t.expect != "do"]
    print(f"{name:<44}{np.mean([o.success for o in do]):>12.1%}{np.mean([o.success for o in other]):>12.1%}"
          f"{sum(unsafe.values()):>8}{np.mean([o.steps for o in do]):>7.1f}{sum(o.rejected_plans for _, o in rows):>10}")
    return unsafe, planner


def perception_report():
    exact, n, merged = 0, 0, 0
    for s in range(300):
        w = random_world(np.random.default_rng(10_000 + s))
        items, _ = perceive(render(w, seed=s))
        truth = sorted((o.color, o.kind) for o in w.items())
        det = sorted((d.color, d.kind) for d in items)
        exact += truth == det
        merged += len(det) < len(truth)
        n += 1
    print(f"\nperception: scene graph exactly right in {exact}/{n} scenes ({exact / n:.1%}); "
          f"{merged} scenes with touching same-colour objects merged into one")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = random_world(np.random.default_rng(3))
    img = render(w, seed=0)
    scene = scene_from_image(img)
    plan = RulePlanner().plan("sort the objects into the bins that match their colour", scene)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(img)
    for it in scene.items.values():
        ax.annotate(it.id + (f"↑{it.on}" if it.on else ""), (it.xy[0] * 400, it.xy[1] * 400 - 18),
                    ha="center", fontsize=8, color="k", weight="bold")
    steps = " → ".join(f"{s['skill']}({s.get('obj') or s.get('bin') or ''})" for s in plan.steps)
    ax.set_title("camera view + perceived ids\n\"sort the objects into the bins that match their colour\"", fontsize=9)
    import textwrap
    ax.set_xlabel("\n".join(textwrap.wrap("plan: " + steps, 80)), fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    os.makedirs("docs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("docs/scene.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude")
    args = ap.parse_args()
    SUITE = suite()
    kinds = collections.Counter(t.expect for _, t in SUITE)
    print(f"{len(SUITE)} tasks over {N_SCENES} scenes: {kinds['do']} to do, {kinds['refuse']} to refuse, "
          f"{kinds['clarify']} needing clarification\n")
    print(f"{'configuration':<44}{'tasks done':>12}{'refuse/ask':>12}{'unsafe':>8}{'steps':>7}{'rejected':>10}")
    evaluate("rule planner, open loop, perfect state", RulePlanner, closed_loop=False, perception="truth")
    evaluate("rule planner, open loop, camera", RulePlanner, closed_loop=False)
    evaluate("rule planner, closed loop, camera", RulePlanner)
    for p in (0.3, 0.6):
        u, _ = evaluate(f"{p:.0%}-error planner, closed loop, NO checker", lambda: NoisyPlanner(RulePlanner(), p, 1),
                        use_validator=False)
        print(f"{'':<6}unsafe actions: {dict(u)}")
        _, planner = evaluate(f"{p:.0%}-error planner, closed loop, checker", lambda: NoisyPlanner(RulePlanner(), p, 1))
    if args.claude:
        evaluate(f"Claude ({args.claude}), closed loop, checker", lambda: ClaudePlanner(args.claude))
    perception_report()
    figure()
    print("saved docs/scene.png")
