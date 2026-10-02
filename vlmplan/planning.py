"""Instruction → goal → plan, and the symbolic checker that every plan must pass.

* ``compile_goal`` turns an instruction into a symbolic goal over the current scene,
  or a refusal / clarification request with a reason (no such object, no such bin,
  ambiguous reference, not graspable, cannot stack on a ball, out of reach, bin full).
* ``RulePlanner`` produces a correct plan for a compiled goal (unstacking first if needed).
* ``NoisyPlanner`` imitates the mistakes language models make when planning.
* ``ClaudePlanner`` asks a vision-language model (image + scene + skill API) for a JSON plan.
* ``validate`` simulates any plan on the belief scene: preconditions of every skill, then
  the goal. A planner proposes; the checker decides.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .scene import Scene
from .world import STACK_LIMIT

SKILLS = {"pick": ["obj"], "place_in": ["bin"], "place_on": ["obj"], "place_table": []}


@dataclass
class Goal:
    kind: str  # in_bin | on | all_in | sort | none
    items: list[str] = field(default_factory=list)
    target: str | None = None  # bin id or item id


@dataclass
class Decision:
    status: str  # plan | refuse | clarify
    steps: list[dict] = field(default_factory=list)
    reason: str = ""
    goal: Goal | None = None


# ----------------------------------------------------------------------------- goals
def _one(scene: Scene, color: str, kind: str) -> tuple[str | None, str]:
    ids = scene.find(color, kind)
    if not ids:
        return None, f"I can't see a {color} {kind}"
    if len(ids) > 1:
        return None, f"there are {len(ids)} {color} {kind}s; which one do you mean?"
    return ids[0], ""


def compile_goal(text: str, scene: Scene) -> Decision:
    t = text.lower().strip()
    if re.search(r"\b(pick up|lift|move|carry)\b.*\bbin\b", t) and not re.search(r"\bin the \w+ bin\b", t):
        return Decision("refuse", reason="bins are fixtures; they are not graspable")
    m = re.match(r"put the (\w+) (cube|ball) in the (\w+) bin", t)
    if m:
        c, k, bc = m.groups()
        oid, why = _one(scene, c, k)
        if oid is None:
            return Decision("clarify" if "which one" in why else "refuse", reason=why)
        bid = scene.bin_by_color(bc)
        if bid is None:
            return Decision("refuse", reason=f"there is no {bc} bin")
        return _checked(scene, Goal("in_bin", [oid], bid))
    m = re.match(r"stack the (\w+) (cube|ball) on the (\w+) (cube|ball)", t)
    if m:
        c1, k1, c2, k2 = m.groups()
        if k2 == "ball":
            return Decision("refuse", reason="nothing can be stacked on a ball")
        top, why1 = _one(scene, c1, k1)
        base, why2 = _one(scene, c2, k2)
        if top is None or base is None:
            why = why1 or why2
            return Decision("clarify" if "which one" in why else "refuse", reason=why)
        return _checked(scene, Goal("on", [top], base))
    m = re.match(r"put all (\w+) objects in the (\w+) bin", t)
    if m:
        c, bc = m.groups()
        bid = scene.bin_by_color(bc)
        if bid is None:
            return Decision("refuse", reason=f"there is no {bc} bin")
        return _checked(scene, Goal("all_in", scene.find(c), bid))
    if re.match(r"sort the objects", t):
        items = [i for i, it in scene.items.items() if scene.bin_by_color(it.color)]
        return _checked(scene, Goal("sort", items))
    return Decision("refuse", reason="I don't know how to do that with my skills (pick, place in bin, stack, put down)")


def _checked(scene: Scene, goal: Goal) -> Decision:
    # only objects that still have to move (or be stacked onto) must be reachable
    if goal.kind == "sort":
        to_move = [i for i in goal.items if not (scene.items[i].in_bin and scene.bins[scene.items[i].in_bin] == scene.items[i].color)]
    elif goal.kind == "on":
        to_move = [] if scene.items[goal.items[0]].on == goal.target else goal.items + [goal.target]
    else:
        to_move = [i for i in goal.items if scene.items[i].in_bin != goal.target]
    involved = to_move

    def with_above(i):  # anything stacked on an involved item must be moved too
        return [i] + [x for a in scene.above(i) for x in with_above(a)]

    involved = list(dict.fromkeys(x for i in involved for x in with_above(i)))
    far = [i for i in involved if not scene.reachable(i)]
    if far:
        names = ", ".join(scene.items[i].name for i in far)
        return Decision("refuse", reason=f"out of reach: {names}; please move it closer")
    if goal.kind in ("in_bin", "all_in"):
        new = sum(scene.items[i].in_bin != goal.target for i in goal.items)
        if scene.bin_count(goal.target) + new > scene.capacity:
            return Decision("refuse", reason=f"the {scene.bins[goal.target]} bin does not have room")
    if goal.kind == "sort":
        for b in scene.bins:
            want = [i for i in goal.items if scene.bins[b] == scene.items[i].color]
            if len(want) + sum(1 for i, it in scene.items.items() if it.in_bin == b and i not in want) > scene.capacity:
                return Decision("refuse", reason=f"the {scene.bins[b]} bin does not have room")
    if goal.kind == "on" and scene.height(goal.target) + 1 > STACK_LIMIT:
        return Decision("refuse", reason=f"a stack taller than {STACK_LIMIT} is unsafe")
    return Decision("plan", goal=goal)


def goal_met(scene: Scene, goal: Goal) -> bool:
    if goal.kind == "in_bin":
        return scene.items[goal.items[0]].in_bin == goal.target
    if goal.kind == "on":
        return scene.items[goal.items[0]].on == goal.target
    if goal.kind == "all_in":
        return all(scene.items[i].in_bin == goal.target for i in goal.items)
    if goal.kind == "sort":
        return all(scene.items[i].in_bin and scene.bins[scene.items[i].in_bin] == scene.items[i].color for i in goal.items)
    return False


# ----------------------------------------------------------------------------- checker
def apply(scene: Scene, step: dict) -> str | None:
    """Apply one skill to the belief scene in place. Returns a violation message, or None."""
    sk = step.get("skill")
    if sk not in SKILLS or any(a not in step for a in SKILLS[sk]):
        return f"unknown skill or missing argument: {step}"
    if sk == "pick":
        o = step["obj"]
        if scene.holding is not None:
            return "already holding something"
        if o not in scene.items:
            return f"no object {o!r} in the scene"
        if scene.above(o):
            return f"{scene.items[o].name} has something on top of it"
        if not scene.reachable(o):
            return f"{scene.items[o].name} is out of reach"
        scene.holding = o
        it = scene.items[o]
        it.on, it.in_bin = None, None
        return None
    if scene.holding is None:
        return f"{sk} without holding anything"
    held = scene.items[scene.holding]
    if sk == "place_in":
        b = step["bin"]
        if b not in scene.bins:
            return f"no bin {b!r}"
        if scene.bin_count(b) >= scene.capacity:
            return f"the {scene.bins[b]} bin is full"
        held.in_bin, held.xy = b, scene.bin_xy.get(b, held.xy)
    elif sk == "place_on":
        t = step["obj"]
        if t not in scene.items or t == scene.holding:
            return f"invalid stacking target {t!r}"
        tgt = scene.items[t]
        if tgt.kind != "cube":
            return f"cannot stack on a {tgt.kind}"
        if scene.above(t):
            return f"{tgt.name} is not clear"
        if scene.height(t) + 1 > STACK_LIMIT:
            return "stack too tall"
        if not scene.reachable(t):
            return f"{tgt.name} is out of reach"
        held.on, held.in_bin = t, None
    scene.holding = None
    return None


def validate(steps: list[dict], scene: Scene, goal: Goal | None) -> str | None:
    """None if the plan is executable on the belief scene and achieves the goal."""
    if goal is None:
        return "no valid goal for this instruction" if steps else None
    sim = copy.deepcopy(scene)
    for k, st in enumerate(steps):
        why = apply(sim, st)
        if why:
            return f"step {k + 1} ({st.get('skill')}): {why}"
    if sim.holding is not None:
        return "plan ends while still holding an object"
    if not goal_met(sim, goal):
        return "plan does not achieve the instruction"
    return None


# ----------------------------------------------------------------------------- planners
class RulePlanner:
    """Correct-by-construction planner for compiled goals."""

    name = "rule"

    def plan(self, text: str, scene: Scene, image: np.ndarray | None = None) -> Decision:
        d = compile_goal(text, scene)
        if d.status != "plan":
            return d
        sim = copy.deepcopy(scene)
        steps: list[dict] = []
        g = d.goal

        def do(st):
            assert apply(sim, st) is None, st
            steps.append(st)

        def clear(iid):
            for top in sim.above(iid):
                clear(top)
                do({"skill": "pick", "obj": top})
                do({"skill": "place_table"})

        def move(iid, place):
            clear(iid)
            do({"skill": "pick", "obj": iid})
            do(place)

        if g.kind == "in_bin" and sim.items[g.items[0]].in_bin != g.target:
            move(g.items[0], {"skill": "place_in", "bin": g.target})
        elif g.kind == "on" and sim.items[g.items[0]].on != g.target:
            clear(g.target)
            move(g.items[0], {"skill": "place_on", "obj": g.target})
        elif g.kind in ("all_in", "sort"):
            for iid in sorted(g.items, key=lambda i: -sim.height(i)):
                target = g.target or sim.bin_by_color(sim.items[iid].color)
                if sim.items[iid].in_bin != target:
                    move(iid, {"skill": "place_in", "bin": target})
        d.steps = steps
        return d


class NoisyPlanner:
    """Wraps a correct planner and injects the kinds of mistakes LLM planners make."""

    MISTAKES = ("skip_unstack", "wrong_object", "hallucinated_id", "stack_on_ball", "dropped_pick", "comply_anyway")

    def __init__(self, base: RulePlanner, error_rate: float = 0.3, seed: int = 0):
        self.base, self.error_rate = base, error_rate
        self.rng = np.random.default_rng(seed)
        self.name = f"noisy({error_rate:.0%})"
        self.injected: list[str] = []

    def plan(self, text: str, scene: Scene, image=None) -> Decision:
        d = self.base.plan(text, scene)
        if self.rng.random() >= self.error_rate:
            return d
        if d.status != "plan":  # models are prone to comply with requests they should decline
            self.injected.append("comply_anyway")
            target = next(iter(scene.items), None)
            steps = [{"skill": "pick", "obj": target}, {"skill": "place_in", "bin": next(iter(scene.bins))}] if target else []
            return Decision("plan", steps, "complied", None)
        steps = [dict(s) for s in d.steps]
        kind = self.MISTAKES[int(self.rng.integers(len(self.MISTAKES) - 1))]
        picks = [k for k, s in enumerate(steps) if s["skill"] == "pick"]
        others = [i for i in scene.items if all(i != s.get("obj") for s in steps)]
        if kind == "skip_unstack" and any(s["skill"] == "place_table" for s in steps):
            k = next(k for k, s in enumerate(steps) if s["skill"] == "place_table")
            del steps[k - 1:k + 1]
        elif kind == "wrong_object" and picks and others:
            steps[picks[-1]]["obj"] = others[int(self.rng.integers(len(others)))]
        elif kind == "hallucinated_id" and picks:
            steps[picks[0]]["obj"] = "d99"
        elif kind == "stack_on_ball" and any(it.kind == "ball" for it in scene.items.values()):
            ball = next(i for i, it in scene.items.items() if it.kind == "ball")
            k = max(k for k, s in enumerate(steps) if s["skill"] != "pick") if len(steps) > 1 else 0
            if steps:
                steps[k] = {"skill": "place_on", "obj": ball}
        elif picks:
            del steps[picks[-1]]
            kind = "dropped_pick"
        self.injected.append(kind)
        return Decision("plan", steps, f"(mistake injected: {kind})", d.goal)


PLANNER_PROMPT = """You control a one-arm tabletop robot. Skills (JSON objects):
  {{"skill": "pick", "obj": "<item id>"}}        pick a clear, reachable item
  {{"skill": "place_in", "bin": "<bin id>"}}      drop the held item in a bin
  {{"skill": "place_on", "obj": "<cube id>"}}     stack the held item on a clear cube
  {{"skill": "place_table"}}                      put the held item down on the table
Scene (from the camera; ids are what you must use):
{scene}
Instruction: "{text}"
Reply with JSON only: {{"status": "plan"|"refuse"|"clarify", "steps": [...], "reason": "<short>"}}.
Refuse what the skills cannot do safely; ask to clarify ambiguous references."""


class ClaudePlanner:
    """A vision-language model as the planner (``pip install anthropic``; ANTHROPIC_API_KEY)."""

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model, self.name = model, f"claude:{model}"

    def plan(self, text: str, scene: Scene, image: np.ndarray | None = None) -> Decision:
        from PIL import Image

        content = []
        if image is not None:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": PLANNER_PROMPT.format(scene=scene.describe(), text=text)})
        r = self.client.messages.create(model=self.model, max_tokens=800, messages=[{"role": "user", "content": content}])
        raw = "".join(b.text for b in r.content if b.type == "text")
        try:
            d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            goal = compile_goal(text, scene).goal  # the checker always uses the deterministic goal
            return Decision(d.get("status", "refuse"), list(d.get("steps", [])), str(d.get("reason", "")), goal)
        except ValueError:
            return Decision("refuse", reason="unparseable model output")
