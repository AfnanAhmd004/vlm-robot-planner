"""Execution on the (simulated) real world, and the perceive → plan → check → act → re-perceive loop."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planning import Decision, compile_goal, validate
from .scene import Scene, scene_from_image, scene_from_world
from .tasks import Task
from .vision import render
from .world import BASE, BIN_HALF, REACH, STACK_OFFSET, World

GRASP_OK = {"cube": 0.95, "ball": 0.80}
STACK_OK = 0.92


@dataclass
class Executor:
    """Runs skills on the true world. Skills act on *locations* from the belief, like a real arm."""

    world: World
    rng: np.random.Generator
    unsafe: list[str] = field(default_factory=list)

    def _at(self, xy) -> str | None:
        cands = [o for o in self.world.items() if np.hypot(o.xy[0] - xy[0], o.xy[1] - xy[1]) < 0.04]
        return max(cands, key=lambda o: self.world.height(o.id)).id if cands else None  # the top of a stack

    def _drop_near(self, oid: str, xy) -> None:
        o = self.world.objects[oid]
        o.on = None
        for _ in range(50):
            p = (float(np.clip(xy[0] + self.rng.normal(0, 0.12), 0.08, 0.92)),
                 float(np.clip(xy[1] + self.rng.normal(0, 0.12), 0.36, 0.95)))
            if np.hypot(p[0] - BASE[0], p[1] - BASE[1]) <= REACH - 0.05 and all(np.hypot(p[0] - q.xy[0], p[1] - q.xy[1]) > 0.15 for q in self.world.items() if q.id != oid):
                o.xy = p
                return
        o.xy = (0.3, 0.6)

    def run(self, step: dict, scene: Scene) -> bool:
        w, sk = self.world, step.get("skill")
        if sk == "pick":
            it = scene.items.get(step.get("obj"))
            if it is None or w.holding is not None:
                return False
            if np.hypot(it.xy[0] - BASE[0], it.xy[1] - BASE[1]) > REACH:
                self.unsafe.append("reached beyond workspace")
                return False
            oid = self._at(it.xy)
            if oid is None:
                return False
            above = w.above(oid)
            if above:  # pulling an item out from under a stack topples what is on it
                self.unsafe.append("toppled a stack")
                for a in above:
                    self._drop_near(a, w.objects[a].xy)
            if self.rng.random() > GRASP_OK[w.objects[oid].kind]:
                return False  # slipped
            w.holding = oid
            w.objects[oid].on = None
            return True
        if w.holding is None:
            return False
        held = w.objects[w.holding]
        if sk == "place_in":
            b = step.get("bin")
            bins = {f"bin_{x.color}": x for x in w.bins()} | {x.id: x for x in w.bins()}
            if b not in bins:
                return False
            bo = bins[b]
            inside = [o for o in w.items() if w.in_bin(o.id) == bo.id]
            if len(inside) >= bo.capacity:
                self.unsafe.append("overfilled a bin")
                self._drop_near(held.id, bo.xy)
            else:
                held.on, held.xy = bo.id, (bo.xy[0] - BIN_HALF[0] + 0.05 + 0.08 * len(inside), bo.xy[1])
        elif sk == "place_on":
            t = scene.items.get(step.get("obj"))
            tid = self._at(t.xy) if t else None
            if tid is None or tid == held.id:
                self._drop_near(held.id, BASE)
            elif w.objects[tid].kind == "ball":
                self.unsafe.append("stacked on a ball")
                self._drop_near(held.id, w.objects[tid].xy)
            elif self.rng.random() > STACK_OK:
                self._drop_near(held.id, w.objects[tid].xy)
            else:
                held.on = tid
                held.xy = (w.objects[tid].xy[0] + STACK_OFFSET[0], w.objects[tid].xy[1] + STACK_OFFSET[1])
        elif sk == "place_table":
            self._drop_near(held.id, (0.35, 0.65))
        else:
            return False
        w.holding = None
        return True


@dataclass
class Outcome:
    status: str  # done | refuse | clarify | gave_up
    success: bool
    steps: int
    replans: int
    rejected_plans: int
    unsafe: list[str]
    reason: str = ""


def run_task(task: Task, world: World, planner, closed_loop: bool = True, use_validator: bool = True,
             perception: str = "camera", max_steps: int = 30, max_replans: int = 6, seed: int = 0) -> Outcome:
    """Perceive → plan → (check) → act; in closed loop, re-perceive after failures and replan.

    Plans refer to object ids of the frame they were made on, so each plan is executed
    against that frame's positions; a fresh frame is only used to plan again.
    """
    from .planning import goal_met

    rng = np.random.default_rng(seed)
    ex = Executor(world, rng)
    frame = [0]

    def observe() -> tuple[Scene, np.ndarray | None]:
        frame[0] += 1
        if perception == "camera":
            img = render(world, seed=seed * 1000 + frame[0])
            return scene_from_image(img), img
        return scene_from_world(world), None

    steps = replans = rejected = 0
    while True:
        scene, img = observe()
        decision = planner.plan(task.text, scene, img)
        if use_validator and decision.status == "plan":
            # the goal always comes from the deterministic compiler, never from the planner
            gate = compile_goal(task.text, scene)
            why = validate(decision.steps, scene, gate.goal) if gate.status == "plan" else f"{gate.status}: {gate.reason}"
            if why:
                rejected += 1
                if gate.status != "plan":
                    decision = gate
                elif rejected <= max_replans:
                    continue
                else:
                    return Outcome("gave_up", False, steps, replans, rejected, ex.unsafe, why)
        if decision.status != "plan":
            # declining is right only if nothing was done first
            right = task.expect != "do" or _unreachable_target(task, world)
            ok = steps == 0 and right and (decision.status == task.expect or task.expect == "do")
            return Outcome(decision.status, ok, steps, replans, rejected, ex.unsafe, decision.reason)
        failed = False
        for st in decision.steps:
            steps += 1
            if not ex.run(st, scene):
                failed = True
                if closed_loop:
                    break
            if steps >= max_steps:
                break
        if world.holding is not None:  # never leave something in the gripper
            ex.run({"skill": "place_table"}, scene)
        if not closed_loop or steps >= max_steps:
            break
        check, _ = observe()
        gate = compile_goal(task.text, check)
        if not failed and gate.status == "plan" and gate.goal is not None and goal_met(check, gate.goal):
            break
        if not failed and gate.status == "plan" and not decision.steps:
            break
        replans += 1
        if replans > max_replans:
            break
    success = task.expect == "do" and task.satisfied(world)
    return Outcome("done", success, steps, replans, rejected, ex.unsafe)


def _unreachable_target(task: Task, w: World) -> bool:
    """Refusing is the right answer when, in the TRUE world, something that must move is out of reach."""
    gt = compile_goal(task.text, scene_from_world(w))
    return gt.status == "refuse" and gt.reason.startswith("out of reach")
