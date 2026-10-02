import numpy as np
import pytest

from vlmplan import (NoisyPlanner, RulePlanner, Task, compile_goal, make_tasks, perceive, random_world, render,
                     run_task, scene_from_image, scene_from_world, validate)
from vlmplan.world import Obj, World


def table(*objs, bins=(("bin0", "red", (0.22, 0.18)), ("bin1", "blue", (0.58, 0.18)))):
    d = {o.id: o for o in objs}
    for bid, c, xy in bins:
        d[bid] = Obj(bid, "bin", c, xy, capacity=4)
    return World(d)


def test_perception_matches_ground_truth_on_most_scenes():
    ok = 0
    for s in range(40):
        w = random_world(np.random.default_rng(s))
        items, bins = perceive(render(w, seed=s))
        ok += sorted((d.color, d.kind) for d in items) == sorted((o.color, o.kind) for o in w.items())
        assert sorted(b.color for b in bins) == sorted(b.color for b in w.bins())
    assert ok >= 36


def test_perception_recovers_stacks():
    w = table(Obj("a", "cube", "green", (0.3, 0.6)), Obj("b", "cube", "red", (0.38, 0.68), on="a"))
    sc = scene_from_image(render(w))
    top = next(i for i in sc.items.values() if i.color == "red")
    base = next(i for i in sc.items.values() if i.color == "green")
    assert top.on == base.id


def test_compiler_refuses_and_asks():
    w = table(Obj("a", "cube", "green", (0.3, 0.6)), Obj("b", "ball", "red", (0.45, 0.6)),
              Obj("c", "ball", "red", (0.3, 0.8)), Obj("far", "cube", "yellow", (0.9, 0.9)))
    sc = scene_from_world(w)
    assert compile_goal("put the green cube in the yellow bin", sc).status == "refuse"   # no such bin
    assert compile_goal("put the red ball in the blue bin", sc).status == "clarify"      # two red balls
    assert compile_goal("stack the green cube on the red ball", sc).status == "refuse"   # can't stack on a ball
    assert compile_goal("pick up the red bin and move it", sc).status == "refuse"        # fixture
    assert "out of reach" in compile_goal("put the yellow cube in the red bin", sc).reason
    assert compile_goal("make me a sandwich", sc).status == "refuse"


def test_rule_plan_unstacks_first_and_validates():
    w = table(Obj("a", "cube", "green", (0.3, 0.6)), Obj("b", "cube", "red", (0.38, 0.68), on="a"))
    sc = scene_from_world(w)
    d = RulePlanner().plan("put the green cube in the blue bin", sc)
    assert [s["skill"] for s in d.steps] == ["pick", "place_table", "pick", "place_in"]
    assert d.steps[0]["obj"] == "b" and validate(d.steps, sc, d.goal) is None


@pytest.mark.parametrize("bad,why", [
    ([{"skill": "pick", "obj": "a"}, {"skill": "place_in", "bin": "bin1"}], "has something on top"),
    ([{"skill": "place_in", "bin": "bin1"}], "without holding"),
    ([{"skill": "pick", "obj": "zz"}], "no object"),
    ([{"skill": "pick", "obj": "b"}, {"skill": "place_on", "obj": "c"}], "cannot stack on a ball"),
    ([{"skill": "pick", "obj": "b"}, {"skill": "place_in", "bin": "bin1"}], "does not achieve"),
    ([{"skill": "fly"}], "unknown skill"),
])
def test_checker_rejects_bad_plans(bad, why):
    w = table(Obj("a", "cube", "green", (0.3, 0.6)), Obj("b", "cube", "red", (0.38, 0.68), on="a"),
              Obj("c", "ball", "yellow", (0.2, 0.8)))
    sc = scene_from_world(w)
    goal = compile_goal("put the green cube in the blue bin", sc).goal
    assert why in validate(bad, sc, goal)


def test_closed_loop_recovers_from_slips():
    rates = {}
    for closed in (False, True):
        ok = []
        for s in range(40):
            w = table(Obj("b", "ball", "red", (0.3, 0.6)))
            ok.append(run_task(Task("put the red ball in the blue bin", "place_in",
                                    {"color": "red", "kind": "ball", "bin": "blue"}), w, RulePlanner(),
                               closed_loop=closed, seed=s).success)
        rates[closed] = np.mean(ok)
    assert rates[True] >= 0.95 and rates[True] > rates[False]


def test_checker_makes_an_unreliable_planner_safe():
    suite = []
    for s in range(25):
        rng = np.random.default_rng(s)
        suite += [(s, t) for t in make_tasks(random_world(rng), rng)]
    unsafe = {}
    for use in (False, True):
        planner = NoisyPlanner(RulePlanner(), 0.6, seed=0)
        unsafe[use] = sum(len(run_task(t, random_world(np.random.default_rng(s)), planner, use_validator=use,
                                       seed=i).unsafe) for i, (s, t) in enumerate(suite))
    assert unsafe[False] > 10 and unsafe[True] == 0


def test_refusal_after_acting_is_not_credited():
    w = table(Obj("a", "cube", "green", (0.3, 0.6)))

    class Complier:
        calls = 0

        def plan(self, text, scene, image=None):
            from vlmplan.planning import Decision
            self.calls += 1
            if self.calls == 1:
                return Decision("plan", [{"skill": "pick", "obj": next(iter(scene.items))},
                                         {"skill": "place_in", "bin": next(iter(scene.bins))}])
            return Decision("refuse", reason="changed my mind")

    out = run_task(Task("pick up the red bin and move it", "refuse", {}, "refuse"), w, Complier(),
                   use_validator=False, seed=0)
    assert not out.success
