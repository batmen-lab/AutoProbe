"""Offline verification of the unified Stage-4 improvement loop.

Runs the REAL orchestrator code (pipeline.stages) against a throwaway
workspace, with `agent_call` replaced by a scripted stub. No `claude`
subprocess, no GPU, no LLM spend. `train.py` is a 20-line fake that writes a
probe_result JSON, so `run_training_with_autofix`, the output normalization,
the snapshot-git tagging and the keep/revert logic all execute for real.

What it pins down:
  1. auto-research setup mirrors stage 3 (one training run == round 1, both
     snapshot tags, one iteration row) and never calls the commentor.
  2. auto-research improvement rounds ARE auto-probe's fix-plan rounds
     (PROMPT_NINE -> PROMPT_TEN -> PROMPT_ELEVEN), honour the user hint, and
     carry the auto-research addendum.
  3. AUTO-PROBE SEES NOTHING BUT THE SHARED PROMPT — exactly the constant
     plus its round index / hint / plan JSON, no addendum and no mode
     branching. The one conditional AUTO-RESEARCH MODE line inside those
     shared constants is inert without the addendum. This is the regression
     guard that matters most.
  4. the manual fix-plan path (generate + user-selected apply) works in
     auto-research mode, where it used to raise.
  5. the stale `iterate_once` engine is still disabled.

Run:  venv/bin/python tests/test_unified_stage4.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hard_prompt.agent_fix_plan_apply import PROMPT_ELEVEN
from hard_prompt.agent_fix_plan_gen import PROMPT_NINE, PROMPT_TEN
from hard_prompt.agent_improve_commentor import PROMPT_SIX
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_FIX_PLAN_ADDENDUM as ADDENDUM,
)
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION as SETUP_PROMPT,
)

# The conditional mode line the shared prompts carry, and the name of the
# block it is conditional on.
MODE_LINE_KEY = "AUTO-RESEARCH MODE (conditional"
ADDENDUM_HEADER = "AUTO-RESEARCH MODE ADDENDUM"
# The block's own banner. Distinct from ADDENDUM_HEADER because the
# conditional line quotes that name, so the name alone cannot tell you whether
# the block itself is present.
ADDENDUM_BLOCK_MARK = "AUTO-RESEARCH MODE ADDENDUM — READ THIS LAST"

from pipeline import snapshot_git as snap
from pipeline import stages as st
from pipeline import state as state_mod

FAKE_TRAIN = '''\
"""Fake trainer: writes a probe_result JSON, no model, no data."""
import json
from pathlib import Path

VALUE = 0.5000  # the "hyperparameter" a fix plan edits
TRAIN_LOSS = 0.4000  # ANCHOR: original train metric

OUT = Path(".agent_probe/metric")
OUT.mkdir(parents=True, exist_ok=True)
n = 1 + max(
    [int(p.stem.rsplit("_", 1)[-1]) for p in OUT.glob("probe_result_*.json")] or [0]
)
values = [{"epoch": i, "value": VALUE - (0.01 if i == 0 else 0.0)} for i in range(6)]
OUT.joinpath(f"probe_result_{n}.json").write_text(
    json.dumps(
        {
            "metric_name": "val_auprc",
            "direction": "higher_is_better",
            "values": values,
            "tail_mean": VALUE,
            "final_value": VALUE,
            "conclusion": "fake run",
            "original_train_metric_0": {
                "name": "train_loss",
                "value": TRAIN_LOSS,
                "direction": "lower_is_better",
            },
            "original_train_metric_1": {
                "name": "val_auprc",
                "value": VALUE,
                "direction": "higher_is_better",
            },
        },
        indent=2,
    )
)
print(f"wrote probe_result_{n}.json")
'''

FAKE_PROBER = '"""Fake prober written by the stubbed setup agent."""\n'


# ── scripted agent stub ─────────────────────────────────────────────────────
class AgentStub:
    """Stands in for `pipeline.stages.agent_call`. Records every prompt and
    performs the minimum filesystem effect each real agent would have."""

    def __init__(self, *, improve_by: float = 0.02) -> None:
        self.prompts: list[str] = []
        self.improve_by = improve_by

    # Dispatch on a phrase unique to each prompt. NOT on a prefix: the
    # auto-research setup prompt and PROMPT_ELEVEN now open with the same
    # shared "PROTECTED CODE - user_analyze()" paragraph.
    MARKERS = {
        "setup": "There is no probe design document for you to read",
        "generate": "propose THREE distinct, concrete fix plans",
        "confidence": "fill in their `confidence` scores",
        "apply": "The user has selected ONE specific fix plan",
    }

    @classmethod
    def kind(cls, prompt: str) -> str:
        hits = [k for k, m in cls.MARKERS.items() if m in prompt]
        if len(hits) != 1:
            raise AssertionError(f"ambiguous prompt ({hits}): {prompt[:120]!r}")
        return hits[0]

    def __call__(self, prompt, *, cwd, log_path=None, model=None, label=None):
        self.prompts.append(prompt)
        cwd = Path(cwd)
        if PROMPT_SIX.strip()[:80] in prompt:
            raise AssertionError("the improve-commentor must never be called")
        kind = self.kind(prompt)
        if kind == "setup":
            (cwd / "prober.py").write_text(FAKE_PROBER)
        elif kind == "generate":
            self._write_fix_plans(cwd)
        elif kind == "confidence":
            self._score_fix_plans(cwd)
        elif kind == "apply":
            self._apply_plan(cwd)

    # the round index the agents are working on == next probe_result index
    @staticmethod
    def _round(cwd: Path) -> int:
        metric = cwd / ".agent_probe" / "metric"
        idx = [
            int(p.stem.rsplit("_", 1)[-1])
            for p in metric.glob("probe_result_*.json")
        ]
        return (max(idx) + 1) if idx else 1

    def _plans_path(self, cwd: Path) -> Path:
        d = cwd / ".agent_probe" / "fix_plans"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"fix_plans_{self._round(cwd)}.json"

    def _write_fix_plans(self, cwd: Path) -> None:
        plans = [
            {
                "title": f"plan {i}",
                "content": f"raise VALUE in train.py (variant {i})",
                "target_files": ["train.py"],
                "confidence": 0.0,
            }
            for i in (1, 2, 3)
        ]
        self._plans_path(cwd).write_text(json.dumps({"fix_plans": plans}, indent=2))

    def _score_fix_plans(self, cwd: Path) -> None:
        p = self._plans_path(cwd)
        doc = json.loads(p.read_text())
        for i, plan in enumerate(doc["fix_plans"]):
            plan["confidence"] = [0.4, 0.9, 0.6][i]  # plan #2 wins
        p.write_text(json.dumps(doc, indent=2))

    def _apply_plan(self, cwd: Path) -> None:
        train = cwd / "train.py"
        src = train.read_text()
        cur = float(re.search(r"^VALUE = ([\d.]+)", src, re.M).group(1))
        src = re.sub(
            r"^VALUE = [\d.]+",
            f"VALUE = {cur + self.improve_by:.4f}",
            src,
            count=1,
            flags=re.M,
        )
        train.write_text(src)
        n = self._round(cwd)
        (cwd / ".agent_probe" / f"change_log_{n}.txt").write_text(
            f"bumped VALUE to {cur + self.improve_by:.4f}\n"
        )


# ── harness ─────────────────────────────────────────────────────────────────
class Harness:
    """A temp workspace + a temp response dir, with agent_call stubbed."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="autoprobe-test-"))
        self.ws = self.tmp / "ws"
        self.ws.mkdir()
        (self.ws / "train.py").write_text(FAKE_TRAIN)
        self.runs = self.tmp / "response"
        self.runs.mkdir()
        self._orig_run_base = state_mod.RUN_BASE
        self._orig_agent = st.agent_call
        state_mod.RUN_BASE = self.runs
        self.stub = AgentStub()
        st.agent_call = self.stub
        self.state = state_mod.new_run(self.ws)

    def close(self) -> None:
        state_mod.RUN_BASE = self._orig_run_base
        st.agent_call = self._orig_agent
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers
    def metric_indices(self) -> list[int]:
        d = self.ws / ".agent_probe" / "metric"
        return sorted(
            int(p.stem.rsplit("_", 1)[-1]) for p in d.glob("probe_result_*.json")
        )

    def seed_auto_probe_stage4(self) -> None:
        """Put the run where auto-probe's stage 4 starts, without stage 1-3."""
        (self.ws / "prober.py").write_text(FAKE_PROBER)
        snap.commit_train(self.ws, "stage 3 implement (pre-iter)", tag="pre-iter")
        st.run_training_with_autofix(self.state, expected_index=1)
        snap.commit_train(self.ws, "round 1 post", tag="round-1-post")
        snapshot = st._read_latest_metric(self.ws)
        self.state.record_iteration(
            st._iteration_record_from_snapshot(snapshot, note="stage 3 first run")
        )
        self.state.set_phase(state_mod.Stage.FOUR, "ready")


# ── assertions ──────────────────────────────────────────────────────────────
FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def eq(got, want, label: str) -> None:
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}")
        FAILURES.append(label)


# ── 1. auto-research setup mirrors stage 3 ─────────────────────────────────
def test_auto_research_setup() -> None:
    print("── auto-research setup mirrors stage 3 ──────────────────")
    h = Harness()
    try:
        st.auto_research_setup(h.state)
        rec = h.state.record
        eq(rec.stage, 4, "setup parks the run at stage 4")
        eq(h.metric_indices(), [1], "exactly one training run -> probe_result_1")
        eq(len(rec.iterations), 1, "one iteration row (the setup run)")
        eq(rec.iterations[0]["index"], 1, "that row is round 1")
        eq(rec.iterations[0]["tail_mean"], 0.5, "row carries the decision value")
        check(snap.tag_exists(h.ws, "pre-iter"), "pre-iter tag written")
        check(snap.tag_exists(h.ws, "round-1-post"), "round-1-post tag written")
        check((h.ws / "prober.py").exists(), "prober.py created")
        eq(len(h.stub.prompts), 1, "exactly one agent call (setup only, no commentor)")
        check(
            not any(PROMPT_SIX[:60] in p for p in h.stub.prompts),
            "commentor prompt never sent",
        )
        eq(rec.auto_research_best_value, 0.5, "running best seeded from the setup run")
        eq(
            rec.auto_research_best_direction,
            "higher_is_better",
            "direction seeded from the probe result",
        )
    finally:
        h.close()


# ── 2. auto-research rounds are auto-probe fix-plan rounds ────────────────
def test_auto_research_batch() -> None:
    print("── auto-research batch == auto-probe fix-plan rounds ────")
    h = Harness()
    try:
        st.auto_research_setup(h.state)
        h.stub.prompts.clear()
        out = st.auto_research_iterate_batch(h.state, 2, "focus on the learning rate")
        rec = h.state.record

        eq(len(h.stub.prompts), 6, "2 rounds x 3 agent calls (gen/score/apply)")
        kinds = [AgentStub.kind(p) for p in h.stub.prompts]
        eq(
            kinds,
            ["generate", "confidence", "apply"] * 2,
            "fix-plan flow, in order, twice",
        )
        check(
            all(ADDENDUM in p for p in h.stub.prompts),
            "every shared prompt carries the auto-research addendum",
        )
        check(
            all(
                "User hint:\nfocus on the learning rate" in p
                for p, k in zip(h.stub.prompts, kinds)
                if k == "generate"
            ),
            "user hint reaches the fix-plan generator each round",
        )
        eq(h.metric_indices(), [1, 2, 3], "rounds 2 and 3 produced probe results")
        eq(len(rec.iterations), 3, "three iteration rows")
        eq(
            [r["index"] for r in rec.iterations],
            [1, 2, 3],
            "round indices are contiguous",
        )
        check(
            snap.tag_exists(h.ws, "round-2-post")
            and snap.tag_exists(h.ws, "round-3-post"),
            "per-round snapshot tags written by the orchestrator",
        )
        eq(
            [r["fix_plan_chosen_index"] for r in rec.iterations[1:]],
            [2, 2],
            "auto-pilot picked the highest-confidence plan (#2)",
        )
        eq(
            [r["note"] for r in rec.iterations[1:]],
            ["auto-pick fix plan #2: plan 2"] * 2,
            "improving rounds kept (no revert suffix on the note)",
        )
        eq(rec.iterations[-1]["tail_mean"], 0.54, "metric advanced 0.50 -> 0.54")
        eq(
            [r["best_value"] for r in rec.iterations],
            [0.5, 0.52, 0.54],
            "monotonic best series stamped on every row",
        )
        check(
            all(
                r["threshold"] is None
                and r["acceptable_threshold"] is None
                and r["acceptable_met"] is None
                for r in rec.iterations
            ),
            "no threshold fields leak into auto-research rows",
        )
        eq(out["completed"], 2, "batch reports both rounds completed")
        eq(rec.auto_research_runs_completed, 2, "runs_completed tracked for the UI")
        eq(rec.auto_research_target_runs, 2, "target_runs tracked for the UI")
        check(
            (h.ws / ".agent_probe" / "fix_plans" / "fix_plans_2.json").exists()
            and (h.ws / ".agent_probe" / "fix_plans" / "fix_plans_3.json").exists(),
            "per-round fix-plan artifacts on disk",
        )
        check(
            (h.state.run_dir / "fix_plans_2.json").exists(),
            "fix plans mirrored into the run's response folder",
        )
    finally:
        h.close()


# ── 3. a regressing round reverts, same as auto-probe ────────────────────
def test_auto_research_revert_on_regression() -> None:
    print("── auto-research revert-on-regression (shared logic) ────")
    h = Harness()
    try:
        st.auto_research_setup(h.state)
        h.stub.improve_by = -0.05  # every plan now makes things worse
        st.auto_research_iterate_batch(h.state, 1)
        rec = h.state.record
        row = rec.iterations[-1]
        check("reverted (no improvement)" in row["note"], "round marked reverted")
        eq(
            float(re.search(r"^VALUE = ([\d.]+)", (h.ws / "train.py").read_text(), re.M).group(1)),
            0.5,
            "train.py rewound to the previous round's snapshot",
        )
        eq(row["best_value"], 0.5, "best holds at the prior value")
    finally:
        h.close()


# ── 4. AUTO-PROBE PROMPTS BYTE-IDENTICAL ────────────────────────────────
def test_auto_probe_prompts_unchanged() -> None:
    print("── auto-probe prompts byte-identical (regression guard) ─")
    h = Harness()
    try:
        h.seed_auto_probe_stage4()
        h.stub.prompts.clear()
        st._do_auto_fix_round(h.state)
        gen, score, apply_p = h.stub.prompts
        eq(gen, f"{PROMPT_NINE}2", "auto-pilot generate prompt == PROMPT_NINE + round")
        eq(score, f"{PROMPT_TEN}2", "auto-pilot confidence prompt == PROMPT_TEN + round")
        plans = json.loads(
            (h.ws / ".agent_probe" / "fix_plans" / "fix_plans_2.json").read_text()
        )["fix_plans"]
        eq(
            apply_p,
            f"{PROMPT_ELEVEN}\n\n{json.dumps(plans[1], indent=2)}",
            "auto-pilot apply prompt == PROMPT_ELEVEN + selected plan",
        )
        check(ADDENDUM not in gen + score + apply_p, "no addendum in auto-probe mode")
        # The conditional line is present but inert: it fires only when the
        # addendum block it names is appended, which auto-probe never does.
        check(
            MODE_LINE_KEY in gen and MODE_LINE_KEY in apply_p,
            "shared prompts carry the conditional auto-research line",
        )
        check(
            ADDENDUM_BLOCK_MARK not in gen and ADDENDUM_BLOCK_MARK not in apply_p,
            "the block that line is conditional on is absent in auto-probe mode",
        )

        # ... and the manual generate path, with a hint.
        h.stub.prompts.clear()
        st.generate_fix_plans(h.state, "look at the optimizer")
        gen, score = h.stub.prompts
        eq(
            gen,
            f"{PROMPT_NINE}3\n\nUser hint:\nlook at the optimizer\n",
            "manual generate prompt unchanged (hint block verbatim)",
        )
        eq(score, f"{PROMPT_TEN}3", "manual confidence prompt unchanged")

        # ... and the user-selected apply path.
        h.stub.prompts.clear()
        st.select_and_apply_fix_plan(h.state, 1)
        plans = json.loads(
            (h.ws / ".agent_probe" / "fix_plans" / "fix_plans_3.json").read_text()
        )["fix_plans"]
        eq(
            h.stub.prompts[0],
            f"{PROMPT_ELEVEN}\n\n{json.dumps(plans[0], indent=2)}",
            "manual apply prompt unchanged",
        )
        eq(
            [r["index"] for r in h.state.record.iterations],
            [1, 2, 3],
            "auto-probe iteration ledger unchanged in shape",
        )
        check(
            h.state.record.iterations[-1]["best_value"] is None,
            "best_value stays None in auto-probe mode",
        )
    finally:
        h.close()


# ── 4b. auto-probe auto-pilot still stops where it used to ──────────────
def test_auto_probe_loop_unchanged() -> None:
    print("── auto-probe auto-pilot terminal behaviour ─────────────")
    h = Harness()
    try:
        h.seed_auto_probe_stage4()
        out = st.auto_fix_loop(h.state)
        eq(out["rounds"], 3, "auto-pilot hard-caps at 3 fix-attempt rounds")
        eq(
            len(h.state.record.iterations),
            4,
            "stops at 4 rows (baseline + 3 attempts)",
        )
        eq(h.state.record.phase, "ready", "parks the run at phase=ready")
        check(
            all(r["best_value"] is None for r in h.state.record.iterations),
            "no auto-research best bookkeeping leaks into auto-probe rows",
        )
        try:
            st.auto_fix_loop(h.state)
            check(True, "auto-pilot is re-enterable at a terminal state (no-op)")
        except Exception as e:  # noqa: BLE001
            check(False, f"auto-pilot re-entry raised {e!r}")
    finally:
        h.close()


# ── 5. manual fix-plan path works in auto-research ──────────────────────
def test_auto_research_manual_round() -> None:
    print("── auto-research manual fix-plan round ─────────────────")
    h = Harness()
    try:
        st.auto_research_setup(h.state)
        h.stub.prompts.clear()
        out = st.generate_fix_plans(h.state, "try stronger regularisation")
        eq(out["round"], 2, "fix plans opened for round 2")
        eq(len(out["fix_plans"]), 3, "three candidate plans returned to the UI")
        check(
            "User hint:\ntry stronger regularisation" in h.stub.prompts[0],
            "hint forwarded in auto-research mode too",
        )
        check(ADDENDUM in h.stub.prompts[0], "addendum present on the generate prompt")
        res = st.select_and_apply_fix_plan(h.state, 3)
        eq(res["iteration"]["index"], 2, "user-selected plan produced round 2")
        eq(res["iteration"]["fix_plan_chosen_index"], 3, "the user's pick was applied")
        eq(res["iteration"]["best_value"], 0.52, "running best advanced")
        eq(
            h.state.record.iterations[-1]["best_value"],
            0.52,
            "stored row carries the best series",
        )
    finally:
        h.close()


# ── 5b. a legacy auto-research run (empty ledger) still works ──────────
def test_legacy_auto_research_resume() -> None:
    print("── legacy auto-research run recovers a baseline row ────")
    h = Harness()
    try:
        st.auto_research_setup(h.state)
        # Simulate a run parked at stage 4 by the RETIRED setup, which cleared
        # the iteration ledger and left only the metric artifact behind.
        h.state.record.iterations = []
        h.state.record.auto_research_best_value = None
        h.state.save()
        st.auto_research_iterate_batch(h.state, 1)
        rows = h.state.record.iterations
        eq(len(rows), 2, "baseline row recovered, then one real round")
        check(
            "recovered from existing metric" in rows[0]["note"],
            "baseline row is labelled as recovered",
        )
        eq(rows[0]["index"], 1, "recovered row keeps the on-disk round index")
        eq(rows[-1]["index"], 2, "the new round is round 2")
        eq(rows[-1]["best_value"], 0.52, "best advanced from the recovered baseline")
    finally:
        h.close()


# ── 6. static guards ────────────────────────────────────────────────────
def test_static_guards() -> None:
    print("── static guards ───────────────────────────────────────")
    stages_src = (REPO / "pipeline" / "stages.py").read_text()
    check(
        "agent_improve_commentor" not in stages_src
        and "{PROMPT_SIX}" not in stages_src,
        "stages.py neither imports nor sends the commentor prompt",
    )
    check(
        "ITERATION_IMPROVEMENT" not in re.sub(r"(?m)^\s*#.*$", "", stages_src),
        "stages.py no longer uses the comment-driven iteration prompt",
    )
    for fn in ("generate_fix_plans", "auto_fix_loop", "select_and_apply_fix_plan"):
        body = stages_src.split(f"def {fn}(")[1].split("\ndef ")[0]
        check(
            "not used in auto-research mode" not in body,
            f"{fn} no longer rejects auto-research runs",
        )
    # The conditional line: present in the two shared prompts that assume
    # thresholds, absent from the one that doesn't (PROMPT_TEN never mentions
    # a threshold), and keyed to the addendum block by name.
    check(MODE_LINE_KEY in PROMPT_NINE, "PROMPT_NINE carries the conditional line")
    check(MODE_LINE_KEY in PROMPT_ELEVEN, "PROMPT_ELEVEN carries the conditional line")
    check(
        MODE_LINE_KEY not in PROMPT_TEN,
        "PROMPT_TEN left alone (it makes no threshold assumption)",
    )
    for name, prompt in (("PROMPT_NINE", PROMPT_NINE), ("PROMPT_ELEVEN", PROMPT_ELEVEN)):
        line = next(l for l in prompt.split("\n") if MODE_LINE_KEY in l)
        check(
            line.lstrip().startswith("AUTO-RESEARCH MODE (conditional")
            and ADDENDUM_HEADER in line
            and "If that block is absent" in line,
            f"{name}'s line is a single conditional keyed to the addendum block",
        )
        eq(
            prompt.count(MODE_LINE_KEY), 1, f"{name} gained exactly one such line",
        )
    check(
        ADDENDUM_BLOCK_MARK in ADDENDUM,
        "the addendum announces itself under the name the line looks for",
    )
    # Trimming the addendum's retractions must not have dropped the facts.
    check(
        "no `status` field" in ADDENDUM and "`conclude()` takes no" in ADDENDUM,
        "addendum still states no-threshold + no-argument conclude()",
    )
    tier1 = SETUP_PROMPT.index("TIER 1: REUSE THE METRIC")
    tier2 = SETUP_PROMPT.index("TIER 2: AUPRC")
    check(tier1 < tier2, "setup prompt prefers train.py's own metric over AUPRC")
    check(
        "ANCHOR: original train metric" in SETUP_PROMPT,
        "setup prompt wires the anchor the shared prompts protect",
    )
    try:
        st.iterate_once(None)  # type: ignore[arg-type]
        check(False, "stale iterate_once still raises")
    except RuntimeError:
        check(True, "stale iterate_once still raises")

    # Terminal-state rule (auto-probe only) must be untouched.
    mk = lambda n, status: [  # noqa: E731
        {"index": i + 1, "tail_mean": 0.5, "direction": "higher_is_better",
         "status": status, "acceptable_met": False}
        for i in range(n)
    ]

    class _Fake:
        def __init__(self, iters):
            self.record = type("R", (), {"iterations": iters})()

    eq(st._at_terminal_state(_Fake(mk(3, "FAIL"))), False, "3 FAIL rows: keep going")
    eq(st._at_terminal_state(_Fake(mk(4, "FAIL"))), True, "4 FAIL rows: terminal")
    eq(st._at_terminal_state(_Fake(mk(2, "PASS"))), True, "PASS: terminal")


if __name__ == "__main__":
    test_auto_research_setup()
    test_auto_research_batch()
    test_auto_research_revert_on_regression()
    test_auto_probe_prompts_unchanged()
    test_auto_probe_loop_unchanged()
    test_auto_research_manual_round()
    test_legacy_auto_research_resume()
    test_static_guards()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
