"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, FixPlan, IterationRow, RunRecord } from "@/lib/api";
import {
  Button,
  Card,
  ConfidenceBar,
  Modal,
  ordinal,
  Pill,
  SectionLabel,
  Spinner,
} from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";
import { useActionLatch } from "@/lib/useActionLatch";

// One state machine drives every popup on this stage. Only TERMINAL
// classifications open a modal — auto-pilot handles non-terminal rounds
// silently.
//   - status == PASS → "pass" modal
//   - status == FAIL + improving enough + AT met → "best-effort" modal
//   - status == FAIL + not-improving → "stagnant" modal (give up | drop to
//     manual). The "continue-fixing" FailMode is non-terminal and does not
//     open a modal — the auto-pilot just loops another round.
type ModalStep =
  | null
  | "pass"
  | "pass-discard"
  | "pass-keep"
  | "best-effort"
  | "best-effort-discard"
  | "best-effort-keep"
  | "stagnant"
  // Auto-research post-batch flow (unchanged):
  | "ar-done"
  | "ar-more";

type FailMode = "continue-fixing" | "best-effort" | "stagnant";

export function Stage4({
  run,
  onUpdate,
  onKeepRebase,
}: {
  run: RunRecord;
  onUpdate: () => void;
  // Fired after a successful "keep changes & re-baseline" revert. The page
  // owns the follow-up info modal because Stage4 unmounts once stage flips
  // back to 1.
  onKeepRebase?: () => void;
}) {
  const { inProgress: running, begin, fail } = useActionLatch(run.busy);
  const [error, setError] = useState<string | null>(null);
  const [reverting, setReverting] = useState(false);
  const [step, setStep] = useState<ModalStep>(null);
  // Flips on when the user opted into the fix-plan flow this round. We use it
  // to label the big button "Generate fixing plans" instead of "Start auto
  // probe-fixing".
  const [fixingMode, setFixingMode] = useState(false);
  // Auto-research: how many rounds to run in the next batch.
  const [arRounds, setArRounds] = useState(10);
  const [arMoreRounds, setArMoreRounds] = useState(10);
  // Fix-plan list cached locally so the polled run state doesn't force a
  // refetch every tick. We hydrate from the run record's fix_plan_round.
  const [fixPlans, setFixPlans] = useState<FixPlan[] | null>(null);
  const [pickingPlan, setPickingPlan] = useState<number | null>(null);
  // Optional user hint, fed verbatim to the fix-plan generator agent.
  // Cleared after a generation kicks off so it doesn't bleed into the next.
  const [fixHint, setFixHint] = useState("");

  const passed = run.iterations.some((i) => i.status === "PASS");
  const isAutoResearch = run.debug_flags.auto_research;
  const lastIter = run.iterations[run.iterations.length - 1];
  const hasOpenFixPlans = run.fix_plan_round != null;

  // Track which iteration we've already prompted on (so polling/refresh
  // doesn't re-pop the modal). Key: `<run_id>:<iter_index>:<status>`.
  const lastPromptKeyRef = useRef<string | null>(null);
  const lastArBatchKeyRef = useRef<string | null>(null);

  // Reset per-run UI state when switching runs.
  useEffect(() => {
    setFixingMode(false);
    setFixPlans(null);
    setFixHint("");
    lastPromptKeyRef.current = null;
    lastArBatchKeyRef.current = null;
  }, [run.run_id]);

  // Pull the open fix-plan set whenever the server says one exists.
  useEffect(() => {
    if (!hasOpenFixPlans) {
      setFixPlans(null);
      return;
    }
    api
      .getFixPlans(run.run_id)
      .then((r) => setFixPlans(r.fix_plans ?? null))
      .catch(() => {});
  }, [hasOpenFixPlans, run.run_id, run.fix_plan_round]);

  // Auto-research: pop the post-batch dialog once a batch finishes.
  useEffect(() => {
    if (!isAutoResearch) return;
    if (running || run.busy) return;
    const target = run.auto_research_target_runs ?? 0;
    const completed = run.auto_research_runs_completed ?? 0;
    if (target <= 0 || completed < target) return;
    const key = `${run.run_id}:${target}:${completed}`;
    if (lastArBatchKeyRef.current === key) return;
    lastArBatchKeyRef.current = key;
    setStep("ar-done");
  }, [
    isAutoResearch,
    running,
    run.busy,
    run.run_id,
    run.auto_research_target_runs,
    run.auto_research_runs_completed,
  ]);

  // After each iteration completes, choose which modal to show.
  useEffect(() => {
    if (isAutoResearch) return;
    if (running || run.busy) return;
    if (!lastIter || !lastIter.status) return;
    const key = `${run.run_id}:${lastIter.index}:${lastIter.status}`;
    if (lastPromptKeyRef.current === key) return;
    lastPromptKeyRef.current = key;
    if (lastIter.status === "PASS") {
      setStep("pass");
      return;
    }
    // Only terminal classifications open a modal. "continue-fixing" is the
    // implicit state — the auto-pilot handles it without surfacing UI; the
    // user only needs to act on terminal outcomes (best-effort / stagnant)
    // or PASS.
    const mode = classifyFail(run.iterations);
    if (mode === "best-effort") setStep("best-effort");
    else if (mode === "stagnant") setStep("stagnant");
    // mode === "continue-fixing": no modal, let the loop / user advance.
  }, [
    isAutoResearch,
    running,
    run.busy,
    run.run_id,
    run.iterations,
    lastIter?.index,
    lastIter?.status,
  ]);

  // "Start auto probe-fixing" — backend loops fix-plan rounds (auto-pick
  // highest confidence) until terminal state. Only stops at PASS /
  // best-effort / stagnant.
  async function handleAutoFixLoop() {
    begin();
    setError(null);
    try {
      await api.autoFixLoop(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    }
  }

  const handleGenerateFixPlans = useCallback(async () => {
    begin();
    setError(null);
    const hintToSend = fixHint.trim() || undefined;
    try {
      const r = await api.generateFixPlans(run.run_id, hintToSend);
      setFixPlans(r.fix_plans ?? null);
      setFixHint("");
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    }
  }, [run.run_id, onUpdate, fixHint, begin, fail]);

  async function handlePickFixPlan(index: number) {
    begin();
    setPickingPlan(index);
    setError(null);
    try {
      await api.selectFixPlan(run.run_id, index);
      setFixPlans(null);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    } finally {
      setPickingPlan(null);
    }
  }

  async function handleAutoResearchBatch(count: number) {
    if (count <= 0) return;
    lastArBatchKeyRef.current = null;
    begin();
    setError(null);
    setStep(null);
    try {
      await api.autoResearchIterate(run.run_id, count);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    }
  }

  async function performRevert(toStage: number, keep: boolean = false) {
    setReverting(true);
    setError(null);
    try {
      await api.revert(run.run_id, toStage, keep);
      setStep(null);
      if (keep) onKeepRebase?.();
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setReverting(false);
    }
  }

  // "Continue" on the stagnant modal drops the user into manual mode: the
  // hint textarea appears and the big button becomes "Generate fixing
  // plans" so they can drive the next round themselves. Auto-pilot is off
  // once we're here — they explicitly took control.
  function handleStagnantContinueChoice() {
    setFixingMode(true);
    setStep(null);
  }

  const trd = computeTrd(run.iterations);
  const bigButtonDisabled = running || run.busy || passed || hasOpenFixPlans;
  const bigButtonLabel = fixingMode
    ? "Generate fixing plans"
    : "Start auto probe-fixing";

  return (
    <div className="space-y-6">
      <Header
        n={4}
        title="Probe Fixing"
        subtitle={
          isAutoResearch
            ? "Each round, the agent applies one of the 10 # potential_improvement comments seeded in train.py and re-runs training. Auto-research has no PASS gate — keep running rounds as long as the metric is still moving."
            : "Each round, the agent edits train.py to push the probe metric. When a round fails, you'll see fix plans you can pick from. Stops automatically when status becomes PASS."
        }
      />

      {isAutoResearch ? (
        <section className="flex items-center gap-3 flex-wrap">
          <label className="text-[13px] text-ink-700 font-medium">
            Rounds
          </label>
          <input
            type="number"
            min={1}
            max={100}
            value={arRounds}
            onChange={(e) => setArRounds(clampInt(e.target.value, 1, 100, 10))}
            disabled={running || run.busy}
            className="w-20 h-9 px-2 rounded-md border border-ink-200 bg-white font-mono text-[13px] focus:border-ink-400"
          />
          <Button
            onClick={() => handleAutoResearchBatch(arRounds)}
            disabled={running || run.busy}
          >
            {running ? (
              <>
                <Spinner /> Running…
              </>
            ) : (
              `Start auto-research (${arRounds} round${arRounds === 1 ? "" : "s"})`
            )}
          </Button>
          <span className="text-[12px] text-ink-500">
            {run.iterations.length} run{run.iterations.length === 1 ? "" : "s"}{" "}
            so far
          </span>
          <Pill tone="neutral">auto-research</Pill>
          {run.auto_research_best_value != null && (
            <span className="text-[12px] text-ink-700 font-mono">
              best: {run.auto_research_best_value.toFixed(4)}
            </span>
          )}
        </section>
      ) : (
        <section className="flex items-center gap-3 flex-wrap">
          <Button
            onClick={fixingMode ? handleGenerateFixPlans : handleAutoFixLoop}
            disabled={bigButtonDisabled}
          >
            {running ? (
              <>
                <Spinner /> Running…
              </>
            ) : passed ? (
              "PASSED — done"
            ) : fixingMode && hasOpenFixPlans ? (
              "Pick a fix plan below"
            ) : (
              bigButtonLabel
            )}
          </Button>
          <span className="text-[12px] text-ink-500">
            {run.iterations.length} run{run.iterations.length === 1 ? "" : "s"}{" "}
            so far
          </span>
          {passed && <Pill tone="pass">PASS</Pill>}
          {!passed && trd != null && (
            <span
              className="text-[12px] text-ink-500 font-mono"
              title="Signed improvement between the latest tail_mean and the tail_mean 3 rounds ago, relative to direction. Positive = improving."
            >
              TRD: {trd.toFixed(4)}
            </span>
          )}
        </section>
      )}

      {/* Optional hint for the fix-plan generator — only shown while in
          fixing mode and no plans are currently open. */}
      {!isAutoResearch && fixingMode && !hasOpenFixPlans && !passed && (
        <section className="space-y-1.5">
          <SectionLabel>Hint for the fix-plan agent (optional)</SectionLabel>
          <textarea
            value={fixHint}
            onChange={(e) => setFixHint(e.target.value)}
            rows={3}
            disabled={running || run.busy}
            placeholder="e.g. 'try data augmentation on the minority classes' or 'the model might be underfitting'. The agent treats this as non-binding context."
            className="w-full rounded-md border border-ink-200 bg-white px-3 py-2 text-[13px] focus:border-ink-400 disabled:bg-ink-50/50"
          />
          <div className="text-[11px] text-ink-500">
            Leave empty to let the agent decide entirely from the codebase and
            iteration history.
          </div>
        </section>
      )}

      {isAutoResearch ? (
        <>
          <section className="space-y-2">
            <SectionLabel>Live metric (current run)</SectionLabel>
            <MetricChart runId={run.run_id} live={running || run.busy} />
          </section>
          <section className="space-y-2">
            <SectionLabel>Best metric per run (monotonic)</SectionLabel>
            <PerRunBestChart
              rows={run.iterations}
              direction={run.auto_research_best_direction}
            />
          </section>
        </>
      ) : (
        <section className="space-y-2">
          <SectionLabel>Live metric (current run)</SectionLabel>
          <MetricChart runId={run.run_id} live={running || run.busy} />
        </section>
      )}

      {/* Fix plans — manual mode only. During auto-pilot the backend
          briefly sets fix_plan_round between generate and apply; we keep
          the panel hidden then so it doesn't flicker. fixingMode is only
          true after the user clicked "Continue" on the stagnant modal. */}
      {!isAutoResearch && fixingMode && hasOpenFixPlans && (
        <section className="space-y-3">
          <SectionLabel>
            Fix plans for {ordinal(run.fix_plan_round ?? 0)} run
          </SectionLabel>
          {fixPlans === null ? (
            <div className="rounded-md border border-dashed border-ink-200 px-3 py-6 text-center text-[12px] text-ink-400">
              {running || run.busy ? "generating fix plans…" : "loading…"}
            </div>
          ) : (
            <div className="space-y-3">
              {fixPlans.map((p, i) => {
                const idx = i + 1;
                return (
                  <Card key={i} className="p-4">
                    <div className="flex items-start gap-3">
                      <div className="font-mono text-[11px] text-ink-500 w-7 mt-0.5">
                        {String(idx).padStart(2, "0")}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Pill tone="neutral">fix plan</Pill>
                          <span className="font-medium text-[13px] text-ink-900">
                            {p.title}
                          </span>
                          <div className="ml-auto">
                            <ConfidenceBar value={p.confidence} />
                          </div>
                        </div>
                        <p className="text-[12.5px] text-ink-700 leading-relaxed whitespace-pre-wrap">
                          {p.content}
                        </p>
                        {p.target_files?.length > 0 && (
                          <div className="text-[11px] text-ink-500">
                            <span className="uppercase tracking-wide font-medium">
                              files:
                            </span>{" "}
                            <span className="font-mono">
                              {p.target_files.join(" · ")}
                            </span>
                          </div>
                        )}
                        <div className="pt-1">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => handlePickFixPlan(idx)}
                            disabled={pickingPlan !== null || running || run.busy}
                          >
                            {pickingPlan === idx ? (
                              <Spinner />
                            ) : (
                              "Pick & apply"
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </section>
      )}

      <section className="space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <SectionLabel>Run history</SectionLabel>
          {!isAutoResearch && (
            <ResultLegend
              standard={lastIter?.threshold ?? null}
              acceptable={lastIter?.acceptable_threshold ?? null}
            />
          )}
        </div>
        {run.iterations.length === 0 ? (
          <div className="rounded-md border border-dashed border-ink-200 px-3 py-6 text-center text-[12px] text-ink-400">
            no runs yet
          </div>
        ) : (
          <IterTable rows={run.iterations} />
        )}
      </section>

      {error && (
        <div className="px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
          {error}
        </div>
      )}

      {/* ── PASS path ──────────────────────────────────────────────────── */}
      <Modal open={step === "pass"} onClose={() => setStep(null)}>
        <div className="p-7 space-y-5">
          <div className="flex items-center gap-3">
            <Pill tone="pass">PASS</Pill>
            <h2 className="text-[20px] font-semibold text-ink-950">
              Probe passed — try another?
            </h2>
          </div>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            The metric crossed the standard threshold. You can pick a
            different probe to test against the same project. Choose how to
            handle the train.py modifications produced during probe-fixing:
          </p>
          <div className="space-y-3">
            <ChoiceRow
              title="Discard changes & re-pick probe"
              body="Restore train.py to its original baseline, return to Probe Design, keep the existing probe candidates. The probe you just passed will be greyed out."
              onClick={() => setStep("pass-discard")}
            />
            <ChoiceRow
              title="Keep changes & re-baseline"
              body="Treat the modified train.py as the new starting point. Probe candidates will be regenerated against the updated code for coherence."
              onClick={() => setStep("pass-keep")}
            />
          </div>
          <div className="flex justify-end pt-1">
            <Button variant="ghost" onClick={() => setStep(null)}>
              Stay on Probe Fixing
            </Button>
          </div>
        </div>
      </Modal>

      <Modal open={step === "pass-discard"} onClose={() => setStep("pass")}>
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Discard changes?
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            All Dev Plan, Implementation and Probe Fixing artifacts will be
            erased and{" "}
            <span className="font-mono">train.py</span> will be restored to its
            original baseline. The probe you passed will appear as
            <span className="px-1.5 mx-1 rounded bg-amber-50 text-amber-700 border border-amber-100 text-[14px]">
              already tried
            </span>
            so you can pick a different one.
          </p>
          <div className="flex justify-end gap-3 pt-1">
            <Button variant="ghost" onClick={() => setStep("pass")}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => performRevert(1, false)}
              disabled={reverting}
            >
              {reverting ? <Spinner /> : "Discard & re-pick"}
            </Button>
          </div>
        </div>
      </Modal>

      <Modal open={step === "pass-keep"} onClose={() => setStep("pass")}>
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Keep changes & re-baseline?
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            The current <span className="font-mono">train.py</span> becomes the
            new baseline. Probe candidates, dev plans, prober and metrics will
            all be cleared — you'll need to re-generate probe candidates so
            they reflect the updated code.
          </p>
          <div className="flex justify-end gap-3 pt-1">
            <Button variant="ghost" onClick={() => setStep("pass")}>
              Cancel
            </Button>
            <Button
              onClick={() => performRevert(1, true)}
              disabled={reverting}
            >
              {reverting ? <Spinner /> : "Keep & re-baseline"}
            </Button>
          </div>
        </div>
      </Modal>

      {/* ── Best-effort: improving + AT met, but ST never crossed ───────── */}
      <Modal open={step === "best-effort"} onClose={() => setStep(null)}>
        <div className="p-7 space-y-5">
          <div className="flex items-center gap-3">
            <Pill tone="warn">ACCEPTABLE</Pill>
            <h2 className="text-[20px] font-semibold text-ink-950">
              Best we could do
            </h2>
          </div>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            The metric is improving and has crossed the{" "}
            <span className="font-mono">acceptable threshold</span>, but
            hasn't reached the strict standard threshold. Given the project's
            constraints and base conditions, this is realistically as far as
            this probe can be pushed. Choose how to wrap up:
          </p>
          <div className="space-y-3">
            <ChoiceRow
              title="Discard changes & re-pick probe"
              body="Restore train.py and return to Probe Design. The probe you just exhausted will be greyed out."
              onClick={() => setStep("best-effort-discard")}
            />
            <ChoiceRow
              title="Keep changes & re-baseline"
              body="Lock in the improvements made so far as the new baseline. Probe candidates regenerate against the updated code."
              onClick={() => setStep("best-effort-keep")}
            />
          </div>
          <div className="flex justify-end pt-1">
            <Button variant="ghost" onClick={() => setStep(null)}>
              Stay on Probe Fixing
            </Button>
          </div>
        </div>
      </Modal>

      <Modal
        open={step === "best-effort-discard"}
        onClose={() => setStep("best-effort")}
      >
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Discard and re-pick?
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            All Dev Plan, Implementation and Probe Fixing artifacts will be
            erased and <span className="font-mono">train.py</span> will be
            restored to its original baseline.
          </p>
          <div className="flex justify-end gap-3 pt-1">
            <Button variant="ghost" onClick={() => setStep("best-effort")}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => performRevert(1, false)}
              disabled={reverting}
            >
              {reverting ? <Spinner /> : "Discard & re-pick"}
            </Button>
          </div>
        </div>
      </Modal>

      <Modal
        open={step === "best-effort-keep"}
        onClose={() => setStep("best-effort")}
      >
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Keep changes & re-baseline?
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            The current <span className="font-mono">train.py</span> becomes the
            new baseline. Probe candidates, dev plans, prober and metrics will
            all be cleared — you'll need to re-generate probe candidates so
            they reflect the updated code.
          </p>
          <div className="flex justify-end gap-3 pt-1">
            <Button variant="ghost" onClick={() => setStep("best-effort")}>
              Cancel
            </Button>
            <Button
              onClick={() => performRevert(1, true)}
              disabled={reverting}
            >
              {reverting ? <Spinner /> : "Keep & re-baseline"}
            </Button>
          </div>
        </div>
      </Modal>

      {/* ── Stagnant: not improving (and/or improved but AT untouched) ──── */}
      <Modal open={step === "stagnant"} onClose={() => setStep(null)}>
        <div className="p-7 space-y-5">
          <div className="flex items-center gap-3">
            <Pill tone="fail">STAGNANT</Pill>
            <h2 className="text-[20px] font-semibold text-ink-950">
              Auto-pilot stopped
            </h2>
          </div>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            After 3 fix-attempt rounds the metric hasn't reached either the
            standard or the acceptable threshold (and isn't on a clear path
            to). The scripted loop won't get there on its own. Pick one:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <FailChoice
              tone="danger"
              title="Give up"
              body="Back to Probe Design and pick a different probe. The one you just tried will be greyed out."
              disabled={reverting}
              busy={reverting}
              onClick={() => performRevert(1, false)}
            />
            <FailChoice
              tone="secondary"
              title="Continue manually"
              body="Drop to manual mode: an optional hint textarea + you pick which fix plan to apply each round. Auto-pilot stays off — you drive each round explicitly."
              disabled={reverting}
              onClick={handleStagnantContinueChoice}
            />
          </div>
        </div>
      </Modal>

      {/* ── Auto-research: batch complete ──────────────────────────────── */}
      <Modal open={step === "ar-done"} onClose={() => setStep(null)}>
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Auto-research batch complete
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            Finished {run.auto_research_runs_completed} of{" "}
            {run.auto_research_target_runs} round
            {run.auto_research_target_runs === 1 ? "" : "s"}. Best metric so
            far:{" "}
            <span className="font-mono font-medium">
              {run.auto_research_best_value != null
                ? run.auto_research_best_value.toFixed(4)
                : "—"}
            </span>
            {run.auto_research_best_direction && (
              <>
                {" "}
                ({run.auto_research_best_direction.replace(/_/g, " ")})
              </>
            )}
            .
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              onClick={() => {
                setArMoreRounds(arRounds);
                setStep("ar-more");
              }}
              className="text-left rounded-lg border-2 border-ink-300 hover:border-ink-500 hover:bg-ink-50 px-5 py-4 transition-colors"
            >
              <div className="text-[16px] font-semibold text-ink-900">
                Run more rounds
              </div>
              <div className="mt-1 text-[13px] text-ink-600 leading-relaxed">
                Pick how many additional rounds to run on top of the current
                best train.py.
              </div>
            </button>
            <button
              onClick={() => performRevert(1, false)}
              disabled={reverting}
              className="text-left rounded-lg border-2 border-red-200 hover:border-red-400 hover:bg-red-50 px-5 py-4 transition-colors disabled:opacity-50"
            >
              <div className="text-[16px] font-semibold text-red-700 flex items-center gap-2">
                {reverting && <Spinner size={14} />}
                Back to Probe Design (clean everything)
              </div>
              <div className="mt-1 text-[13px] text-ink-600 leading-relaxed">
                Discard the prober, all auto-research changes, and the metric
                history. train.py is restored to its original baseline and
                you'll start over at the beginning of Probe Design.
              </div>
            </button>
          </div>
          <div className="flex justify-end pt-1">
            <Button variant="ghost" onClick={() => setStep(null)}>
              Stay on Probe Fixing
            </Button>
          </div>
        </div>
      </Modal>

      <Modal open={step === "ar-more"} onClose={() => setStep("ar-done")}>
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            How many more rounds?
          </h2>
          <div className="flex items-center gap-3">
            <input
              type="number"
              min={1}
              max={100}
              value={arMoreRounds}
              onChange={(e) =>
                setArMoreRounds(clampInt(e.target.value, 1, 100, 10))
              }
              autoFocus
              className="w-28 h-10 px-3 rounded-md border-2 border-ink-200 bg-white font-mono text-[16px] focus:border-ink-400"
            />
            <span className="text-[14px] text-ink-600">
              rounds (1–100)
            </span>
          </div>
          <div className="flex justify-end gap-3 pt-1">
            <Button variant="ghost" onClick={() => setStep("ar-done")}>
              Cancel
            </Button>
            <Button
              onClick={() => handleAutoResearchBatch(arMoreRounds)}
              disabled={running || run.busy}
            >
              {running ? <Spinner /> : `Run ${arMoreRounds} more`}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

// ── TRD computation ─────────────────────────────────────────────────────────
// TRD = signed delta between the latest tail_mean and the tail_mean 3 rounds
// ago, oriented so positive always means "moving toward better". TRD is only
// meaningful once we have at least 3 actual fix-attempt rounds — row 1 is
// the stage-3 integration baseline (not a fix attempt), so the minimum row
// count for a valid TRD is 4 (baseline + 3 fix attempts).
const MIN_ROWS_FOR_TRD = 4;

function computeTrd(rows: IterationRow[]): number | null {
  if (rows.length < MIN_ROWS_FOR_TRD) return null;
  const last = rows[rows.length - 1];
  const tm = (r: IterationRow): number | null =>
    r.tail_mean != null ? r.tail_mean : r.metric_value;
  const latest = tm(last);
  if (latest == null) return null;
  const ref = rows[rows.length - MIN_ROWS_FOR_TRD];
  const refVal = tm(ref);
  if (refVal == null) return null;
  const dir = last.direction ?? ref.direction ?? "higher_is_better";
  const raw = latest - refVal;
  return dir === "lower_is_better" ? -raw : raw;
}

// "Improving enough" = TRD positive and beyond noise (1% of |ref|).
function isImproving(rows: IterationRow[]): boolean {
  const trd = computeTrd(rows);
  if (trd == null) return false;
  const last = rows[rows.length - 1];
  const ref = last.tail_mean ?? last.metric_value ?? 1;
  const noise = Math.max(0.01 * Math.abs(ref), 1e-6);
  return trd > noise;
}

// Classifies a FAIL row. Only "best-effort" and "stagnant" are terminal —
// "continue-fixing" tells the auto-loop to keep going and produces no modal.
// Rules:
//   - < MIN_ROWS_FOR_TRD rows: continue-fixing (auto-loop hasn't accumulated
//     enough data yet).
//   - improving + AT met → best-effort.
//   - anything else after MIN_ROWS_FOR_TRD rows → stagnant.
// Net effect: the auto-loop hard-caps at 3 fix-attempt rounds. After that
// the only terminal outcomes are PASS, best-effort, or stagnant.
function classifyFail(rows: IterationRow[]): FailMode {
  const last = rows[rows.length - 1];
  if (!last) return "continue-fixing";
  if (rows.length < MIN_ROWS_FOR_TRD) return "continue-fixing";
  const improving = isImproving(rows);
  const atMet = last.acceptable_met === true;
  if (improving && atMet) return "best-effort";
  return "stagnant";
}

function ResultLegend({
  standard,
  acceptable,
}: {
  standard: string | null;
  acceptable: string | null;
}) {
  // Surface the actual numbers next to each pill — the chart's auto y-range
  // can hide the threshold lines when they sit far outside the data range,
  // so the legend is sometimes the only place the user can see them.
  return (
    <div className="flex items-center gap-3 flex-wrap text-[11px] text-ink-500">
      <span className="inline-flex items-center gap-1.5">
        <Pill tone="pass">PASS</Pill>
        <span>
          met standard
          {standard != null && (
            <>
              {" "}
              (<span className="font-mono text-green-600">{standard}</span>)
            </>
          )}
        </span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <Pill tone="fail">FAIL</Pill>
        <span>
          didn't reach standard
          {standard != null && (
            <>
              {" "}
              (<span className="font-mono text-green-600">{standard}</span>)
            </>
          )}
        </span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <Pill tone="warn">ACCEPT</Pill>
        <span>
          met acceptable
          {acceptable != null && (
            <>
              {" "}
              (<span className="font-mono text-amber-600">{acceptable}</span>)
            </>
          )}
        </span>
      </span>
    </div>
  );
}

function FailChoice({
  tone,
  title,
  body,
  onClick,
  disabled,
  busy,
}: {
  tone: "danger" | "secondary" | "primary";
  title: string;
  body: string;
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
}) {
  const styles =
    tone === "danger"
      ? "border-red-200 hover:border-red-400 hover:bg-red-50 text-red-700"
      : tone === "primary"
        ? "border-ink-300 hover:border-ink-500 hover:bg-ink-50 text-ink-900"
        : "border-amber-200 hover:border-amber-400 hover:bg-amber-50 text-amber-800";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`text-left rounded-lg border-2 px-5 py-4 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${styles}`}
    >
      <div className="text-[16px] font-semibold flex items-center gap-2">
        {busy && <Spinner size={14} />}
        {title}
      </div>
      <div className="mt-1 text-[13px] text-ink-600 leading-relaxed">
        {body}
      </div>
    </button>
  );
}

function ChoiceRow({
  title,
  body,
  onClick,
}: {
  title: string;
  body: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left rounded-lg border-2 border-ink-200 px-5 py-4 hover:border-ink-400 hover:bg-ink-50 transition-colors"
    >
      <div className="text-[16px] font-semibold text-ink-900">{title}</div>
      <div className="mt-1 text-[14px] text-ink-600 leading-relaxed">
        {body}
      </div>
    </button>
  );
}

function clampInt(raw: string, lo: number, hi: number, fallback: number): number {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

function PerRunBestChart({
  rows,
  direction,
  height = 180,
}: {
  rows: RunRecord["iterations"];
  direction: "higher_is_better" | "lower_is_better" | null;
  height?: number;
}) {
  const points = rows
    .map((r) => ({
      x: r.index,
      y: r.best_value != null ? r.best_value : r.metric_value,
    }))
    .filter((p): p is { x: number; y: number } => p.y != null);

  if (points.length === 0) {
    return (
      <div
        className="rounded-md border border-dashed border-ink-200 bg-white text-[12px] text-ink-400 italic flex items-center justify-center"
        style={{ height }}
      >
        no rounds yet
      </div>
    );
  }

  const w = 720;
  const h = height;
  const padL = 40;
  const padR = 16;
  const padT = 14;
  const padB = 24;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 0.5;
    yMax += 0.5;
  } else {
    const pad = (yMax - yMin) * 0.12;
    yMin -= pad;
    yMax += pad;
  }
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const x = (e: number) => padL + ((e - xMin) / xRange) * (w - padL - padR);
  const y = (v: number) => padT + (1 - (v - yMin) / yRange) * (h - padT - padB);
  const linePts = points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
  const last = points[points.length - 1];

  return (
    <div className="rounded-md border border-ink-200 bg-white">
      <div className="px-3 py-2 border-b border-ink-100 flex items-center gap-2">
        <SectionLabel>Best per run</SectionLabel>
        <span className="font-mono text-[11px] text-ink-500">
          {direction ? direction.replace(/_/g, " ") : "monotonic"}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="font-mono tabular-nums text-[13px] text-ink-900">
            {last.y.toFixed(4)}
          </span>
          <span className="font-mono text-[11px] text-ink-500">
            @ run {last.x}
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" className="block">
        {ticks.map((t, i) => {
          const yy = y(t);
          return (
            <g key={i}>
              <line
                x1={padL}
                x2={w - padR}
                y1={yy}
                y2={yy}
                stroke="#e8e8ea"
                strokeWidth={1}
              />
              <text
                x={padL - 6}
                y={yy + 3}
                fontSize="9"
                textAnchor="end"
                fill="#71717a"
                fontFamily="ui-monospace, monospace"
              >
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}
        <polyline
          fill="none"
          stroke="#16a34a"
          strokeWidth={2}
          points={linePts}
        />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={x(p.x)}
            cy={y(p.y)}
            r={3.4}
            fill="#16a34a"
            stroke="white"
            strokeWidth={1.5}
          />
        ))}
        {points.map((p, i) => (
          <text
            key={`t-${i}`}
            x={x(p.x)}
            y={h - 8}
            fontSize="9"
            textAnchor="middle"
            fill="#71717a"
            fontFamily="ui-monospace, monospace"
          >
            {p.x}
          </text>
        ))}
      </svg>
    </div>
  );
}

function IterTable({ rows }: { rows: RunRecord["iterations"] }) {
  return (
    <div className="rounded-md border border-ink-200 bg-white overflow-hidden">
      <div className="grid grid-cols-[60px_1fr_120px_140px] text-[10px] uppercase tracking-wider text-ink-500 px-3 py-2 border-b border-ink-100 bg-ink-50/50">
        <div>#</div>
        <div>probe metric</div>
        <div className="text-right">probe value</div>
        <div>probe result</div>
      </div>
      {rows.map((r, i) => (
        <div
          key={i}
          className="grid grid-cols-[60px_1fr_120px_140px] items-center px-3 py-2 border-b border-ink-50 last:border-0 text-[12px]"
        >
          <div className="font-mono text-ink-700">{r.index}</div>
          <div className="text-ink-700 truncate" title={r.metric_name ?? ""}>
            {r.metric_name ?? "—"}
            {r.note && (
              <span className="ml-2 text-[10px] text-ink-400">{r.note}</span>
            )}
          </div>
          <div className="font-mono tabular-nums text-ink-800 text-right">
            {r.metric_value != null ? r.metric_value.toFixed(4) : "—"}
          </div>
          <div className="flex items-center gap-1.5">
            {r.status === "PASS" && <Pill tone="pass">PASS</Pill>}
            {r.status === "FAIL" && <Pill tone="fail">FAIL</Pill>}
            {!r.status && <span className="text-ink-400">—</span>}
            {r.acceptable_met === true && r.status !== "PASS" && (
              <Pill tone="warn">ACCEPT</Pill>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
