"use client";

import { useEffect, useRef, useState } from "react";
import { api, RunRecord } from "@/lib/api";
import { Button, Modal, ordinal, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";

// One state machine drives every popup on this stage.
// Triggers:
//   - last iteration status == PASS (and unseen this run): "pass"
//   - last iteration status == FAIL (and unseen this run): "fail"
// "fail" is a single modal with three side-by-side options (give up / relax
// threshold / continue) — no chain.
type ModalStep =
  | null
  | "pass"
  | "pass-discard"
  | "pass-keep"
  | "fail"
  // Auto-research post-batch flow:
  | "ar-done"   // batch finished — choose run-more or back
  | "ar-more";  // user wants to run more rounds — pick count

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
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reverting, setReverting] = useState(false);
  const [step, setStep] = useState<ModalStep>(null);
  // Flips to true once the user clicks "Continue probe-fixing" in the FAIL
  // modal. The big action button rewords itself to "Continue auto
  // probe-fixing" so the user knows what to do next.
  const [continueMode, setContinueMode] = useState(false);
  // Auto-research: how many rounds to run in the next batch.
  const [arRounds, setArRounds] = useState(10);
  // Auto-research: input bound inside the "run more" dialog.
  const [arMoreRounds, setArMoreRounds] = useState(10);

  const passed = run.iterations.some((i) => i.status === "PASS");
  const isAutoResearch = run.debug_flags.auto_research;
  const lastIter = run.iterations[run.iterations.length - 1];

  // Track which iteration we've already prompted on (so polling/refresh
  // doesn't re-pop the modal). Key: `<run_id>:<iter_index>:<status>`.
  const lastPromptKeyRef = useRef<string | null>(null);
  // Tracks the most recent auto-research batch we've already shown the
  // "done" dialog for, so polling/refresh doesn't re-pop it.
  const lastArBatchKeyRef = useRef<string | null>(null);

  // Reset per-run UI state when switching runs.
  useEffect(() => {
    setContinueMode(false);
    lastPromptKeyRef.current = null;
    lastArBatchKeyRef.current = null;
  }, [run.run_id]);

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

  useEffect(() => {
    if (isAutoResearch) return;
    if (running || run.busy) return;
    if (!lastIter || !lastIter.status) return;
    const key = `${run.run_id}:${lastIter.index}:${lastIter.status}`;
    if (lastPromptKeyRef.current === key) return;
    lastPromptKeyRef.current = key;
    if (lastIter.status === "PASS") setStep("pass");
    else if (lastIter.status === "FAIL") setStep("fail");
  }, [
    isAutoResearch,
    running,
    run.busy,
    run.run_id,
    lastIter?.index,
    lastIter?.status,
  ]);

  async function handleIterate() {
    setRunning(true);
    setError(null);
    try {
      await api.iterateOnce(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setRunning(false);
    }
  }

  async function handleAutoResearchBatch(count: number) {
    if (count <= 0) return;
    // Clear the prior batch's "done" key so the post-batch dialog can fire
    // again when this new batch completes (it'd otherwise stay suppressed if
    // the user runs another batch of the same size).
    lastArBatchKeyRef.current = null;
    setRunning(true);
    setError(null);
    setStep(null);
    try {
      await api.autoResearchIterate(run.run_id, count);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setRunning(false);
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

  // "Continue probe-fixing" branch of the FAIL modal: just close it. The
  // user is opting out of give-up / relax-threshold and wants to keep
  // iterating manually with the big button.
  function handleContinueChoice() {
    setContinueMode(true);
    setStep(null);
  }

  return (
    <div className="space-y-6">
      <Header
        n={4}
        title="Probe Fixing"
        subtitle={
          isAutoResearch
            ? "Each round, the agent applies one of the 10 # potential_improvement comments seeded in train.py and re-runs training. Auto-research has no PASS gate — keep running rounds as long as the metric is still moving."
            : "Each round, the agent edits train.py to push the probe metric. Stops automatically when status becomes PASS."
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
        <section className="flex items-center gap-3">
          <Button
            onClick={handleIterate}
            disabled={running || run.busy || passed}
          >
            {running ? (
              <>
                <Spinner /> Running…
              </>
            ) : passed ? (
              "PASSED — done"
            ) : continueMode ? (
              "Continue auto probe-fixing"
            ) : (
              "Start auto probe-fixing"
            )}
          </Button>
          <span className="text-[12px] text-ink-500">
            {run.iterations.length} run{run.iterations.length === 1 ? "" : "s"}{" "}
            so far
          </span>
          {passed && <Pill tone="pass">PASS</Pill>}
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

      <section className="space-y-2">
        <SectionLabel>Run history</SectionLabel>
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
            You can pick a different probe to test against the same project.
            Choose how to handle the train.py modifications produced during
            probe-fixing:
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

      {/* ── Auto-research: "run more" input dialog ─────────────────────── */}
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

      {/* ── FAIL: single popup, three side-by-side options ─────────────── */}
      <Modal open={step === "fail"} onClose={() => setStep(null)}>
        <div className="p-7 space-y-5">
          <div className="flex items-center gap-3">
            <Pill tone="fail">FAIL</Pill>
            <h2 className="text-[20px] font-semibold text-ink-950">
              {lastIter?.index != null ? `${ordinal(lastIter.index)} run` : "Run"}{" "}
              did not pass
            </h2>
          </div>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            The probe metric is{" "}
            <span className="font-mono font-medium">
              {lastIter?.metric_value?.toFixed(4) ?? "—"}
            </span>{" "}
            against threshold{" "}
            <span className="font-mono font-medium">
              {lastIter?.threshold ?? "—"}
            </span>
            . What would you like to do?
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
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
              title="Relax threshold"
              body="Back to Implementation so you can adjust the PASS threshold and re-run."
              disabled={reverting}
              busy={reverting}
              onClick={() => performRevert(3, false)}
            />
            <FailChoice
              tone="primary"
              title="Continue probe-fixing"
              body="Stay here and click the action button to run another fixing round."
              disabled={reverting}
              onClick={handleContinueChoice}
            />
          </div>
        </div>
      </Modal>
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
  // Plot one dot per iteration using its `best_value` (running best). If
  // best_value isn't set (older runs / non-auto-research data), fall back to
  // metric_value so the chart still shows something.
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
  const values = rows
    .map((r) => r.metric_value)
    .filter((v): v is number => v != null);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const span = max - min || 1;

  return (
    <div className="rounded-md border border-ink-200 bg-white overflow-hidden">
      <div className="grid grid-cols-[60px_1fr_140px_110px] text-[10px] uppercase tracking-wider text-ink-500 px-3 py-2 border-b border-ink-100 bg-ink-50/50">
        <div>#</div>
        <div>probe metric</div>
        <div>probe value</div>
        <div>probe result</div>
      </div>
      {rows.map((r, i) => {
        const pct = r.metric_value != null ? ((r.metric_value - min) / span) * 100 : 0;
        return (
          <div
            key={i}
            className="grid grid-cols-[60px_1fr_140px_110px] items-center px-3 py-2 border-b border-ink-50 last:border-0 text-[12px]"
          >
            <div className="font-mono text-ink-700">{r.index}</div>
            <div className="text-ink-700 truncate" title={r.metric_name ?? ""}>
              {r.metric_name ?? "—"}
              {r.note && (
                <span className="ml-2 text-[10px] text-ink-400">{r.note}</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1 bg-ink-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-ink-900"
                  style={{ width: `${Math.max(4, pct)}%` }}
                />
              </div>
              <span className="font-mono tabular-nums text-ink-800 w-14 text-right">
                {r.metric_value != null ? r.metric_value.toFixed(4) : "—"}
              </span>
            </div>
            <div>
              {r.status === "PASS" && <Pill tone="pass">PASS</Pill>}
              {r.status === "FAIL" && <Pill tone="fail">FAIL</Pill>}
              {!r.status && <span className="text-ink-400">—</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
