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
  | "fail";

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

  const passed = run.iterations.some((i) => i.status === "PASS");
  const isAutoResearch = run.debug_flags.auto_research;
  const lastIter = run.iterations[run.iterations.length - 1];

  // Track which iteration we've already prompted on (so polling/refresh
  // doesn't re-pop the modal). Key: `<run_id>:<iter_index>:<status>`.
  const lastPromptKeyRef = useRef<string | null>(null);

  // Reset per-run UI state when switching runs.
  useEffect(() => {
    setContinueMode(false);
    lastPromptKeyRef.current = null;
  }, [run.run_id]);

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

      <section className="flex items-center gap-3">
        <Button
          onClick={handleIterate}
          disabled={running || run.busy || (passed && !isAutoResearch)}
        >
          {running ? (
            <>
              <Spinner /> Running…
            </>
          ) : passed && !isAutoResearch ? (
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
        {isAutoResearch && <Pill tone="neutral">auto-research</Pill>}
      </section>

      <section className="space-y-2">
        <SectionLabel>Live metric (current run)</SectionLabel>
        <MetricChart runId={run.run_id} live={running || run.busy} />
      </section>

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
