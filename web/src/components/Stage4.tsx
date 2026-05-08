"use client";

import { useEffect, useRef, useState } from "react";
import { api, RunRecord } from "@/lib/api";
import { Button, Modal, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";

type PassChoice = "discard" | "keep" | null;

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

  const passed = run.iterations.some((i) => i.status === "PASS");
  const isAutoResearch = run.debug_flags.auto_research;

  // Auto-popup the PASS dialog the first time we see PASS in this session.
  // We track the seen state per run_id so navigating away and back doesn't
  // re-pop, and dismissing keeps it dismissed.
  const [showPass, setShowPass] = useState(false);
  const [confirmChoice, setConfirmChoice] = useState<PassChoice>(null);
  const [reverting, setReverting] = useState(false);
  const seenPassRef = useRef<string | null>(null);

  useEffect(() => {
    if (passed && !isAutoResearch && seenPassRef.current !== run.run_id) {
      seenPassRef.current = run.run_id;
      setShowPass(true);
    }
  }, [passed, isAutoResearch, run.run_id]);

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

  async function performRevert(keep: boolean) {
    setReverting(true);
    setError(null);
    try {
      await api.revert(run.run_id, 1, keep);
      setShowPass(false);
      setConfirmChoice(null);
      if (keep) onKeepRebase?.();
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setReverting(false);
    }
  }

  return (
    <div className="space-y-6">
      <Header
        n={4}
        title="Iteration"
        subtitle={
          isAutoResearch
            ? "Each round, the agent applies one of the 10 # potential_improvement comments seeded in train.py and re-runs training. Auto-research has no PASS gate — keep iterating as long as the metric is still moving."
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
              <Spinner /> Iterating…
            </>
          ) : passed && !isAutoResearch ? (
            "PASSED — done"
          ) : (
            "Run next iteration"
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
        <SectionLabel>Iteration history</SectionLabel>
        {run.iterations.length === 0 ? (
          <div className="rounded-md border border-dashed border-ink-200 px-3 py-6 text-center text-[12px] text-ink-400">
            no iterations yet
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

      {/* PASS choice — primary popup */}
      <Modal
        open={showPass && confirmChoice === null}
        onClose={() => setShowPass(false)}
      >
        <div className="p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Pill tone="pass">PASS</Pill>
            <h2 className="text-[15px] font-semibold text-ink-950">
              Probe passed — try another?
            </h2>
          </div>
          <p className="text-[12.5px] text-ink-700 leading-relaxed">
            You can pick a different probe to test against the same project.
            Choose how to handle the train.py modifications produced during
            iteration:
          </p>
          <div className="space-y-2 pt-1">
            <ChoiceRow
              title="Discard changes & re-pick probe"
              body="Restore train.py to its original baseline, return to stage 1, keep the existing probe candidates. The probe you just passed will be greyed out."
              onClick={() => setConfirmChoice("discard")}
            />
            <ChoiceRow
              title="Keep changes & re-baseline"
              body="Treat the modified train.py as the new starting point. Probe candidates will be regenerated against the updated code for coherence."
              onClick={() => setConfirmChoice("keep")}
            />
          </div>
          <div className="flex justify-end pt-2">
            <Button variant="ghost" onClick={() => setShowPass(false)}>
              Stay on stage 4
            </Button>
          </div>
        </div>
      </Modal>

      {/* Confirm revert (discard path) */}
      <Modal
        open={confirmChoice === "discard"}
        onClose={() => setConfirmChoice(null)}
      >
        <div className="p-5 space-y-3">
          <h2 className="text-[15px] font-semibold text-ink-950">
            Discard changes?
          </h2>
          <p className="text-[12.5px] text-ink-700 leading-relaxed">
            All stage 2/3/4 artifacts will be erased and{" "}
            <span className="font-mono">train.py</span> will be restored to its
            original baseline. The probe you passed will appear as
            <span className="px-1 mx-1 rounded bg-amber-50 text-amber-700 border border-amber-100 text-[11.5px]">
              already tried
            </span>
            so you can pick a different one.
          </p>
          <div className="flex justify-end gap-2 pt-1">
            <Button variant="ghost" onClick={() => setConfirmChoice(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => performRevert(false)}
              disabled={reverting}
            >
              {reverting ? <Spinner /> : "Discard & re-pick"}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Confirm revert (keep path) */}
      <Modal
        open={confirmChoice === "keep"}
        onClose={() => setConfirmChoice(null)}
      >
        <div className="p-5 space-y-3">
          <h2 className="text-[15px] font-semibold text-ink-950">
            Keep changes & re-baseline?
          </h2>
          <p className="text-[12.5px] text-ink-700 leading-relaxed">
            The current <span className="font-mono">train.py</span> becomes the
            new baseline. Probe candidates, dev plans, prober and metrics will
            all be cleared — you'll need to re-generate probe candidates so
            they reflect the updated code.
          </p>
          <div className="flex justify-end gap-2 pt-1">
            <Button variant="ghost" onClick={() => setConfirmChoice(null)}>
              Cancel
            </Button>
            <Button onClick={() => performRevert(true)} disabled={reverting}>
              {reverting ? <Spinner /> : "Keep & re-baseline"}
            </Button>
          </div>
        </div>
      </Modal>

    </div>
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
      className="w-full text-left rounded-md border border-ink-200 px-3 py-2.5 hover:border-ink-400 hover:bg-ink-50 transition-colors"
    >
      <div className="text-[13px] font-medium text-ink-900">{title}</div>
      <div className="mt-0.5 text-[11.5px] text-ink-600 leading-relaxed">
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
      <div className="grid grid-cols-[60px_1fr_140px_80px] text-[10px] uppercase tracking-wider text-ink-500 px-3 py-2 border-b border-ink-100 bg-ink-50/50">
        <div>#</div>
        <div>metric</div>
        <div>value</div>
        <div>status</div>
      </div>
      {rows.map((r, i) => {
        const pct = r.metric_value != null ? ((r.metric_value - min) / span) * 100 : 0;
        return (
          <div
            key={i}
            className="grid grid-cols-[60px_1fr_140px_80px] items-center px-3 py-2 border-b border-ink-50 last:border-0 text-[12px]"
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
