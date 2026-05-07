"use client";

import { useState } from "react";
import { api, RunRecord } from "@/lib/api";
import { Button, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";

export function Stage4({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const passed = run.iterations.some((i) => i.status === "PASS");
  const isAutoResearch = run.debug_flags.auto_research;

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
