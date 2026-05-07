"use client";

import { useEffect, useState } from "react";
import { api, DevPlan, RunRecord } from "@/lib/api";
import { Button, Card, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";

export function Stage3({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<DevPlan | null>(null);
  const [thresholdInput, setThresholdInput] = useState("");
  const [overriding, setOverriding] = useState(false);

  useEffect(() => {
    if (run.plan_index != null) {
      api
        .getStage2(run.run_id)
        .then((r) => {
          const p =
            r.dev_plans && run.plan_index
              ? r.dev_plans[run.plan_index - 1]
              : null;
          setPlan(p);
          if (p) setThresholdInput(p.threshold);
        })
        .catch(() => {});
    }
  }, [run.run_id, run.plan_index, run.debug_flags.threshold_override]);

  async function handleImplement() {
    setRunning(true);
    setError(null);
    try {
      await api.implement(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setRunning(false);
    }
  }

  async function handleOverride() {
    if (!thresholdInput.trim()) return;
    setOverriding(true);
    setError(null);
    try {
      await api.setThreshold(run.run_id, thresholdInput.trim());
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setOverriding(false);
    }
  }

  const stage3Done = run.stage > 3 || (run.stage === 3 && run.phase === "done");
  const canEdit = run.stage === 3 && !stage3Done && !run.busy;
  const hasFirstRun = run.iterations.length > 0;
  const firstRun = hasFirstRun ? run.iterations[0] : null;

  return (
    <div className="space-y-6">
      <Header
        n={3}
        title="Implementation"
        subtitle="The code agent writes prober.py and integrates it into train.py, then runs training once to validate the integration."
      />

      {/* Selected plan + threshold override */}
      {plan && (
        <section className="space-y-2">
          <SectionLabel>Selected dev plan #{run.plan_index}</SectionLabel>
          <Card className="p-3 space-y-2">
            <div className="text-[12px] text-ink-700 leading-relaxed line-clamp-3">
              {plan.content}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
              <FieldBox label="metric" value={plan.metric} />
              <ThresholdBox
                value={plan.threshold}
                editable={canEdit}
                input={thresholdInput}
                onInput={setThresholdInput}
                onApply={handleOverride}
                applying={overriding}
                overridden={!!run.debug_flags.threshold_override}
              />
            </div>
          </Card>
        </section>
      )}

      {/* Implement */}
      <section className="flex items-center gap-3">
        {!stage3Done && (
          <Button onClick={handleImplement} disabled={running || run.busy}>
            {running ? (
              <>
                <Spinner /> Running agent…
              </>
            ) : (
              "Implement & Run"
            )}
          </Button>
        )}
        {stage3Done && <Pill tone="pass">implementation complete</Pill>}
      </section>

      {/* First training result + live chart */}
      {hasFirstRun && firstRun && (
        <section className="space-y-3">
          <SectionLabel>First training result</SectionLabel>
          <div className="rounded-md border border-ink-200 bg-white p-4">
            <div className="flex items-center gap-3">
              <Pill tone={firstRun.status === "PASS" ? "pass" : "fail"}>
                {firstRun.status ?? "—"}
              </Pill>
              <span className="font-mono text-[12px] text-ink-700">
                {firstRun.metric_name}
              </span>
              <span className="ml-auto font-mono tabular-nums text-[14px] text-ink-950">
                {firstRun.metric_value !== null
                  ? firstRun.metric_value.toFixed(4)
                  : "—"}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-ink-500">
              threshold: <span className="font-mono">{firstRun.threshold ?? "—"}</span>
            </div>
          </div>
          <MetricChart runId={run.run_id} live={running || run.busy} />
        </section>
      )}

      {/* Live chart while training is in flight (before first iteration row exists) */}
      {!hasFirstRun && (running || run.busy) && (
        <section className="space-y-2">
          <SectionLabel>Live metric</SectionLabel>
          <MetricChart runId={run.run_id} live={true} />
        </section>
      )}

      {error && (
        <div className="px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
          {error}
        </div>
      )}
    </div>
  );
}

function FieldBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-ink-200 bg-ink-50/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-500 mb-0.5">
        {label}
      </div>
      <div className="text-[12px] text-ink-800 whitespace-pre-wrap">{value}</div>
    </div>
  );
}

function ThresholdBox({
  value,
  editable,
  input,
  onInput,
  onApply,
  applying,
  overridden,
}: {
  value: string;
  editable: boolean;
  input: string;
  onInput: (v: string) => void;
  onApply: () => void;
  applying: boolean;
  overridden: boolean;
}) {
  const dirty = editable && input !== value;
  return (
    <div className="rounded-md border border-ink-200 bg-ink-50/50 px-3 py-2">
      <div className="flex items-center gap-2 mb-1">
        <div className="text-[10px] uppercase tracking-wider text-ink-500">
          threshold
        </div>
        {overridden && <Pill tone="warn">overridden</Pill>}
      </div>
      {editable ? (
        <div className="flex items-center gap-1.5">
          <input
            value={input}
            onChange={(e) => onInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && dirty) onApply();
            }}
            spellCheck={false}
            className="flex-1 h-7 px-2 rounded border border-ink-200 bg-white font-mono text-[12px] focus:border-ink-400"
            placeholder="e.g. 0.013"
          />
          <Button
            size="sm"
            variant={dirty ? "primary" : "secondary"}
            onClick={onApply}
            disabled={!dirty || applying}
          >
            {applying ? <Spinner /> : "Apply"}
          </Button>
        </div>
      ) : (
        <div className="text-[12px] text-ink-800 font-mono whitespace-pre-wrap">
          {value}
        </div>
      )}
    </div>
  );
}
