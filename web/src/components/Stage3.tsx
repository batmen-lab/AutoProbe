"use client";

import { useEffect, useState } from "react";
import { api, DevPlan, RunRecord } from "@/lib/api";
import { Button, Card, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { MetricChart } from "./MetricChart";
import { useActionLatch } from "@/lib/useActionLatch";

export function Stage3({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const { inProgress: running, begin, fail } = useActionLatch(run.busy);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<DevPlan | null>(null);

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
        })
        .catch(() => {});
    }
  }, [run.run_id, run.plan_index]);

  async function handleImplement() {
    begin();
    setError(null);
    try {
      await api.implement(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    }
  }

  const stage3Done = run.stage > 3 || (run.stage === 3 && run.phase === "done");
  const hasFirstRun = run.iterations.length > 0;
  const firstRun = hasFirstRun ? run.iterations[0] : null;

  return (
    <div className="space-y-6">
      <Header
        n={3}
        title="Implementation"
        subtitle="The code agent writes prober.py and integrates it into train.py, then runs training once to validate the integration."
      />

      {/* Selected plan + both thresholds (read-only) */}
      {plan && (
        <section className="space-y-2">
          <SectionLabel>Selected dev plan #{run.plan_index}</SectionLabel>
          <Card className="p-3 space-y-2">
            <div className="text-[12px] text-ink-700 leading-relaxed line-clamp-3">
              {plan.content}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
              <FieldBox label="metric" value={plan.metric} />
              <FieldBox
                label="standard threshold"
                value={plan.standard_threshold}
                mono
              />
              <FieldBox
                label="acceptable threshold"
                value={plan.acceptable_threshold}
                mono
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
            <div className="mt-2 text-[11px] text-ink-500 flex flex-wrap gap-x-4 gap-y-1">
              <span>
                standard:{" "}
                <span className="font-mono">{firstRun.threshold ?? "—"}</span>
              </span>
              <span>
                acceptable:{" "}
                <span className="font-mono">
                  {firstRun.acceptable_threshold ?? "—"}
                </span>
              </span>
              {firstRun.acceptable_met != null && (
                <span>
                  acceptable met:{" "}
                  <span className="font-mono">
                    {firstRun.acceptable_met ? "yes" : "no"}
                  </span>
                </span>
              )}
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

function FieldBox({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-md border border-ink-200 bg-ink-50/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-500 mb-0.5">
        {label}
      </div>
      <div
        className={`text-[12px] text-ink-800 ${mono ? "font-mono" : ""} whitespace-pre-wrap`}
      >
        {value}
      </div>
    </div>
  );
}
