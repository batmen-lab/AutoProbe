"use client";

import { useEffect, useState } from "react";
import { api, DevPlan, ProbeDesign, RunRecord } from "@/lib/api";
import {
  Button,
  Card,
  ConfidenceBar,
  Pill,
  SectionLabel,
  Spinner,
  stageName,
} from "./ui";
import { Header } from "./Stage1";
import { useActionLatch } from "@/lib/useActionLatch";

export function Stage2({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [plans, setPlans] = useState<DevPlan[] | null>(null);
  const [probe, setProbe] = useState<ProbeDesign | null>(null);
  const { inProgress: generating, begin, fail } = useActionLatch(run.busy);
  const [picking, setPicking] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (run.stage >= 2 && (run.phase === "generated" || run.stage > 2)) {
      api
        .getStage2(run.run_id)
        .then((r) => setPlans(r.dev_plans ?? null))
        .catch(() => {});
    } else {
      setPlans(null);
    }
  }, [run.run_id, run.stage, run.phase]);

  // Look up the probe the user chose in stage 1 so they can refresh their
  // memory of what these dev plans are being generated against.
  useEffect(() => {
    if (run.probe_index == null) {
      setProbe(null);
      return;
    }
    api
      .getStage1(run.run_id)
      .then((r) => {
        const p =
          r.probe_designs && run.probe_index
            ? r.probe_designs[run.probe_index - 1]
            : null;
        setProbe(p ?? null);
      })
      .catch(() => setProbe(null));
  }, [run.run_id, run.probe_index]);

  async function handleGenerate() {
    setError(null);
    begin();
    try {
      await api.generateDevPlans(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
      fail();
    }
  }

  async function handleSelect(idx: number) {
    setPicking(idx);
    setError(null);
    try {
      await api.selectPlan(run.run_id, idx);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setPicking(null);
    }
  }

  const selected = run.plan_index;

  return (
    <div className="space-y-6">
      <Header
        n={2}
        title="Dev Plan"
        subtitle="The model translates your selected probe into 3 concrete development plans, each rated for confidence."
      />

      {probe && (
        <section className="space-y-2">
          <SectionLabel>
            Probe you chose (#{run.probe_index})
          </SectionLabel>
          <Card className="p-3 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Pill tone="neutral">{probe.probe_type}</Pill>
              <span className="font-medium text-[13px] text-ink-900">
                {probe.probe_name}
              </span>
              <div className="ml-auto">
                <ConfidenceBar value={probe.confidence} />
              </div>
            </div>
            <p className="text-[12px] text-ink-700 leading-relaxed">
              {probe.content}
            </p>
            {probe.possible_sources?.length > 0 && (
              <div className="text-[11px] text-ink-500">
                <span className="uppercase tracking-wide font-medium">
                  sources:
                </span>{" "}
                {probe.possible_sources.join(" · ")}
              </div>
            )}
          </Card>
        </section>
      )}

      {run.stage === 2 && run.phase === "input" && (
        <div>
          <Button onClick={handleGenerate} disabled={generating}>
            {generating ? (
              <>
                <Spinner /> Generating…
              </>
            ) : (
              "Generate Dev Plans"
            )}
          </Button>
        </div>
      )}

      {run.stage === 2 && run.phase === "generated" && plans && (
        <div className="text-[11px] text-ink-500">
          select one to advance to Implementation
        </div>
      )}

      {run.stage > 2 && (
        <Pill tone="pass">
          plan #{run.plan_index} selected · {stageName(run.stage)}
        </Pill>
      )}

      {plans && (
        <section className="space-y-3">
          {plans.map((p, i) => {
            const idx = i + 1;
            const isSel = selected === idx;
            const isTried = run.tried_plan_indices?.includes(idx);
            return (
              <Card
                key={i}
                selected={isSel}
                className={`p-4 ${isTried ? "opacity-55 bg-ink-50/60" : ""}`}
              >
                <div className="flex items-start gap-3">
                  <div className="font-mono text-[11px] text-ink-500 w-7 mt-0.5">
                    {String(idx).padStart(2, "0")}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <Pill tone="neutral">plan</Pill>
                      {isTried && <Pill tone="warn">already tried</Pill>}
                      <div className="ml-auto">
                        <ConfidenceBar value={p.confidence} />
                      </div>
                    </div>
                    <p className="text-[12.5px] text-ink-700 leading-relaxed whitespace-pre-wrap">
                      {p.content}
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
                      <FieldBox label="metric" value={p.metric} />
                      <FieldBox
                        label="standard threshold"
                        value={p.standard_threshold}
                        mono
                      />
                      <FieldBox
                        label="acceptable threshold"
                        value={p.acceptable_threshold}
                        mono
                      />
                    </div>
                    {run.stage === 2 && (
                      <div className="pt-1">
                        <Button
                          size="sm"
                          variant={isSel ? "primary" : "secondary"}
                          onClick={() => handleSelect(idx)}
                          disabled={picking !== null || isTried}
                          title={
                            isTried
                              ? "Already tried in this run — pick a different plan."
                              : undefined
                          }
                        >
                          {picking === idx ? (
                            <Spinner />
                          ) : isTried ? (
                            "Already tried"
                          ) : isSel ? (
                            "Selected"
                          ) : (
                            "Select & Continue"
                          )}
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
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
