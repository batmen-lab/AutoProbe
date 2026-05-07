"use client";

import { useEffect, useState } from "react";
import { api, ProbeDesign, RunRecord } from "@/lib/api";
import { Button, Card, ConfidenceBar, Pill, SectionLabel, Spinner } from "./ui";

export function Stage1({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [context, setContext] = useState(run.context ?? "");
  const [designs, setDesigns] = useState<ProbeDesign[] | null>(null);
  const [autoResearch, setAutoResearch] = useState(
    run.debug_flags.auto_research,
  );
  const [generating, setGenerating] = useState(false);
  const [picking, setPicking] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setContext(run.context ?? "");
    setAutoResearch(run.debug_flags.auto_research);
    if (run.stage >= 1 && (run.phase === "generated" || run.stage > 1)) {
      api
        .getStage1(run.run_id)
        .then((r) => setDesigns(r.probe_designs ?? null))
        .catch(() => {});
    } else {
      setDesigns(null);
    }
  }, [run.run_id, run.stage, run.phase, run.context, run.debug_flags.auto_research]);

  async function handleGenerate() {
    setError(null);
    setGenerating(true);
    try {
      await api.setContext(run.run_id, context);
      await api.generateProbes(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setGenerating(false);
    }
  }

  async function handleAutoResearch() {
    setError(null);
    setGenerating(true);
    try {
      await api.autoResearch(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setGenerating(false);
    }
  }

  async function handleSelect(idx: number) {
    setPicking(idx);
    setError(null);
    try {
      await api.selectProbe(run.run_id, idx);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setPicking(null);
    }
  }

  const selected = run.probe_index;
  const stage1Active = run.stage === 1;

  return (
    <div className="space-y-6">
      <Header
        n={1}
        title="Probe Design"
        subtitle="Describe your project and the model proposes probe designs with self-rated confidence."
      />

      {/* Mode toggle (only at stage 1, before any work) */}
      {stage1Active && !designs && (
        <ModeToggle
          autoResearch={autoResearch}
          onChange={setAutoResearch}
          disabled={generating}
        />
      )}

      {/* Auto-research path */}
      {stage1Active && autoResearch && !designs && (
        <Card className="p-4">
          <div className="text-[12.5px] text-ink-700 leading-relaxed">
            The agent will read <span className="font-mono">train.py</span>,
            pick a standard performance metric appropriate for your task, write{" "}
            <span className="font-mono">prober.py</span>, integrate it into{" "}
            <span className="font-mono">train.py</span>, and run training to
            produce the first metric. There is no probe selection or dev plan
            in this mode — you'll jump straight to stage 4 (iteration).
          </div>
          <div className="mt-4">
            <Button onClick={handleAutoResearch} disabled={generating}>
              {generating ? (
                <>
                  <Spinner /> Running setup…
                </>
              ) : (
                "Run Auto-Research Setup"
              )}
            </Button>
          </div>
        </Card>
      )}

      {/* Regular path: context + generate */}
      {(!autoResearch || !stage1Active) && (
        <section>
          <SectionLabel>Project context</SectionLabel>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            rows={4}
            className="mt-2 w-full rounded-md border border-ink-200 bg-white px-3 py-2 text-[13px] focus:border-ink-400"
            placeholder="One or two sentences about the project + dataset description."
            disabled={!stage1Active || autoResearch}
          />
          <div className="mt-2 flex items-center gap-2">
            {stage1Active && !autoResearch && (
              <Button
                onClick={handleGenerate}
                disabled={generating || !context.trim()}
              >
                {generating ? (
                  <>
                    <Spinner /> Generating…
                  </>
                ) : designs ? (
                  "Regenerate"
                ) : (
                  "Generate Probes"
                )}
              </Button>
            )}
            {!stage1Active && (
              <Pill tone="pass">
                {run.debug_flags.auto_research
                  ? "auto-research mode"
                  : `probe #${run.probe_index} selected`}{" "}
                · stage {run.stage}
              </Pill>
            )}
          </div>
        </section>
      )}

      {/* Designs list (only in regular mode) */}
      {designs && !run.debug_flags.auto_research && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionLabel>Probe candidates ({designs.length})</SectionLabel>
            <div className="text-[11px] text-ink-500">
              {stage1Active ? "select one to continue" : "selection locked"}
            </div>
          </div>
          {designs.map((d, i) => {
            const idx = i + 1;
            const isSel = selected === idx;
            return (
              <Card key={i} selected={isSel} className="p-4">
                <div className="flex items-start gap-3">
                  <div className="font-mono text-[11px] text-ink-500 w-7 mt-0.5">
                    {String(idx).padStart(2, "0")}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Pill tone="neutral">{d.probe_type}</Pill>
                      <span className="font-medium text-[13px] text-ink-900">
                        {d.probe_name}
                      </span>
                      <div className="ml-auto">
                        <ConfidenceBar value={d.confidence} />
                      </div>
                    </div>
                    <p className="mt-2 text-[12.5px] text-ink-700 leading-relaxed">
                      {d.content}
                    </p>
                    {d.possible_sources?.length > 0 && (
                      <div className="mt-2 text-[11px] text-ink-500">
                        <span className="uppercase tracking-wide font-medium">
                          sources:
                        </span>{" "}
                        {d.possible_sources.join(" · ")}
                      </div>
                    )}
                    {stage1Active && (
                      <div className="mt-3">
                        <Button
                          size="sm"
                          variant={isSel ? "primary" : "secondary"}
                          onClick={() => handleSelect(idx)}
                          disabled={picking !== null}
                        >
                          {picking === idx ? <Spinner /> : isSel ? "Selected" : "Select & Continue"}
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

function ModeToggle({
  autoResearch,
  onChange,
  disabled,
}: {
  autoResearch: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <button
          role="switch"
          aria-checked={autoResearch}
          disabled={disabled}
          onClick={() => onChange(!autoResearch)}
          className={`mt-0.5 w-9 h-5 rounded-full transition-colors flex-shrink-0 relative ${
            autoResearch ? "bg-ink-900" : "bg-ink-200"
          } ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
        >
          <span
            className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${
              autoResearch ? "left-[18px]" : "left-0.5"
            }`}
          />
        </button>
        <div className="flex-1">
          <div className="text-[13px] font-medium text-ink-900">
            Auto-research mode
          </div>
          <p className="mt-0.5 text-[12px] text-ink-600 leading-relaxed">
            Skip probe design and dev plan. The agent picks a standard
            performance metric for your task and goes straight from train.py
            inspection → prober + integration → iteration. No threshold, no
            PASS/FAIL — just push the metric round after round.
          </p>
        </div>
      </div>
    </Card>
  );
}

export function Header({
  n,
  title,
  subtitle,
}: {
  n: number;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="border-b border-ink-200 pb-4">
      <div className="flex items-baseline gap-3">
        <div className="font-mono text-[11px] text-ink-500">STAGE {n}</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink-950">{title}</h1>
      </div>
      {subtitle && (
        <p className="mt-1.5 text-[13px] text-ink-600">{subtitle}</p>
      )}
    </div>
  );
}
