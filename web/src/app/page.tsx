"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, RunRecord, RunSummary, WorkspaceState } from "@/lib/api";
import { WorkspaceBar } from "@/components/WorkspaceBar";
import { Sidebar } from "@/components/Sidebar";
import { Stage1 } from "@/components/Stage1";
import { Stage2 } from "@/components/Stage2";
import { Stage3 } from "@/components/Stage3";
import { Stage4 } from "@/components/Stage4";
import { Home } from "@/components/Home";
import { GlobalLogDock } from "@/components/LogPanel";
import {
  ActionStatusBar,
  Button,
  Modal,
  stageName,
  Toast,
  ToastSpec,
} from "@/components/ui";

type View = "home" | "session";

export default function Page() {
  const [view, setView] = useState<View>("home");
  const [workspace, setWorkspace] = useState<WorkspaceState>({
    current: null,
    recent: [],
  });
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [active, setActive] = useState<RunRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastSpec | null>(null);
  const [keepInfoOpen, setKeepInfoOpen] = useState(false);
  // Last seen current_action — used to detect transitions and fire toasts.
  const prevActionRef = useRef<string | null>(null);
  // Monotonic sequence so a slow polling GET (sent pre-revert, returning
  // post-revert) can't overwrite a fresher setActive from the revert/POST
  // response. Each setActive bumps lastAppliedSeq; polling drops responses
  // whose seq is older than the latest applied.
  const reqSeqRef = useRef(0);
  const lastAppliedSeqRef = useRef(0);
  const applyState = useCallback((s: RunRecord, seq: number) => {
    if (seq < lastAppliedSeqRef.current) return; // stale, drop
    lastAppliedSeqRef.current = seq;
    setActive(s);
  }, []);

  // We do NOT auto-load workspace or active run on mount.
  // The home page is always the entry; the user must click to proceed.

  // ── home actions ────────────────────────────────────────────────────────
  async function openWorkspaceFromHome(path: string) {
    try {
      const ws = await api.openWorkspace(path);
      const { runs } = await api.listRuns(ws.current ?? undefined);
      setWorkspace(ws);
      setRuns(runs);
      setActive(null);
      setView("session");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  async function resumeRunFromHome(run: RunSummary) {
    try {
      const ws = await api.openWorkspace(run.workspace);
      const { runs } = await api.listRuns(run.workspace);
      const r = await api.getRun(run.run_id);
      setWorkspace(ws);
      setRuns(runs);
      setActive(r);
      setView("session");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  // ── refresh active run + runs list ──────────────────────────────────────
  const refreshActive = useCallback(async () => {
    if (!active) return;
    try {
      const seq = ++reqSeqRef.current;
      const r = await api.getRun(active.run_id);
      applyState(r, seq);
      if (workspace.current) {
        const { runs } = await api.listRuns(workspace.current);
        setRuns(runs);
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }, [active, workspace.current, applyState]);

  // ── poll the active run so the status bar picks up intermediate actions ─
  // We poll unconditionally (not just when busy=true), because long-running
  // stage POSTs hold the request open for minutes — `busy` flips on the
  // server the moment the lock is acquired, but the frontend would never
  // see it without polling, and a busy-gated effect would never start.
  useEffect(() => {
    if (!active?.run_id) return;
    const runId = active.run_id;
    let cancelled = false;
    const tick = async () => {
      const seq = ++reqSeqRef.current;
      try {
        const r = await api.getRun(runId);
        if (!cancelled) applyState(r, seq);
      } catch {
        // ignore transient polling errors
      }
    };
    const id = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [active?.run_id, applyState]);

  // ── toast on probe-generate / dev-plan-generate completion ──────────────
  useEffect(() => {
    const prev = prevActionRef.current;
    const cur = active?.current_action ?? null;
    if (prev && !cur) {
      // Transitioned from in-flight to idle — fire the right toast.
      if (prev === "probe-generate") {
        setToast({
          id: Date.now(),
          tone: "success",
          text: "Probe candidates ready — pick one to continue.",
        });
      } else if (prev === "dev-plan-generate") {
        setToast({
          id: Date.now(),
          tone: "success",
          text: "Dev plans ready — pick one to continue.",
        });
      }
    }
    prevActionRef.current = cur;
  }, [active?.current_action]);

  // ── session actions ─────────────────────────────────────────────────────
  async function handleWorkspaceChange(s: WorkspaceState) {
    setWorkspace(s);
    setActive(null);
    setRuns([]);
    if (s.current) {
      try {
        const { runs } = await api.listRuns(s.current);
        setRuns(runs);
      } catch (e) {
        setError(String((e as Error).message ?? e));
      }
    }
  }

  async function handlePickRun(id: string) {
    try {
      const seq = ++reqSeqRef.current;
      const r = await api.getRun(id);
      applyState(r, seq);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  async function handleNewRun() {
    try {
      const seq = ++reqSeqRef.current;
      const r = await api.newRun();
      const { runs } = await api.listRuns(workspace.current ?? undefined);
      setRuns(runs);
      applyState(r, seq);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  async function handleHome() {
    if (active && active.busy) {
      const ok = window.confirm(
        `Run ${active.run_id} has an in-flight action.\n\n` +
          `Going home will KILL the running subprocess and reset this run's phase. Continue?`,
      );
      if (!ok) return;
      try {
        await api.cancel();
      } catch (e) {
        setError(String((e as Error).message ?? e));
        return;
      }
    }
    setActive(null);
    setRuns([]);
    setView("home");
    setError(null);
  }

  async function handleRevert(toStage: number) {
    if (!active) return;
    const target = stageName(toStage);
    const message =
      toStage === 1
        ? `Back to ${target} for run ${active.run_id}?\n\n` +
          `Probe candidates are kept; your selection will be cleared so you can re-pick. Dev Plan, Implementation and Probe Fixing artifacts are erased. ` +
          `(To regenerate the probe candidates themselves, use the Regenerate button inside Probe Design.)`
        : toStage === 2
          ? `Back to ${target} for run ${active.run_id}?\n\n` +
            `Dev plans are kept; your plan selection will be cleared so you can re-pick. Implementation and Probe Fixing artifacts are erased.`
          : `Back to ${target} for run ${active.run_id}?\n\n` +
            `This will erase ${target}'s outputs and any later-stage artifacts. Earlier inputs are kept.`;
    const ok = window.confirm(message);
    if (!ok) return;
    try {
      const seq = ++reqSeqRef.current;
      const { state } = await api.revert(active.run_id, toStage);
      applyState(state, seq);
      if (workspace.current) {
        const { runs } = await api.listRuns(workspace.current);
        setRuns(runs);
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  // ── render ──────────────────────────────────────────────────────────────
  if (view === "home") {
    return (
      <div className="min-h-screen bg-ink-50">
        <div className="h-12 border-b border-ink-200 bg-white px-4 flex items-center">
          <div className="font-semibold tracking-tight text-[14px] text-ink-900">
            Agentic Probe
          </div>
        </div>
        <Home
          onOpenWorkspace={openWorkspaceFromHome}
          onResumeRun={resumeRunFromHome}
        />
        {error && (
          <div className="max-w-2xl mx-auto px-8 -mt-6 mb-8">
            <div className="px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
              {error}
              <button
                className="ml-2 text-ink-500 hover:text-ink-900"
                onClick={() => setError(null)}
              >
                dismiss
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-ink-50">
      <WorkspaceBar
        state={workspace}
        onChange={handleWorkspaceChange}
        onHome={handleHome}
        homeDisabled={false}
      />
      <div className="flex-1 flex overflow-hidden min-h-0">
        <Sidebar
          runs={runs}
          active={active}
          onPickRun={handlePickRun}
          onNewRun={handleNewRun}
          onRevert={handleRevert}
          workspaceOpen={!!workspace.current}
        />
        <div className="flex-1 flex flex-col min-w-0">
          <main className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-8 py-8">
              {active && (
                <div className="mb-4">
                  <ActionStatusBar
                    action={active.current_action}
                    busy={active.busy}
                    idleMessage={idleMessageFor(active)}
                  />
                </div>
              )}
              {!active ? (
                <EmptyState onNewRun={handleNewRun} onHome={handleHome} />
              ) : (
                <ActiveStage
                  run={active}
                  onUpdate={refreshActive}
                  onKeepRebase={() => setKeepInfoOpen(true)}
                />
              )}

              {error && (
                <div className="mt-6 px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
                  {error}
                  <button
                    className="ml-2 text-ink-500 hover:text-ink-900"
                    onClick={() => setError(null)}
                  >
                    dismiss
                  </button>
                </div>
              )}
            </div>
          </main>
          <GlobalLogDock
            runId={active?.run_id ?? null}
            live={active?.busy ?? false}
          />
        </div>
      </div>

      <Toast toast={toast} onClose={() => setToast(null)} />

      <Modal open={keepInfoOpen} onClose={() => setKeepInfoOpen(false)}>
        <div className="p-7 space-y-5">
          <h2 className="text-[20px] font-semibold text-ink-950">
            Re-generation needed
          </h2>
          <p className="text-[15px] text-ink-700 leading-relaxed">
            <span className="font-mono">train.py</span> has been re-baselined.
            The previous probe candidates would be incoherent with the updated
            code, so we cleared them. Click <b>Generate Probes</b> on Probe
            Design (your project context is preserved) to produce fresh
            candidates.
          </p>
          <div className="flex justify-end pt-1">
            <Button onClick={() => setKeepInfoOpen(false)}>Got it</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function ActiveStage({
  run,
  onUpdate,
  onKeepRebase,
}: {
  run: RunRecord;
  onUpdate: () => void;
  onKeepRebase: () => void;
}) {
  switch (run.stage) {
    case 1:
      return <Stage1 run={run} onUpdate={onUpdate} />;
    case 2:
      return <Stage2 run={run} onUpdate={onUpdate} />;
    case 3:
      return <Stage3 run={run} onUpdate={onUpdate} />;
    case 4:
      return (
        <Stage4 run={run} onUpdate={onUpdate} onKeepRebase={onKeepRebase} />
      );
    default:
      return <div>unknown stage</div>;
  }
}

// Surface a phase-specific "what's expected next" message when the pipeline
// is idle, so the user always knows where input is needed.
function idleMessageFor(run: RunRecord): string | null {
  if (run.busy || run.current_action) return null;
  const { stage, phase, probe_index, plan_index } = run;
  if (stage === 1 && phase === "input") {
    return "Waiting for project context — describe your project, then click Generate Probes.";
  }
  if (stage === 1 && phase === "generated" && probe_index == null) {
    return "Waiting for probe selection — pick one of the candidates below to continue.";
  }
  if (stage === 2 && phase === "input") {
    return "Waiting to generate dev plans — click Generate Dev Plans to proceed.";
  }
  if (stage === 2 && phase === "generated" && plan_index == null) {
    return "Waiting for dev plan selection — pick one of the candidates below to continue.";
  }
  if (stage === 3) {
    if (phase === "ready") {
      // We landed here from Probe Fixing → "Relax threshold". The whole
      // point is to edit the threshold; nudge the user toward that.
      return "Waiting for you to modify the threshold — adjust it on the right and click Apply, then Implement & Run.";
    }
    return "Waiting to start implementation — click Implement & Run when you're ready.";
  }
  if (stage === 4) {
    if (run.debug_flags.auto_research) {
      const target = run.auto_research_target_runs ?? 0;
      const completed = run.auto_research_runs_completed ?? 0;
      if (target > 0 && completed >= target) {
        return `Auto-research batch complete (${completed}/${target}) — run more rounds or go back to Probe Design.`;
      }
      return "Auto-research ready — set the number of rounds and click Start auto-research.";
    }
    return "Waiting for next action — start auto probe-fixing, go back, or stop.";
  }
  return null;
}

function EmptyState({
  onNewRun,
  onHome,
}: {
  onNewRun: () => void;
  onHome: () => void;
}) {
  return (
    <div className="py-16 text-center">
      <div className="text-[18px] font-semibold text-ink-900">
        No run selected
      </div>
      <div className="mt-2 text-[13px] text-ink-600 max-w-md mx-auto">
        Pick an existing run from the sidebar or start a fresh one on this
        workspace.
      </div>
      <div className="mt-6 flex items-center justify-center gap-2">
        <Button onClick={onNewRun}>+ New run</Button>
        <Button variant="ghost" onClick={onHome}>
          ← Home
        </Button>
      </div>
    </div>
  );
}
