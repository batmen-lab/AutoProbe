"use client";

import { useCallback, useEffect, useState } from "react";
import { api, RunRecord, RunSummary, WorkspaceState } from "@/lib/api";
import { WorkspaceBar } from "@/components/WorkspaceBar";
import { Sidebar } from "@/components/Sidebar";
import { Stage1 } from "@/components/Stage1";
import { Stage2 } from "@/components/Stage2";
import { Stage3 } from "@/components/Stage3";
import { Stage4 } from "@/components/Stage4";
import { Home } from "@/components/Home";
import { GlobalLogDock } from "@/components/LogPanel";
import { Button } from "@/components/ui";

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
      const r = await api.getRun(active.run_id);
      setActive(r);
      if (workspace.current) {
        const { runs } = await api.listRuns(workspace.current);
        setRuns(runs);
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }, [active, workspace.current]);

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
      const r = await api.getRun(id);
      setActive(r);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  async function handleNewRun() {
    try {
      const r = await api.newRun();
      const { runs } = await api.listRuns(workspace.current ?? undefined);
      setRuns(runs);
      setActive(r);
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
    const ok = window.confirm(
      `Revert run ${active.run_id} to stage ${toStage}?\n\n` +
        `This will erase stage ${toStage}'s outputs and any later stage artifacts. Earlier-stage inputs are kept.`,
    );
    if (!ok) return;
    try {
      const { state } = await api.revert(active.run_id, toStage);
      setActive(state);
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
              {!active ? (
                <EmptyState onNewRun={handleNewRun} onHome={handleHome} />
              ) : (
                <ActiveStage run={active} onUpdate={refreshActive} />
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
    </div>
  );
}

function ActiveStage({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  switch (run.stage) {
    case 1:
      return <Stage1 run={run} onUpdate={onUpdate} />;
    case 2:
      return <Stage2 run={run} onUpdate={onUpdate} />;
    case 3:
      return <Stage3 run={run} onUpdate={onUpdate} />;
    case 4:
      return <Stage4 run={run} onUpdate={onUpdate} />;
    default:
      return <div>unknown stage</div>;
  }
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
