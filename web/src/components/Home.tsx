"use client";

import { useEffect, useState } from "react";
import { api, RunSummary, WorkspaceState } from "@/lib/api";
import { Button, Card, Pill, SectionLabel, Spinner } from "./ui";
import { WorkspacePicker } from "./WorkspaceBar";

/**
 * Entry page. The app *never* auto-loads a workspace or auto-picks a run.
 * The user always has to deliberately:
 *
 *   - click a recent workspace (or browse for a new one), OR
 *   - click a recent run to resume.
 *
 * The previous workspace is remembered and surfaced as the default in the
 * "Recent workspaces" list, but the user still has to click "Open" — it never
 * loads silently behind their back.
 */
export function Home({
  onOpenWorkspace,
  onResumeRun,
}: {
  onOpenWorkspace: (path: string) => Promise<void>;
  onResumeRun: (run: RunSummary) => Promise<void>;
}) {
  const [workspace, setWorkspace] = useState<WorkspaceState>({
    current: null,
    recent: [],
  });
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [picker, setPicker] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getWorkspace(), api.listRuns()])
      .then(([ws, r]) => {
        setWorkspace(ws);
        setRuns(r.runs);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  async function handleOpen(path: string) {
    setBusyAction(`open:${path}`);
    try {
      await onOpenWorkspace(path);
    } finally {
      setBusyAction(null);
    }
  }

  async function handleResume(run: RunSummary) {
    setBusyAction(`run:${run.run_id}`);
    try {
      await onResumeRun(run);
    } finally {
      setBusyAction(null);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-ink-500 text-[13px] py-12 justify-center">
        <Spinner /> loading…
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-8 py-12">
      <header className="mb-10">
        <h1 className="text-[28px] font-semibold tracking-tight text-ink-950">
          Agentic Probe
        </h1>
        <p className="mt-2 text-[14px] text-ink-600">
          A 4-stage probe pipeline with forward and backward navigation. Pick a
          project folder to instrument, or resume a previous run.
        </p>
      </header>

      {/* Open a folder */}
      <section className="mb-10">
        <div className="flex items-center justify-between mb-3">
          <SectionLabel>Open a project folder</SectionLabel>
          <Button size="sm" variant="secondary" onClick={() => setPicker(true)}>
            Browse…
          </Button>
        </div>
        {workspace.recent.length === 0 ? (
          <Card className="p-4 text-[12.5px] text-ink-500 italic">
            No recent workspaces. Click <b>Browse…</b> to pick a folder.
          </Card>
        ) : (
          <div className="space-y-2">
            {workspace.recent.map((path, i) => {
              const isDefault = i === 0;
              const busy = busyAction === `open:${path}`;
              return (
                <Card key={path} className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="font-mono text-[12.5px] text-ink-800 truncate">
                        {path}
                      </div>
                      {isDefault && (
                        <div className="mt-0.5">
                          <Pill tone="neutral">last used</Pill>
                        </div>
                      )}
                    </div>
                    <Button
                      size="sm"
                      onClick={() => handleOpen(path)}
                      disabled={busyAction !== null}
                    >
                      {busy ? <Spinner /> : "Open"}
                    </Button>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </section>

      {/* Resume a run */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <SectionLabel>Resume a run</SectionLabel>
          <span className="text-[11px] text-ink-500">
            {runs.length} total
          </span>
        </div>
        {runs.length === 0 ? (
          <Card className="p-4 text-[12.5px] text-ink-500 italic">
            No previous runs. Open a folder above and create your first run.
          </Card>
        ) : (
          <div className="space-y-2">
            {runs.slice(0, 8).map((r) => {
              const busy = busyAction === `run:${r.run_id}`;
              return (
                <Card
                  key={r.run_id}
                  className="p-3"
                  onClick={busyAction ? undefined : () => handleResume(r)}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[12px] text-ink-900">
                          {r.run_id}
                        </span>
                        <Pill tone="neutral">stage {r.stage}</Pill>
                        <span className="text-[11px] text-ink-500">
                          {r.phase}
                        </span>
                      </div>
                      <div className="mt-1 font-mono text-[11px] text-ink-500 truncate">
                        {r.workspace}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={busyAction !== null}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleResume(r);
                      }}
                    >
                      {busy ? <Spinner /> : "Resume"}
                    </Button>
                  </div>
                </Card>
              );
            })}
            {runs.length > 8 && (
              <div className="text-[11px] text-ink-400 text-center pt-2">
                showing 8 of {runs.length}
              </div>
            )}
          </div>
        )}
      </section>

      {picker && (
        <WorkspacePicker
          state={workspace}
          onClose={() => setPicker(false)}
          onPicked={async (s) => {
            setPicker(false);
            if (s.current) {
              await handleOpen(s.current);
            }
          }}
        />
      )}
    </div>
  );
}
