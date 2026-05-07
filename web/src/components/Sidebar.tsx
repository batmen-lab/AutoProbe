"use client";

import { RunRecord, RunSummary } from "@/lib/api";
import { Button, Pill, SectionLabel } from "./ui";

const STAGE_LABELS: Record<number, string> = {
  1: "Probe Design",
  2: "Dev Plan",
  3: "Implementation",
  4: "Iteration",
};

export function Sidebar({
  runs,
  active,
  onPickRun,
  onNewRun,
  onRevert,
  workspaceOpen,
}: {
  runs: RunSummary[];
  active: RunRecord | null;
  onPickRun: (id: string) => void;
  onNewRun: () => void;
  onRevert: (toStage: number) => void;
  workspaceOpen: boolean;
}) {
  return (
    <aside className="w-64 border-r border-ink-200 bg-white flex flex-col">
      {/* Runs list */}
      <div className="px-3 pt-4 pb-2">
        <div className="flex items-center justify-between mb-2">
          <SectionLabel>Runs</SectionLabel>
          <Button
            size="sm"
            variant="ghost"
            disabled={!workspaceOpen}
            onClick={onNewRun}
            title="New run on current workspace"
          >
            + New
          </Button>
        </div>
        <div className="space-y-0.5 max-h-48 overflow-y-auto -mx-1">
          {runs.length === 0 && (
            <div className="px-2 py-1.5 text-[12px] text-ink-400 italic">
              {workspaceOpen ? "no runs yet" : "open a folder first"}
            </div>
          )}
          {runs.map((r) => {
            const isActive = active?.run_id === r.run_id;
            return (
              <button
                key={r.run_id}
                onClick={() => onPickRun(r.run_id)}
                className={`w-full text-left px-2 py-1.5 rounded-md transition-colors group ${
                  isActive ? "bg-ink-900 text-ink-50" : "hover:bg-ink-50 text-ink-700"
                }`}
              >
                <div className="font-mono text-[11px] truncate">{r.run_id}</div>
                <div
                  className={`text-[10px] mt-0.5 ${
                    isActive ? "text-ink-300" : "text-ink-500"
                  }`}
                >
                  stage {r.stage} · {r.phase}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="h-px bg-ink-100 mx-3" />

      {/* Stages */}
      <div className="px-3 pt-3 flex-1 overflow-y-auto">
        <SectionLabel>Stages</SectionLabel>
        <div className="mt-2 space-y-1">
          {[1, 2, 3, 4].map((n) => (
            <StageRow
              key={n}
              n={n}
              label={STAGE_LABELS[n]}
              active={active}
              onRevert={() => onRevert(n)}
            />
          ))}
        </div>
      </div>

      <div className="border-t border-ink-200 px-3 py-2">
        <div className="text-[10px] text-ink-400">
          forward = continue · backward = revert (clean)
        </div>
      </div>
    </aside>
  );
}

function StageRow({
  n,
  label,
  active,
  onRevert,
}: {
  n: number;
  label: string;
  active: RunRecord | null;
  onRevert: () => void;
}) {
  if (!active) {
    return (
      <div className="flex items-center gap-2 h-9 px-2 rounded-md text-ink-400">
        <StageIndex n={n} state="locked" />
        <span className="text-[12px]">{label}</span>
      </div>
    );
  }

  const cur = active.stage;
  let state: "done" | "current" | "future" =
    n < cur ? "done" : n === cur ? "current" : "future";

  // For current stage, treat phase=done as done.
  if (n === cur && active.phase === "done") state = "done";

  const isCurrent = state === "current";
  const isDone = state === "done";

  return (
    <div
      className={`flex items-center gap-2 h-9 px-2 rounded-md group ${
        isCurrent
          ? "bg-ink-100 text-ink-900"
          : isDone
            ? "text-ink-700"
            : "text-ink-400"
      }`}
    >
      <StageIndex n={n} state={state} />
      <span className="text-[12px] flex-1 truncate">{label}</span>
      {isCurrent && <Pill tone="active">current</Pill>}
      {isDone && (
        <button
          onClick={onRevert}
          title={`Revert to stage ${n} (clean start)`}
          className="opacity-0 group-hover:opacity-100 transition-opacity text-[11px] text-ink-500 hover:text-ink-900 px-1.5 py-0.5 rounded hover:bg-white border border-transparent hover:border-ink-200"
        >
          revert
        </button>
      )}
    </div>
  );
}

function StageIndex({
  n,
  state,
}: {
  n: number;
  state: "done" | "current" | "future" | "locked";
}) {
  const cls = {
    done: "bg-ink-900 text-ink-50",
    current: "bg-ink-900 text-ink-50",
    future: "border border-ink-300 text-ink-400",
    locked: "border border-ink-200 text-ink-300",
  }[state];
  const inner = state === "done" ? <Check /> : <span>{n}</span>;
  return (
    <div
      className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono ${cls}`}
    >
      {inner}
    </div>
  );
}

function Check() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
      <path
        d="M2.5 6.5l2.5 2.5 4.5-5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
