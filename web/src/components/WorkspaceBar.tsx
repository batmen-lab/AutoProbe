"use client";

import { useEffect, useState } from "react";
import { api, BrowseResult, WorkspaceState } from "@/lib/api";
import { Button, SectionLabel } from "./ui";

export function WorkspaceBar({
  state,
  onChange,
  onHome,
  homeDisabled,
}: {
  state: WorkspaceState;
  onChange: (s: WorkspaceState) => void;
  onHome: () => void;
  homeDisabled: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="h-12 border-b border-ink-200 bg-white px-4 flex items-center gap-3">
      <button
        onClick={onHome}
        disabled={homeDisabled}
        title="Back to home"
        className="font-semibold tracking-tight text-[14px] text-ink-900 hover:text-ink-600 disabled:text-ink-400 disabled:cursor-default transition-colors"
      >
        Agentic Probe
      </button>
      <div className="h-5 w-px bg-ink-200" />
      <SectionLabel>Workspace</SectionLabel>
      <div className="font-mono text-[12px] text-ink-700 truncate max-w-[480px]">
        {state.current ?? <span className="text-ink-400 italic">no folder open</span>}
      </div>
      <div className="ml-auto flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onHome}
          disabled={homeDisabled}
        >
          ← Home
        </Button>
        <Button variant="secondary" size="sm" onClick={() => setOpen(true)}>
          Open Folder…
        </Button>
      </div>
      {open && (
        <WorkspacePicker
          state={state}
          onClose={() => setOpen(false)}
          onPicked={(s) => {
            onChange(s);
            setOpen(false);
          }}
        />
      )}
    </div>
  );
}

export function WorkspacePicker({
  state,
  onClose,
  onPicked,
}: {
  state: WorkspaceState;
  onClose: () => void;
  onPicked: (s: WorkspaceState) => void;
}) {
  const [browse, setBrowse] = useState<BrowseResult | null>(null);
  const [path, setPath] = useState(state.current ?? "");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initial = state.current || state.recent[0] || "/home";
    api
      .browse(initial)
      .then(setBrowse)
      .catch(() => api.browse("/home").then(setBrowse).catch(() => {}));
  }, [state.current, state.recent]);

  async function pick(p: string) {
    try {
      const s = await api.openWorkspace(p);
      onPicked(s);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  async function go(p: string) {
    try {
      const r = await api.browse(p);
      setBrowse(r);
      setPath(r.path);
      setError(null);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-ink-950/20 backdrop-blur-[2px] flex items-start justify-center pt-24 px-4">
      <div className="bg-white rounded-lg shadow-card border border-ink-200 w-[640px] max-w-full overflow-hidden">
        <div className="px-4 py-3 border-b border-ink-200 flex items-center gap-2">
          <div className="font-medium text-[13px]">Open Folder</div>
          <div className="ml-auto text-[11px] text-ink-500">
            requires <span className="font-mono">train.py</span>
          </div>
        </div>

        {state.recent.length > 0 && (
          <div className="px-4 py-3 border-b border-ink-200">
            <SectionLabel>Recent</SectionLabel>
            <div className="mt-2 space-y-1">
              {state.recent.map((w) => (
                <button
                  key={w}
                  onClick={() => pick(w)}
                  className="block w-full text-left font-mono text-[12px] px-2 py-1.5 rounded hover:bg-ink-50 text-ink-700"
                >
                  {w}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="px-4 py-3 border-b border-ink-200 space-y-2">
          <SectionLabel>Browse</SectionLabel>
          <div className="flex gap-2">
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") go(path);
              }}
              spellCheck={false}
              className="flex-1 h-8 px-2.5 rounded-md border border-ink-200 bg-ink-50 font-mono text-[12px] focus:bg-white focus:border-ink-400"
              placeholder="/path/to/project"
            />
            <Button variant="secondary" size="sm" onClick={() => go(path)}>
              Go
            </Button>
          </div>
          {browse && (
            <div className="rounded-md border border-ink-200 max-h-56 overflow-y-auto bg-white">
              {browse.parent && (
                <button
                  onClick={() => go(browse.parent!)}
                  className="block w-full text-left font-mono text-[12px] px-3 py-1.5 hover:bg-ink-50 text-ink-500 border-b border-ink-100"
                >
                  ../
                </button>
              )}
              {browse.entries.map((e) => (
                <div
                  key={e.path}
                  className="flex items-center gap-2 px-3 py-1.5 hover:bg-ink-50 border-b border-ink-50 last:border-0"
                >
                  <button
                    onClick={() => go(e.path)}
                    className="flex-1 text-left font-mono text-[12px] text-ink-700"
                  >
                    {e.name}/
                  </button>
                  {e.is_workspace && (
                    <Button size="sm" variant="primary" onClick={() => pick(e.path)}>
                      Open
                    </Button>
                  )}
                </div>
              ))}
              {browse.entries.length === 0 && (
                <div className="px-3 py-2 text-[12px] text-ink-400 italic">
                  empty directory
                </div>
              )}
            </div>
          )}
          {browse?.is_workspace && (
            <div className="flex justify-end">
              <Button size="sm" onClick={() => pick(browse.path)}>
                Open this folder
              </Button>
            </div>
          )}
        </div>

        {error && (
          <div className="px-4 py-2 text-[12px] text-red-600 bg-red-50 border-t border-red-100">
            {error}
          </div>
        )}

        <div className="px-4 py-3 flex justify-end">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}
