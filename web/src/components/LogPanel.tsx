"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";

/**
 * Persistent agent-log dock at the bottom of the session view.
 * Live-tails /api/runs/{id}/log/stream when `live` is true; otherwise pulls
 * the static log once. Collapsible, monochrome terminal styling.
 */
export function GlobalLogDock({
  runId,
  live,
}: {
  runId: string | null;
  live: boolean;
}) {
  const [open, setOpen] = useState(true);
  const [lines, setLines] = useState<string[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Always live-tail when there's a run. The `live` prop only drives the visual
  // pulse — we can't rely on it to gate the SSE because the frontend only
  // learns `busy` is true AFTER the call has returned (too late).
  useEffect(() => {
    if (!runId) {
      setLines([]);
      return;
    }
    let cancelled = false;
    setLines([]);

    const es = new EventSource(`${API_BASE}/api/runs/${runId}/log/stream`);
    es.onmessage = (ev) => {
      if (cancelled) return;
      if (!ev.data) return;
      setLines((prev) => [...prev, ev.data].slice(-2000));
    };
    es.onerror = () => {
      // browser handles reconnect
    };

    return () => {
      cancelled = true;
      es.close();
    };
  }, [runId]);

  useEffect(() => {
    if (open && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines, open]);

  if (!runId) return null;

  return (
    <div className="border-t border-ink-200 bg-white shrink-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full px-4 h-9 flex items-center gap-2 text-left hover:bg-ink-50 transition-colors"
      >
        <Caret open={open} />
        <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-ink-700">
          Agent log
        </span>
        <span className="font-mono text-[11px] text-ink-400">
          {lines.length} line{lines.length === 1 ? "" : "s"}
        </span>
        {live && (
          <span className="text-[10px] text-green-600 flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            live
          </span>
        )}
        <span className="ml-auto text-[10px] text-ink-400">
          {open ? "click to collapse" : "click to expand"}
        </span>
      </button>
      {open && (
        <div className="bg-ink-950 border-t border-ink-800">
          <div
            ref={containerRef}
            className="font-mono text-[11.5px] leading-[1.6] text-ink-200 px-4 py-3 h-56 overflow-y-auto whitespace-pre-wrap"
          >
            {lines.length === 0 ? (
              <span className="text-ink-500 italic">no agent activity yet…</span>
            ) : (
              lines.map((l, i) => <div key={i}>{l || " "}</div>)
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Caret({ open }: { open: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      className={`transition-transform text-ink-500 ${open ? "rotate-90" : ""}`}
    >
      <path
        d="M3 2l4 3-4 3"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
