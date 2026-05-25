"use client";

import { ReactNode, ButtonHTMLAttributes, useEffect, useRef } from "react";

export function Button({
  variant = "primary",
  size = "md",
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
}) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-ink-900";
  const sizes = { sm: "h-7 px-2.5 text-[12px]", md: "h-9 px-3.5 text-[13px]" }[size];
  const variants = {
    primary:
      "bg-ink-950 text-ink-50 hover:bg-ink-800 active:bg-ink-900",
    secondary:
      "bg-white text-ink-900 border border-ink-200 hover:bg-ink-50 active:bg-ink-100",
    ghost:
      "bg-transparent text-ink-700 hover:bg-ink-100 active:bg-ink-200",
    danger:
      "bg-white text-red-600 border border-red-200 hover:bg-red-50",
  }[variant];
  return <button className={`${base} ${sizes} ${variants} ${className}`} {...rest} />;
}

export function Card({
  children,
  className = "",
  selected = false,
  onClick,
}: {
  children: ReactNode;
  className?: string;
  selected?: boolean;
  onClick?: () => void;
}) {
  const interactive = onClick ? "cursor-pointer hover:border-ink-300" : "";
  const sel = selected ? "border-ink-900 ring-1 ring-ink-900" : "border-ink-200";
  return (
    <div
      onClick={onClick}
      className={`bg-white rounded-lg border ${sel} ${interactive} transition-colors ${className}`}
    >
      {children}
    </div>
  );
}

export function Pill({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "pass" | "fail" | "warn" | "active";
  children: ReactNode;
}) {
  const tones = {
    neutral: "bg-ink-100 text-ink-700",
    pass: "bg-green-50 text-green-700 border border-green-100",
    fail: "bg-red-50 text-red-700 border border-red-100",
    warn: "bg-amber-50 text-amber-700 border border-amber-100",
    active: "bg-ink-900 text-ink-50",
  }[tone];
  return (
    <span className={`inline-flex items-center h-5 px-2 rounded-full text-[11px] font-medium ${tones}`}>
      {children}
    </span>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2 min-w-[110px]">
      <div className="flex-1 h-1.5 bg-ink-100 rounded-full overflow-hidden">
        <div className="h-full bg-ink-900" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] tabular-nums text-ink-600 w-9 text-right">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-500">
      {children}
    </div>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className="animate-spin text-ink-500"
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

// ── Toast ─────────────────────────────────────────────────────────────────────
// Single-toast surface anchored bottom-right. Auto-dismisses after `ttl` ms.
export type ToastTone = "info" | "success";
export type ToastSpec = { id: number; tone: ToastTone; text: string };

export function Toast({
  toast,
  onClose,
  ttl = 3000,
}: {
  toast: ToastSpec | null;
  onClose: () => void;
  ttl?: number;
}) {
  // Capture onClose in a ref so the dismiss timer doesn't get re-armed every
  // time the parent re-renders (which it does ~1×/s due to polling). The
  // effect should only re-run when a NEW toast appears, keyed on toast.id.
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => onCloseRef.current(), ttl);
    return () => clearTimeout(t);
  }, [toast?.id, ttl]);

  if (!toast) return null;
  const colors =
    toast.tone === "success"
      ? "bg-green-50 border-green-200 text-green-800"
      : "bg-white border-ink-200 text-ink-800";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none animate-fade-in">
      <div
        className={`pointer-events-auto flex items-center gap-5 rounded-xl border-2 px-9 py-7 shadow-2xl ${colors}`}
      >
        {toast.tone === "success" && (
          <svg width="36" height="36" viewBox="0 0 12 12" fill="none">
            <circle cx="6" cy="6" r="5.5" fill="currentColor" opacity="0.15" />
            <path
              d="M3.5 6.5l1.8 1.8L8.5 4.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
        <div className="text-[20px] font-semibold leading-snug max-w-xl">
          {toast.text}
        </div>
        <button
          onClick={onClose}
          className="ml-2 text-ink-400 hover:text-ink-700 text-[28px] leading-none px-2"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}

// ── Action status banner ─────────────────────────────────────────────────────
// Shown above stage content. While the pipeline is busy it spins and shows
// the in-flight action's label; when idle, it surfaces a phase-specific
// "waiting for..." prompt so the user always knows what the system expects
// next.
export function ActionStatusBar({
  action,
  busy,
  idleMessage,
}: {
  action: string | null;
  busy: boolean;
  idleMessage?: string | null;
}) {
  const isBusy = busy || !!action;
  if (!isBusy && !idleMessage) return null;
  const label = isBusy ? humanizeAction(action) : idleMessage!;
  const color = isBusy
    ? "border-ink-200 bg-white text-ink-900"
    : "border-amber-200 bg-amber-50 text-amber-900";
  return (
    <div
      className={`flex items-center gap-4 px-6 py-4 rounded-lg border-2 shadow-md ${color}`}
    >
      {isBusy ? <Spinner size={26} /> : <IdleDot />}
      <div className="text-[18px] font-semibold leading-snug">{label}</div>
    </div>
  );
}

function IdleDot() {
  return (
    <span className="relative inline-flex items-center justify-center w-[26px] h-[26px]">
      <span className="absolute inset-0 rounded-full bg-amber-400 opacity-50 animate-ping" />
      <span className="relative w-3 h-3 rounded-full bg-amber-500" />
    </span>
  );
}

// Canonical stage display names. We surface these in user-facing copy
// instead of "stage N" — the index is internal jargon.
export const STAGE_NAMES: Record<number, string> = {
  1: "Probe Design",
  2: "Dev Plan",
  3: "Implementation",
  4: "Probe Fixing",
};

export function stageName(n: number): string {
  return STAGE_NAMES[n] ?? `stage ${n}`;
}

// English ordinal suffix: 1 → "1st", 2 → "2nd", 3 → "3rd", 11 → "11th", etc.
// Used wherever we'd otherwise say "iteration N" — the user prefers "Nth run".
export function ordinal(n: number): string {
  const mod100 = n % 100;
  const mod10 = n % 10;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
  if (mod10 === 1) return `${n}st`;
  if (mod10 === 2) return `${n}nd`;
  if (mod10 === 3) return `${n}rd`;
  return `${n}th`;
}

export function humanizeAction(action: string | null): string {
  if (!action) return "Working…";
  if (action === "probe-generate") return "Generating probe candidates…";
  if (action === "dev-plan-generate") return "Generating dev plans…";
  if (action === "implementation-apply")
    return "Applying dev-plan implementation (writing prober.py & integrating train.py)…";
  if (action === "post-impl-test-run")
    return "Running first probe test on the integrated train.py…";
  if (action === "auto-research-setup")
    return "Auto-research: writing prober & integrating train.py…";
  if (action.startsWith("improving-implement:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: applying improvement to train.py…`;
  }
  if (action.startsWith("iteration-test-run:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: re-running probe…`;
  }
  if (action.startsWith("fix-plan-generate:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: agent drafting 3 candidate fix plans…`;
  }
  if (action.startsWith("fix-plan-confidence:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: supervisor agent scoring fix plans…`;
  }
  if (action.startsWith("fix-plan-apply:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: applying selected fix plan to train.py…`;
  }
  if (action.startsWith("fix-plan-test-run:")) {
    const n = parseInt(action.split(":")[1], 10);
    return `${ordinal(n)} run: re-running probe after fix…`;
  }
  if (action.startsWith("auto-research-improving:")) {
    const [, k, n] = action.split(":");
    return `Auto-research ${k}/${n}: applying one improvement to train.py…`;
  }
  if (action.startsWith("auto-research-test:")) {
    const [, k, n] = action.split(":");
    return `Auto-research ${k}/${n}: re-running probe and checking for regression…`;
  }
  return action;
}

// ── Modal ────────────────────────────────────────────────────────────────────
// Lightweight dialog for confirms / multi-option choices. Click backdrop or
// Escape to dismiss (when `dismissible`).
export function Modal({
  open,
  onClose,
  children,
  dismissible = true,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  dismissible?: boolean;
}) {
  useEffect(() => {
    if (!open || !dismissible) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, dismissible, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-ink-950/45 px-6 animate-fade-in"
      onClick={() => dismissible && onClose()}
    >
      <div
        className="bg-white rounded-xl border-2 border-ink-200 shadow-2xl max-w-2xl w-full"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
