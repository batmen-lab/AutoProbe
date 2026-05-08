"use client";

import { ReactNode, ButtonHTMLAttributes, useEffect } from "react";

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
  ttl = 4000,
}: {
  toast: ToastSpec | null;
  onClose: () => void;
  ttl?: number;
}) {
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(onClose, ttl);
    return () => clearTimeout(t);
  }, [toast, onClose, ttl]);

  if (!toast) return null;
  const colors =
    toast.tone === "success"
      ? "bg-green-50 border-green-200 text-green-800"
      : "bg-white border-ink-200 text-ink-800";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none animate-fade-in">
      <div
        className={`pointer-events-auto flex items-center gap-3 rounded-lg border-2 px-5 py-4 shadow-lg ${colors}`}
      >
        {toast.tone === "success" && (
          <svg width="22" height="22" viewBox="0 0 12 12" fill="none">
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
        <div className="text-[14.5px] font-medium leading-snug max-w-sm">
          {toast.text}
        </div>
        <button
          onClick={onClose}
          className="ml-2 text-ink-400 hover:text-ink-700 text-[20px] leading-none px-1"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}

// ── Action status banner ─────────────────────────────────────────────────────
// Shown above stage content while a long-running action is in flight. Renders
// the human-readable label for the current action.
export function ActionStatusBar({
  action,
  busy,
}: {
  action: string | null;
  busy: boolean;
}) {
  if (!busy && !action) return null;
  const label = humanizeAction(action);
  return (
    <div className="flex items-center gap-2.5 px-3.5 py-2 rounded-md border border-ink-200 bg-white shadow-sm">
      <Spinner size={14} />
      <div className="text-[12.5px] text-ink-800 font-medium">{label}</div>
    </div>
  );
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
    const n = action.split(":")[1];
    return `Iteration ${n}: applying improvement to train.py…`;
  }
  if (action.startsWith("iteration-test-run:")) {
    const n = action.split(":")[1];
    return `Iteration ${n}: re-running probe…`;
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
      className="fixed inset-0 z-40 flex items-center justify-center bg-ink-950/40 px-4 animate-fade-in"
      onClick={() => dismissible && onClose()}
    >
      <div
        className="bg-white rounded-lg border border-ink-200 shadow-xl max-w-md w-full"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
