"use client";

import { ReactNode, ButtonHTMLAttributes } from "react";

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
