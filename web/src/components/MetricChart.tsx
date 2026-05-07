"use client";

import { useEffect, useRef, useState } from "react";
import { api, LiveMetric } from "@/lib/api";
import { Pill, SectionLabel } from "./ui";

/**
 * Live per-epoch metric chart. Polls /api/runs/{id}/live-metric every 1.2s
 * while `live` is true; otherwise fetches once. Renders a hand-rolled SVG
 * line chart so we don't pull in a chart library.
 */
export function MetricChart({
  runId,
  live,
  height = 180,
}: {
  runId: string;
  live: boolean;
  height?: number;
}) {
  const [data, setData] = useState<LiveMetric | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    let timer: number | null = null;

    async function pull() {
      try {
        const d = await api.getLiveMetric(runId);
        if (!cancelled.current) setData(d);
      } catch (e) {
        if (!cancelled.current) setError(String((e as Error).message ?? e));
      }
    }

    pull();
    if (live) {
      timer = window.setInterval(pull, 1200);
    }
    return () => {
      cancelled.current = true;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [runId, live]);

  if (error) {
    return (
      <div className="text-[12px] text-red-600 px-3 py-2 rounded-md bg-red-50 border border-red-100">
        {error}
      </div>
    );
  }

  if (!data || data.values.length === 0) {
    return (
      <div
        className="rounded-md border border-dashed border-ink-200 bg-white text-[12px] text-ink-400 italic flex items-center justify-center"
        style={{ height }}
      >
        {live ? "waiting for first epoch…" : "no metric data yet"}
      </div>
    );
  }

  const lastValue = data.values[data.values.length - 1];
  const thresholdNum =
    typeof data.threshold === "number"
      ? data.threshold
      : typeof data.threshold === "string"
        ? parseFloat(data.threshold)
        : NaN;

  return (
    <div className="rounded-md border border-ink-200 bg-white">
      <div className="px-3 py-2 border-b border-ink-100 flex items-center gap-2">
        <SectionLabel>{data.source === "live" ? "Metric · live" : "Metric"}</SectionLabel>
        <span className="font-mono text-[11px] text-ink-500 truncate">
          {data.metric_name ?? "—"}
        </span>
        {data.source === "live" && (
          <span className="text-[10px] text-green-600 flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            updating
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {data.status && (
            <Pill tone={data.status === "PASS" ? "pass" : "fail"}>
              {data.status}
            </Pill>
          )}
          <span className="font-mono tabular-nums text-[13px] text-ink-900">
            {lastValue.value.toFixed(4)}
          </span>
          <span className="font-mono text-[11px] text-ink-500">
            @ epoch {lastValue.epoch}
          </span>
        </div>
      </div>
      <Sparkline
        values={data.values}
        threshold={Number.isFinite(thresholdNum) ? thresholdNum : null}
        height={height}
        passing={data.status === "PASS"}
      />
      {data.threshold != null && (
        <div className="px-3 py-1.5 border-t border-ink-100 text-[11px] text-ink-500 flex items-center justify-between">
          <span>
            threshold: <span className="font-mono text-ink-700">{String(data.threshold)}</span>
          </span>
          {data.direction && (
            <span>
              direction: <span className="font-mono text-ink-700">{data.direction}</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function Sparkline({
  values,
  threshold,
  height,
  passing,
}: {
  values: { epoch: number; value: number }[];
  threshold: number | null;
  height: number;
  passing: boolean;
}) {
  const w = 720;
  const h = height;
  const padL = 36;
  const padR = 16;
  const padT = 10;
  const padB = 22;

  const epochs = values.map((v) => v.epoch);
  const vals = values.map((v) => v.value);
  const xMin = Math.min(...epochs);
  const xMax = Math.max(...epochs);
  let yMin = Math.min(...vals, ...(threshold !== null ? [threshold] : []));
  let yMax = Math.max(...vals, ...(threshold !== null ? [threshold] : []));
  if (yMin === yMax) {
    yMin -= 0.5;
    yMax += 0.5;
  } else {
    const padY = (yMax - yMin) * 0.08;
    yMin -= padY;
    yMax += padY;
  }
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const x = (e: number) => padL + ((e - xMin) / xRange) * (w - padL - padR);
  const y = (v: number) => padT + (1 - (v - yMin) / yRange) * (h - padT - padB);

  const pointsAttr = values.map((p) => `${x(p.epoch)},${y(p.value)}`).join(" ");

  // 4 horizontal grid lines
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" className="block">
      {/* y-grid */}
      {ticks.map((t, i) => {
        const yy = y(t);
        return (
          <g key={i}>
            <line
              x1={padL}
              x2={w - padR}
              y1={yy}
              y2={yy}
              stroke="#e8e8ea"
              strokeWidth={1}
            />
            <text
              x={padL - 6}
              y={yy + 3}
              fontSize="9"
              textAnchor="end"
              fill="#71717a"
              fontFamily="ui-monospace, monospace"
            >
              {t.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* threshold */}
      {threshold !== null && (
        <g>
          <line
            x1={padL}
            x2={w - padR}
            y1={y(threshold)}
            y2={y(threshold)}
            stroke="#dc2626"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
          <text
            x={w - padR - 4}
            y={y(threshold) - 4}
            fontSize="9"
            textAnchor="end"
            fill="#dc2626"
            fontFamily="ui-monospace, monospace"
          >
            threshold {threshold.toFixed(3)}
          </text>
        </g>
      )}

      {/* line */}
      <polyline
        fill="none"
        stroke={passing ? "#16a34a" : "#18181b"}
        strokeWidth={1.6}
        points={pointsAttr}
      />

      {/* points */}
      {values.map((p, i) => (
        <circle
          key={i}
          cx={x(p.epoch)}
          cy={y(p.value)}
          r={2.2}
          fill={passing ? "#16a34a" : "#18181b"}
        />
      ))}

      {/* x-axis labels (start, mid, end) */}
      {(() => {
        const labels: { e: number; text: string }[] = [];
        const idxs =
          values.length <= 5
            ? values.map((_, i) => i)
            : [0, Math.floor(values.length / 2), values.length - 1];
        idxs.forEach((i) =>
          labels.push({ e: values[i].epoch, text: String(values[i].epoch) }),
        );
        return labels.map((l, i) => (
          <text
            key={i}
            x={x(l.e)}
            y={h - 6}
            fontSize="9"
            textAnchor="middle"
            fill="#71717a"
            fontFamily="ui-monospace, monospace"
          >
            ep {l.text}
          </text>
        ));
      })()}
    </svg>
  );
}
