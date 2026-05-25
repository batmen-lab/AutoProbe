// Thin typed client for the FastAPI backend.
//
// We call the backend DIRECTLY (not through the Next dev rewrite) because
// rewrites time out around 30s, and stage 1/2 NLP chains routinely run 60–120s
// (two sequential `claude -p` subprocess calls). CORS is whitelisted server-side.
//
// Override the default with NEXT_PUBLIC_API_BASE if your backend runs elsewhere.
export const API_BASE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE) ||
  "http://127.0.0.1:8765";

export type WorkspaceState = {
  current: string | null;
  recent: string[];
};

export type RunSummary = {
  run_id: string;
  workspace: string;
  created_at: string;
  stage: number;
  phase: string;
};

export type IterationRow = {
  index: number;
  metric_name: string | null;
  metric_value: number | null;
  // Standard threshold (PASS bar) — kept under `threshold` for backward-compat
  // with older runs. The acceptable threshold is the looser bar.
  threshold: string | null;
  acceptable_threshold: string | null;
  tail_mean: number | null;
  direction: "higher_is_better" | "lower_is_better" | null;
  status: string | null;
  acceptable_met: boolean | null;
  note: string | null;
  // When the round came from the fix-plan flow, the 1-based plan index that
  // was applied. The full plan text is in response/<run>/fix_plans_<N>.json.
  fix_plan_chosen_index: number | null;
  // Auto-research only: running best metric after this iteration. Used to
  // plot the monotonic per-run chart.
  best_value: number | null;
};

export type RunRecord = {
  run_id: string;
  workspace: string;
  created_at: string;
  stage: number;
  phase: string;
  context: string | null;
  probe_index: number | null;
  plan_index: number | null;
  iterations: IterationRow[];
  debug_flags: { auto_research: boolean; threshold_override: string | null };
  tried_probe_indices: number[];
  tried_plan_indices: number[];
  current_action: string | null;
  auto_research_target_runs: number;
  auto_research_runs_completed: number;
  auto_research_best_value: number | null;
  auto_research_best_direction: "higher_is_better" | "lower_is_better" | null;
  // Fix-plan flow (stage 4). `fix_plan_round` is the iteration index the
  // currently-open set of fix plans belongs to. `fix_plan_index` is the
  // 1-based selection the user made (cleared once apply completes).
  fix_plan_round: number | null;
  fix_plan_index: number | null;
  busy: boolean;
};

export type ProbeDesign = {
  probe_type: string;
  probe_name: string;
  content: string;
  possible_sources: string[];
  confidence: number;
};

export type DevPlan = {
  content: string;
  metric: string;
  standard_threshold: string;
  acceptable_threshold: string;
  confidence: number;
};

export type FixPlan = {
  title: string;
  content: string;
  target_files: string[];
  confidence: number;
};

export type FixPlansArtifact = {
  round: number | null;
  fix_plans: FixPlan[] | null;
};

export type BrowseEntry = {
  name: string;
  path: string;
  is_workspace: boolean;
};

export type BrowseResult = {
  path: string;
  parent: string | null;
  is_workspace: boolean;
  entries: BrowseEntry[];
};

export type LiveMetricPoint = { epoch: number; value: number };

export type LiveMetric = {
  source: "live" | "completed" | "none";
  run_index?: number;
  metric_name?: string | null;
  // `threshold` is the standard threshold (PASS bar) — kept under this name
  // for backward-compat with older probers. `standard_threshold` mirrors it
  // explicitly. `acceptable_threshold` is the looser line.
  threshold?: string | number | null;
  standard_threshold?: string | number | null;
  acceptable_threshold?: string | number | null;
  direction?: "higher_is_better" | "lower_is_better" | null;
  status?: "PASS" | "FAIL" | null;
  values: LiveMetricPoint[];
};

async function http<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  // cache: "no-store" — without explicit Cache-Control headers from FastAPI,
  // the browser may apply heuristic caching to GETs (notably `/api/runs/:id`,
  // which we poll), and a stale response can overwrite fresh post-revert
  // state and erase the tried-index updates we just made.
  const res = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });
  if (!res.ok) {
    let body = "";
    try {
      body = await res.text();
    } catch {}
    throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ""}`);
  }
  return res.json();
}

export const api = {
  // workspace
  getWorkspace: () => http<WorkspaceState>("/api/workspace"),
  openWorkspace: (path: string) =>
    http<WorkspaceState>("/api/workspace/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  browse: (path: string) =>
    http<BrowseResult>("/api/workspace/browse", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  // runs
  listRuns: (workspace?: string) => {
    const q = workspace ? `?workspace=${encodeURIComponent(workspace)}` : "";
    return http<{ runs: RunSummary[] }>(`/api/runs${q}`);
  },
  newRun: (workspace?: string) =>
    http<RunRecord>("/api/runs", {
      method: "POST",
      body: JSON.stringify(workspace ? { workspace } : {}),
    }),
  getRun: (runId: string) => http<RunRecord>(`/api/runs/${runId}`),

  // stage 1
  setContext: (runId: string, context: string) =>
    http<RunRecord>(`/api/runs/${runId}/stage1/context`, {
      method: "POST",
      body: JSON.stringify({ context }),
    }),
  generateProbes: (runId: string) =>
    http<{ probe_designs: ProbeDesign[] | null }>(
      `/api/runs/${runId}/stage1/generate`,
      { method: "POST" },
    ),
  autoResearch: (runId: string) =>
    http<RunRecord>(`/api/runs/${runId}/stage1/auto-research`, {
      method: "POST",
    }),
  selectProbe: (runId: string, index: number) =>
    http<RunRecord>(`/api/runs/${runId}/stage1/select`, {
      method: "POST",
      body: JSON.stringify({ index }),
    }),
  getStage1: (runId: string) =>
    http<{ probe_designs: ProbeDesign[] | null }>(
      `/api/runs/${runId}/stage1/artifact`,
    ),

  // stage 2
  generateDevPlans: (runId: string) =>
    http<{ dev_plans: DevPlan[] | null }>(
      `/api/runs/${runId}/stage2/generate`,
      { method: "POST" },
    ),
  selectPlan: (runId: string, index: number) =>
    http<RunRecord>(`/api/runs/${runId}/stage2/select`, {
      method: "POST",
      body: JSON.stringify({ index }),
    }),
  getStage2: (runId: string) =>
    http<{ dev_plans: DevPlan[] | null }>(
      `/api/runs/${runId}/stage2/artifact`,
    ),

  // stage 3 / 4
  implement: (runId: string) =>
    http<RunRecord>(`/api/runs/${runId}/stage3/implement`, {
      method: "POST",
    }),
  iterateOnce: (runId: string) =>
    http<RunRecord>(`/api/runs/${runId}/stage4/iterate`, {
      method: "POST",
    }),
  autoResearchIterate: (runId: string, count: number) =>
    http<RunRecord>(`/api/runs/${runId}/stage4/auto-research-iterate`, {
      method: "POST",
      body: JSON.stringify({ count }),
    }),

  // stage 4: auto-pilot — loops fix-plan rounds until terminal state.
  autoFixLoop: (runId: string) =>
    http<RunRecord>(`/api/runs/${runId}/stage4/auto-fix-loop`, {
      method: "POST",
    }),

  // stage 4: fix-plan flow
  generateFixPlans: (runId: string, hint?: string) =>
    http<{ state: RunRecord } & FixPlansArtifact>(
      `/api/runs/${runId}/stage4/fix-plans/generate`,
      {
        method: "POST",
        body: JSON.stringify({ hint: hint ?? null }),
      },
    ),
  getFixPlans: (runId: string) =>
    http<FixPlansArtifact>(`/api/runs/${runId}/stage4/fix-plans/artifact`),
  selectFixPlan: (runId: string, index: number) =>
    http<RunRecord>(`/api/runs/${runId}/stage4/fix-plans/select`, {
      method: "POST",
      body: JSON.stringify({ index }),
    }),

  // revert
  revert: (runId: string, toStage: number, keepWorkspace: boolean = false) =>
    http<{ result: { deleted: string[]; stage: number; phase: string }; state: RunRecord }>(
      `/api/runs/${runId}/revert`,
      {
        method: "POST",
        body: JSON.stringify({ to_stage: toStage, keep_workspace: keepWorkspace }),
      },
    ),

  // log
  getLog: (runId: string) => http<{ log: string }>(`/api/runs/${runId}/log`),

  // live metric trajectory (per-epoch, dynamic during training)
  getLiveMetric: (runId: string) =>
    http<LiveMetric>(`/api/runs/${runId}/live-metric`),

  // cancel any in-flight stage action; resets the owning run's phase
  cancel: () =>
    http<{ killed: boolean; run: string | null }>("/api/cancel", {
      method: "POST",
    }),
};
