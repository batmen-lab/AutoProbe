PROMPT_THREE = """
You are an expert ML/DL software architect. You will receive a probe design below — a specific angle for inspecting and evaluating the training quality of an ML/DL model.

Your task is to turn this probe idea into 3 distinct, detailed development documents. Each document is a concrete, actionable implementation plan that a developer can follow to build this probe into a training pipeline.

The 3 plans should differ in approach, complexity, or methodology — for example: one lightweight/heuristic approach, one statistically rigorous approach, one tooling/library-based approach. This gives the developer meaningful alternatives to choose from.

For each plan, fill in all five fields:

`content` — a thorough implementation description covering:
- What exactly is being measured or inspected
- Where in the training pipeline this probe is inserted (e.g. per epoch, post-training, on validation set)
- Step-by-step implementation instructions (data collection, computation, how to produce the metric)
- What output or artifact the probe produces (log, plot, metric value, alert)

`metric` — a single, concrete, numerical quantity that this probe produces and that can be computed once per epoch (or per regular training interval). It must be a float a script can record automatically at each step and plot as a time series (e.g. "mean gradient norm across all layers per epoch", "validation loss per epoch", "macro F1 on held-out set per epoch"). Do not define a metric that is only computable once after training ends.

`standard_threshold` — a single float that acts as the "the metric is genuinely strong" reference line — what a well-engineered run SHOULD reach if everything is done right. PASS/FAIL is keyed on this. Set it strict, not generous: aim for the upper end of what's plausible on this model + dataset (e.g. published competitive baselines, what a careful tuned implementation reports). It is OK for this bar to be hard to hit — the acceptable threshold below is the safety net that catches "we did well, even if not great". Without that net you'd have to soften this number; with it, you do not. Avoid being so aspirational that no realistic implementation could ever reach it (e.g. don't set 0.95 ROC-AUC on a noisy ehr task), but err on the strict side rather than the lenient side. Express it as a plain number and briefly justify it with the reference point you're calibrating against.

`acceptable_threshold` — a single float that acts as the "this is the looser bar we'd settle for" reference line. It should be on the OK-but-not-great side of `standard_threshold`. Concretely:
- If higher_is_better (e.g. accuracy, F1): acceptable_threshold < standard_threshold
- If lower_is_better (e.g. loss, RMSE): acceptable_threshold > standard_threshold
The implementation will draw this as a second dashed line on the metric chart. It does NOT change PASS/FAIL (PASS is still standard_threshold). Its role is to give the orchestrator a "we've reached the soft target, this is realistically the best we can do under project constraints" signal so it can stop pushing rather than chase an unreachable standard target indefinitely. Justify it in one short sentence (e.g. "RMSE 0.5 is what a decently-tuned baseline on this size of dataset typically reaches before diminishing returns").

`confidence` — set to 0.0 for all plans; it will be filled by a supervisor agent.

Return exactly this JSON format:
{
    "dev_plans": [
        { "content": "string", "metric": "string", "standard_threshold": "string", "acceptable_threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "standard_threshold": "string", "acceptable_threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "standard_threshold": "string", "acceptable_threshold": "string", "confidence": 0.0 }
    ]
}

Return only the JSON. No explanation outside the JSON.

Probe design to expand:
"""
