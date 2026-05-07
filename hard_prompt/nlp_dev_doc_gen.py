PROMPT_THREE = """
You are an expert ML/DL software architect. You will receive a probe design below — a specific angle for inspecting and evaluating the training quality of an ML/DL model.

Your task is to turn this probe idea into 3 distinct, detailed development documents. Each document is a concrete, actionable implementation plan that a developer can follow to build this probe into a training pipeline.

The 3 plans should differ in approach, complexity, or methodology — for example: one lightweight/heuristic approach, one statistically rigorous approach, one tooling/library-based approach. This gives the developer meaningful alternatives to choose from.

For each plan, fill in all four fields:

`content` — a thorough implementation description covering:
- What exactly is being measured or inspected
- Where in the training pipeline this probe is inserted (e.g. per epoch, post-training, on validation set)
- Step-by-step implementation instructions (data collection, computation, how to produce the metric)
- What output or artifact the probe produces (log, plot, metric value, alert)

`metric` — a single, concrete, numerical quantity that this probe produces and that can be computed once per epoch (or per regular training interval). It must be a float a script can record automatically at each step and plot as a time series (e.g. "mean gradient norm across all layers per epoch", "validation loss per epoch", "macro F1 on held-out set per epoch"). Do not define a metric that is only computable once after training ends.

`threshold` — a single float value that acts as the reference line on a time-series plot of the metric — the boundary between healthy and problematic (e.g. 0.05, 10.0, 0.75). Express it as a plain number, then briefly justify it. The implementation will draw this as a horizontal line on the metric chart so a human can instantly see whether the curve is above or below the target. The threshold must be realistic and achievable: calibrate it against what a standard training run on this model architecture and dataset can plausibly reach. A threshold that is tighter than the model's natural capability will never be crossed regardless of optimization effort — if in doubt, set a conservative (easier) threshold that a well-tuned run can meet, rather than an aspirational one that cannot.

`confidence` — set to 0.0 for all plans; it will be filled by a supervisor agent.

Return exactly this JSON format:
{
    "dev_plans": [
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 }
    ]
}

Return only the JSON. No explanation outside the JSON.

Probe design to expand:
"""
