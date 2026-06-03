"""Reasoning-injector proxy: sits between ccr and OpenRouter.

Why this exists
---------------
OpenRouter rejects chat/completions requests for reasoning-mandatory models
(gemini-3.5-flash, gemini-2.5-pro, gpt-5.1-codex-max, etc.) with HTTP 400
when the body has no `reasoning` field. ccr's built-in `reasoning`
transformer doesn't emit the shape Google's gemini gateway wants, so we
splice it in here ourselves and keep ccr's config minimal.

Flow:
    claude CLI → ccr :3456 → THIS SHIM :4000 → openrouter.ai/api/v1

Only the `chat/completions` path is needed for ccr's openrouter transformer.
Everything else is forwarded verbatim. Streaming SSE is preserved.
"""

from __future__ import annotations

import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_EFFORT = os.environ.get("GEMINI_SHIM_EFFORT", "low")  # low | medium | high
TIMEOUT_S = float(os.environ.get("GEMINI_SHIM_TIMEOUT", "600"))

app = FastAPI()
_client = httpx.AsyncClient(timeout=TIMEOUT_S)


def _inject_reasoning(body: dict) -> dict:
    """Force-overwrite the reasoning field.

    ccr's openrouter transformer sometimes emits `reasoning: {exclude: true}`
    which OpenRouter's mandatory-reasoning gateway reads as "disabled" and
    rejects with HTTP 400. We always set `{effort: ...}` and drop any
    `reasoning_effort` shorthand so only one shape goes upstream.
    """
    body.pop("reasoning_effort", None)
    body["reasoning"] = {"effort": DEFAULT_EFFORT}
    return body


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    body = _inject_reasoning(body)

    # Pass through auth + content-type; drop hop-by-hop headers.
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "content-type", "http-referer", "x-title")
    }
    headers.setdefault("content-type", "application/json")

    url = f"{OPENROUTER_BASE}/chat/completions"
    payload = json.dumps(body).encode()
    is_stream = bool(body.get("stream"))

    if is_stream:
        # Stream SSE chunks straight back to ccr.
        req = _client.build_request("POST", url, content=payload, headers=headers)
        upstream = await _client.send(req, stream=True)

        async def gen():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            gen(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    resp = await _client.post(url, content=payload, headers=headers)
    return JSONResponse(
        content=resp.json() if resp.content else {},
        status_code=resp.status_code,
    )


# Catch-all forwarder so non-chat endpoints (models list, key info, etc.) work
# transparently if ccr ever hits them.
@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def forward(path: str, request: Request):
    raw = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "content-type", "http-referer", "x-title")
    }
    url = f"{OPENROUTER_BASE}/{path.lstrip('v1/').lstrip('/')}"
    resp = await _client.request(
        request.method, url, content=raw, headers=headers, params=request.query_params
    )
    return JSONResponse(
        content=resp.json() if resp.content else {},
        status_code=resp.status_code,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("GEMINI_SHIM_PORT", "4000")),
        log_level="warning",
    )
