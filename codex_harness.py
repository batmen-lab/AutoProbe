"""
codex_harness.py — Reusable harness for OpenAI Codex (Responses API).

The new OpenAI Codex (2025) uses the Responses API with model codex-mini-latest.
Multi-turn is tracked server-side via previous_response_id, so you don't need to
resend full history — just chain response IDs.

Auth:
  Set OPENAI_API_KEY env var, or pass api_key= to CodexHarness().
  Switch between subscription and API key by swapping the key source.
  Organization: set OPENAI_ORG_ID env var or pass org_id=.

Install: pip install openai
"""

import os
from typing import Iterator, Optional

import openai


class CodexHarness:
    """
    Encapsulated harness for multi-turn Codex / OpenAI conversations.

    Two conversation modes (set mode= on __init__):
      "responses"  — Responses API with server-side state (default, recommended
                     for Codex; multi-turn via previous_response_id)
      "chat"       — Chat Completions API with client-side history
                     (stateless, compatible with all OpenAI models)

    Usage:
        h = CodexHarness()
        h.start_conversation(instructions="You are a helpful coding assistant.")
        reply = h.query("Write a Python hello world.")
        for chunk in h.stream_query("Now add a docstring."):
            print(chunk, end="", flush=True)
        h.exit()
    """

    RESPONSES_MODE = "responses"
    CHAT_MODE = "chat"

    def __init__(
        self,
        model: str = "codex-mini-latest",
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        mode: str = "responses",
        max_output_tokens: int = 8192,
    ):
        self.model = model
        self.mode = mode
        self.max_output_tokens = max_output_tokens
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            organization=org_id or os.environ.get("OPENAI_ORG_ID"),
        )
        # Responses API state
        self._previous_response_id: Optional[str] = None
        self._instructions: Optional[str] = None
        # Chat Completions state
        self._history: list[dict] = []

    # ── Conversation lifecycle ────────────────────────────────────────────────

    def start_conversation(self, instructions: Optional[str] = None) -> None:
        """
        Reset conversation state.

        Args:
            instructions: system-level instructions (Responses API) or system
                          prompt (Chat mode).
        """
        self._previous_response_id = None
        self._instructions = instructions
        self._history = []
        if self.mode == self.CHAT_MODE and instructions:
            self._history.append({"role": "system", "content": instructions})

    def exit(self) -> None:
        """Clear all conversation state."""
        self._previous_response_id = None
        self._instructions = None
        self._history = []

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(self, prompt: str, **kwargs) -> str:
        """Send a message and return the full text response."""
        if self.mode == self.RESPONSES_MODE:
            return self._responses_query(prompt, **kwargs)
        return self._chat_query(prompt, **kwargs)

    def stream_query(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Send a message and yield text chunks as they arrive.

        Example:
            for chunk in h.stream_query("Explain recursion briefly"):
                print(chunk, end="", flush=True)
        """
        if self.mode == self.RESPONSES_MODE:
            yield from self._responses_stream(prompt, **kwargs)
        else:
            yield from self._chat_stream(prompt, **kwargs)

    # ── Introspection ─────────────────────────────────────────────────────────

    def history(self) -> list[dict]:
        """Return chat history (Chat mode) or empty list (Responses mode)."""
        return list(self._history)

    def last_response_id(self) -> Optional[str]:
        """Return the last Responses API response ID (Responses mode only)."""
        return self._previous_response_id

    # ── Responses API internals ───────────────────────────────────────────────

    def _responses_query(self, prompt: str, **kwargs) -> str:
        params = self._responses_params(prompt, stream=False, **kwargs)
        response = self.client.responses.create(**params)
        self._previous_response_id = response.id
        return response.output_text

    def _responses_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        params = self._responses_params(prompt, stream=True, **kwargs)
        full_text = ""
        with self.client.responses.stream(**params) as stream:
            for event in stream:
                # output_text_delta events carry incremental text
                if event.type == "response.output_text.delta":
                    chunk = event.delta
                    full_text += chunk
                    yield chunk
                elif event.type == "response.completed":
                    self._previous_response_id = event.response.id

    def _responses_params(self, prompt: str, stream: bool, **kwargs) -> dict:
        params: dict = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": kwargs.pop("max_output_tokens", self.max_output_tokens),
            "stream": stream,
        }
        if self._previous_response_id:
            params["previous_response_id"] = self._previous_response_id
        if self._instructions:
            params["instructions"] = self._instructions
        params.update(kwargs)
        return params

    # ── Chat Completions internals ────────────────────────────────────────────

    def _chat_query(self, prompt: str, **kwargs) -> str:
        self._history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._history,
            max_completion_tokens=kwargs.pop(
                "max_completion_tokens", self.max_output_tokens
            ),
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": text})
        return text

    def _chat_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        self._history.append({"role": "user", "content": prompt})
        full_text = ""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._history,
            max_completion_tokens=kwargs.pop(
                "max_completion_tokens", self.max_output_tokens
            ),
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            yield delta
        self._history.append({"role": "assistant", "content": full_text})


# ── CLI demo ──────────────────────────────────────────────────────────────────

def _demo():
    print("=== Responses API mode (codex-mini-latest) ===")
    h = CodexHarness(model="codex-mini-latest", mode="responses")
    h.start_conversation(instructions="You are a concise coding assistant.")

    print("Query 1:")
    print(h.query("Write a one-line Python function that adds two numbers."))
    print("Response ID:", h.last_response_id())

    print("\nQuery 2 (continuation):")
    print(h.query("Now add a type hint to it."))

    print("\nStreaming query:")
    for chunk in h.stream_query("Wrap it in a class with a docstring."):
        print(chunk, end="", flush=True)
    print()
    h.exit()

    print("\n=== Chat Completions mode (fallback) ===")
    h2 = CodexHarness(model="gpt-4o-mini", mode="chat")
    h2.start_conversation(instructions="You are a helpful assistant.")
    print(h2.query("What is 3 + 3? One sentence."))
    print("History turns:", len(h2.history()))
    h2.exit()


if __name__ == "__main__":
    _demo()
