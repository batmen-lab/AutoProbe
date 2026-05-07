"""
claude_harness.py — Reusable harness for Anthropic Claude API (Python SDK).

Auth:
  Set ANTHROPIC_API_KEY env var, or pass api_key= to ClaudeHarness().
  Subscription users: Claude Code CLI (claude.ai subscription) does not expose
  a programmatic Python API — you need an Anthropic API key from
  console.anthropic.com. Switch by swapping the api_key value.

Install: pip install anthropic
"""

import os
from typing import Iterator, Optional

import anthropic


class ClaudeHarness:
    """
    Encapsulated harness for multi-turn Claude conversations.

    Usage:
        h = ClaudeHarness()
        h.start_conversation(system="You are a helpful assistant.")
        reply = h.query("Hello!")
        for chunk in h.stream_query("Tell me more"):
            print(chunk, end="", flush=True)
        h.exit()
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # API key: explicit arg > env var > raises at call time
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._system: Optional[str] = None
        self._history: list[dict] = []

    # ── Conversation lifecycle ────────────────────────────────────────────────

    def start_conversation(self, system: Optional[str] = None) -> None:
        """Reset history and set an optional system prompt."""
        self._system = system
        self._history = []

    def exit(self) -> None:
        """Clear conversation state."""
        self._system = None
        self._history = []

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(self, prompt: str, **kwargs) -> str:
        """Send a message and return the full text response."""
        self._history.append({"role": "user", "content": prompt})
        response = self._create(**kwargs)
        text = self._extract_text(response.content)
        self._history.append({"role": "assistant", "content": text})
        return text

    def stream_query(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Send a message and yield text chunks as they arrive.

        Example:
            for chunk in h.stream_query("Explain quantum computing"):
                print(chunk, end="", flush=True)
        """
        self._history.append({"role": "user", "content": prompt})
        full_text = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            messages=self._history,
            **({"system": self._system} if self._system else {}),
            **kwargs,
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                yield chunk
        self._history.append({"role": "assistant", "content": full_text})

    # ── Introspection ─────────────────────────────────────────────────────────

    def history(self) -> list[dict]:
        return list(self._history)

    def count_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt against current history."""
        messages = self._history + [{"role": "user", "content": prompt}]
        result = self.client.messages.count_tokens(
            model=self.model,
            messages=messages,
            **({"system": self._system} if self._system else {}),
        )
        return result.input_tokens

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create(self, **kwargs):
        return self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            messages=self._history,
            **({"system": self._system} if self._system else {}),
            **kwargs,
        )

    @staticmethod
    def _extract_text(content) -> str:
        return "".join(b.text for b in content if b.type == "text")


# ── CLI demo ──────────────────────────────────────────────────────────────────

def _demo():
    h = ClaudeHarness(model="claude-sonnet-4-6")
    h.start_conversation(system="You are a concise assistant. Keep answers short.")

    print("=== Single query ===")
    print(h.query("What is 2 + 2? One sentence."))

    print("\n=== Streaming query ===")
    for chunk in h.stream_query("Name three programming languages. One sentence."):
        print(chunk, end="", flush=True)
    print()

    print("\n=== History length:", len(h.history()), "turns ===")
    h.exit()
    print("Exited. History cleared:", len(h.history()) == 0)


if __name__ == "__main__":
    _demo()
