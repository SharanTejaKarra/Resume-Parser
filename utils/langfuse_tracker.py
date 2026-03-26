"""
utils/langfuse_tracker.py  –  Langfuse observability wrapper
Compatible with langfuse v4.x (new SDK).
Tracks: prompt, response, token usage, estimated cost, latency.
"""
import os
import time
from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger("langfuse_tracker")

# ── Try to import Langfuse; fail gracefully ────────────────────────────────────
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    _LF_AVAILABLE = True
except ImportError:
    _LF_AVAILABLE = False
    log.warning("langfuse not installed – observability disabled")

# ── Cost table (USD per 1 K tokens) – approximate ─────────────────────────────
MODEL_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-oss:120b-cloud": {"input": 0.0025, "output": 0.010},
    "gpt-oss:120b":       {"input": 0.0025, "output": 0.010},
    "llama3.1:70b":       {"input": 0.0009, "output": 0.0009},
    "llama3.1:8b":        {"input": 0.0002, "output": 0.0002},
    "gemma3:27b":         {"input": 0.0009, "output": 0.0009},
    "default":            {"input": 0.001,  "output": 0.002},
}


class LangfuseTracker:
    """Thin wrapper around Langfuse v4 SDK for structured LLM call tracking."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self.enabled: bool = False

        if not _LF_AVAILABLE:
            return

        pk   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk   = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if pk and sk and not pk.startswith("pk-lf-your"):
            try:
                self._client = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=host,
                )
                self.enabled = True
                log.info("Langfuse v4 tracker initialised → %s", host)
            except Exception as exc:
                log.warning("Langfuse init failed: %s", exc)
        else:
            log.info("Langfuse keys not configured – tracking disabled (local mode)")

    # ──────────────────────────────────────────────────────────────────────────
    def track_llm_call(
        self,
        trace_name: str,
        generation_name: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record an LLM generation in Langfuse (v4) and return a summary dict
        that is stored in Streamlit session state for the observability UI.
        """
        cost = self._estimate_cost(model, input_tokens, output_tokens)

        summary: Dict[str, Any] = {
            "trace_name":     trace_name,
            "generation":     generation_name,
            "model":          model,
            "input_tokens":   input_tokens,
            "output_tokens":  output_tokens,
            "total_tokens":   input_tokens + output_tokens,
            "cost_usd":       cost,
            "prompt_chars":   len(prompt),
            "response_chars": len(response),
            "metadata":       metadata or {},
        }

        if self.enabled and self._client is not None:
            client = self._client   # local var so type-checkers are happy
            try:
                # ── Langfuse v4 low-level API ───────────────────────────────
                trace = client.trace(
                    name=trace_name,
                    metadata=metadata or {},
                )
                trace.generation(
                    name=generation_name,
                    model=model,
                    input=prompt,
                    output=response,
                    usage={
                        "input":  input_tokens,
                        "output": output_tokens,
                        "unit":   "TOKENS",
                    },
                    metadata={"cost_usd": cost, **(metadata or {})},
                )
                client.flush()
                log.info(
                    "Langfuse logged | trace=%s gen=%s tokens=%d cost=$%.6f",
                    trace_name, generation_name,
                    input_tokens + output_tokens, cost,
                )
            except Exception as exc:
                log.warning("Langfuse logging error: %s", exc)

        return summary

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = MODEL_COST_PER_1K.get(model, MODEL_COST_PER_1K["default"])
        return (input_tokens / 1000.0 * rates["input"]) + \
               (output_tokens / 1000.0 * rates["output"])

    # ──────────────────────────────────────────────────────────────────────────
    def flush(self) -> None:
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass


# Singleton instance (re-created when settings are changed from the sidebar)
tracker = LangfuseTracker()
