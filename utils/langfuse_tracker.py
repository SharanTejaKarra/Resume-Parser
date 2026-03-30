"""
utils/langfuse_tracker.py  –  Langfuse observability wrapper
Compatible with langfuse v2.20.0 SDK.
Tracks: prompt, response, token usage, estimated cost, latency.
"""
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger("langfuse_tracker")

# ── Try to import Langfuse; fail gracefully ────────────────────────────────────
try:
    from langfuse import Langfuse
    from langfuse.model import ModelUsage   # ← correct import for v2.20.0
    _LF_AVAILABLE = True
except ImportError:
    _LF_AVAILABLE = False

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
    """Thin wrapper around Langfuse v2.20.0 SDK for structured LLM call tracking."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self.enabled: bool = False

        if not _LF_AVAILABLE:
            log.warning("Langfuse SDK not installed")
            return

        pk   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk   = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        pk_masked = pk[:10] + "..." if pk else "NOT_SET"
        sk_masked = sk[:10] + "..." if sk else "NOT_SET"
        log.info("Langfuse init: pk=%s sk=%s host=%s", pk_masked, sk_masked, host)

        if pk and sk and not pk.startswith("pk-lf-your"):
            try:
                self._client = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=host,
                )
                self.enabled = True
                log.info("✓ Langfuse v2.20.0 tracker initialised → %s", host)
            except Exception as exc:
                log.error("✗ Langfuse init failed: %s", exc, exc_info=True)
        else:
            log.warning("Langfuse keys not configured – tracking disabled (local mode)")

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
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Record an LLM generation in Langfuse v2.20.0.
        Uses trace → generation pattern so latency shows correctly.
        """
        now        = datetime.now(timezone.utc)
        start_time = start_time or now
        end_time   = end_time   or now
        latency_ms = (end_time - start_time).total_seconds() * 1000

        cost = self._estimate_cost(model, input_tokens, output_tokens)

        summary: Dict[str, Any] = {
            "trace_name":     trace_name,
            "generation":     generation_name,
            "model":          model,
            "input_tokens":   input_tokens,
            "output_tokens":  output_tokens,
            "total_tokens":   input_tokens + output_tokens,
            "cost_usd":       cost,
            "latency_ms":     latency_ms,
            "prompt_chars":   len(prompt),
            "response_chars": len(response),
            "metadata":       metadata or {},
        }

        log.info(
            "track_llm_call | enabled=%s | trace=%s gen=%s tokens=%d latency=%.1fms",
            self.enabled, trace_name, generation_name,
            input_tokens + output_tokens, latency_ms,
        )

        if self.enabled and self._client is not None:
            try:
                # ── v2.20.0: trace → generation ───────────────────────────────
                trace = self._client.trace(
                    name=trace_name,
                    input=prompt,
                    output=response,
                    metadata=metadata or {},
                )

                trace.generation(
                    name=generation_name,
                    model=model,
                    input=prompt,
                    output=response,
                    start_time=start_time,
                    end_time=end_time,
                    usage=ModelUsage(               # ← TypedDict, pass as dict
                        unit="TOKENS",
                        input=input_tokens,
                        output=output_tokens,
                        total=input_tokens + output_tokens,
                        input_cost=cost * 0.5,      # approximate split
                        output_cost=cost * 0.5,
                        total_cost=cost,            # ← this shows in Total Cost column
                    ),
                    metadata={
                        "cost_usd":   cost,
                        "latency_ms": latency_ms,
                        **(metadata or {}),
                    },
                )

                self._client.flush()
                log.info(
                    "✓ Langfuse logged | trace=%s gen=%s tokens=%d cost=$%.6f latency=%.1fms",
                    trace_name, generation_name,
                    input_tokens + output_tokens, cost, latency_ms,
                )
            except Exception as exc:
                log.error("✗ Langfuse logging error: %s", exc, exc_info=True)
                print(f"LANGFUSE ERROR: {exc}")
        else:
            log.debug("Langfuse tracking disabled (enabled=%s)", self.enabled)

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


# Singleton instance
tracker = LangfuseTracker()