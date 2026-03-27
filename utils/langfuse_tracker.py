"""
utils/langfuse_tracker.py  –  Langfuse observability wrapper
Compatible with langfuse v3.7.0+ SDK.
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
    """Thin wrapper around Langfuse v3.7.0 SDK for structured LLM call tracking."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self.enabled: bool = False

        if not _LF_AVAILABLE:
            log.warning("Langfuse SDK not installed")
            return

        pk   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk   = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Debug: log key info (masked)
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
                log.info("✓ Langfuse v3.7.0 tracker initialised → %s", host)
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
    ) -> Dict[str, Any]:
        """
        Record an LLM generation in Langfuse v3.x using the correct API.
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

        log.info("track_llm_call | enabled=%s client=%s | trace=%s gen=%s tokens=%d",
                 self.enabled, self._client is not None, trace_name, generation_name, 
                 input_tokens + output_tokens)

        if self.enabled and self._client is not None:
            client = self._client
            try:
                # ── Langfuse v3.7.0 API uses start_generation ────────────────
                generation = client.start_generation(
                    name=generation_name,
                    model=model,
                    input=prompt,
                    output=response,
                    usage_details={
                        "input":  input_tokens,
                        "output": output_tokens,
                    },
                    cost_details={"cost_usd": cost},
                    metadata={"trace_name": trace_name, **(metadata or {})},
                )
                
                # Mark generation as complete
                generation.end()
                
                client.flush()
                log.info(
                    "✓ Langfuse logged | trace=%s gen=%s tokens=%d cost=$%.6f",
                    trace_name, generation_name,
                    input_tokens + output_tokens, cost,
                )
            except Exception as exc:
                log.error("✗ Langfuse logging error: %s", exc, exc_info=True)
        else:
            log.debug("Langfuse tracking disabled (enabled=%s client=%s)", 
                     self.enabled, self._client is not None)

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
