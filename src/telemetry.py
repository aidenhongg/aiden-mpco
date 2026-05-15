import time
from contextlib import contextmanager

from langchain_community.callbacks import get_openai_callback


@contextmanager
def track_run():
    """Track token usage and wall-clock latency for a single LLM call.

    Yields a dict that the caller can read after exit. `ttft` is always None
    because the streaming + structured-output combination doesn't compose
    cleanly enough to measure time-to-first-token reliably here.
    """
    stats: dict = {}
    t0 = time.time()
    with get_openai_callback() as cb:
        try:
            yield stats
        finally:
            total_latency = time.time() - t0
            # Caller may have populated prompt_tokens / completion_tokens from
            # raw.usage_metadata. Prefer those if non-zero; else fall back to
            # the langchain callback (which can be lossy on non-OpenAI backends).
            if not stats.get("prompt_tokens"):
                stats["prompt_tokens"] = cb.prompt_tokens
            if not stats.get("completion_tokens"):
                stats["completion_tokens"] = cb.completion_tokens
            stats["ttft"] = None
            stats["total_latency"] = total_latency
            stats["tokens_per_second"] = (
                stats["completion_tokens"] / total_latency if total_latency > 0 else 0.0
            )
