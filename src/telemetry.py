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
            tps = cb.completion_tokens / total_latency if total_latency > 0 else 0.0
            stats.update({
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "ttft": None,
                "total_latency": total_latency,
                "tokens_per_second": tps,
            })
