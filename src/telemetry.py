import os
from uuid import UUID

from langsmith import Client

_client = Client(api_key=os.environ["LANGSMITH_API_KEY"])


def fetch_run_stats(run_id: UUID) -> dict:
    run = _client.read_run(run_id)

    prompt_tokens = run.prompt_tokens or 0
    completion_tokens = run.completion_tokens or 0

    start = run.start_time
    end = run.end_time
    first_token = run.first_token_time

    total_latency = (end - start).total_seconds() if start and end else 0.0
    ttft = (first_token - start).total_seconds() if first_token and start else None
    tps = completion_tokens / total_latency if total_latency > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft,
        "total_latency": total_latency,
        "tokens_per_second": tps,
    }
