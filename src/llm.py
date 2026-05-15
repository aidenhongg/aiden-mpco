import os

import tiktoken
from langchain_ollama import ChatOllama

MAX_SYSTEM_TOKENS = 4000   # cap for system-role tokens
MAX_CODE_TOKENS   = 1500   # cap for human-role code tokens
# Total budget math:
#   4000 (sys) + 1500 (code) + ~250 (human template + scope + structured-output schema)
#   + 2048 (num_predict output) = ~7800.  Fits the 8192 ctx window with a ~390 token margin.
MAX_INPUT_TOKENS = MAX_SYSTEM_TOKENS  # legacy alias; prefer the slice constants
SEED = 42

ENCODING = tiktoken.get_encoding("cl100k_base")


def local_llm() -> ChatOllama:
    """ChatOllama pointed at the local Ollama daemon (native /api/chat path).

    History: we started on `langchain_openai.ChatOpenAI` against Ollama's
    OpenAI-compat endpoint (`/v1/chat/completions`). Two compat-layer issues
    made that path unworkable:
      1. `max_tokens` was silently dropped — Ollama does not translate it to
         its native `num_predict`, so generation ran until some 80K-token
         internal cap and tripped `LengthFinishReasonError` from the OpenAI
         SDK's structured-output parser.
      2. The Modelfile default `presence_penalty 1.5` (Ollama's name for what
         it internally treats as a repeat penalty) combined with `temperature=0`
         drove the model into degenerate output. OpenAI-style `presence_penalty=0`
         was ignored, and `extra_body.options.repeat_penalty=1.0` was likewise
         not honored by the compat layer.

    Switching to `ChatOllama` (langchain-ollama, talks /api/chat directly) lets
    us pass Ollama-native options (`repeat_penalty`, `num_predict`, `num_ctx`,
    `top_p`, `top_k`) as first-class kwargs — verified end-to-end against the
    daemon: bounded output, `done_reason=stop`/`length` honored.

    `temperature=0` + `seed=42` give the best reproducibility we can get; CUDA
    non-determinism may still produce rare token-level drift.
    """
    return ChatOllama(
        model=os.environ["LOCAL_LLM_MODEL"],
        base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1").replace("/v1", ""),
        temperature=0,
        seed=SEED,
        num_predict=2048,
        num_ctx=8192,
        repeat_penalty=1.0,
        top_p=1.0,
        top_k=0,
        # Qwen3.5 ships with chain-of-thought "thinking" mode enabled. Default
        # behavior emits a 1k–8k token <think>...</think> block before the
        # answer, which (a) blew through num_predict before producing content
        # and (b) made every call take minutes. Disabling reasoning gets us
        # terse direct answers (~10 tok for trivial inputs, ~hundreds for real
        # snippets). The migration plan's `considerations.json` already tells
        # the model "no thinking tags in code output" — this enforces it.
        reasoning=False,
        # Unload Qwen from VRAM immediately after each generation. The default
        # 5-min keep_alive made baseline pytest runs (no LLM yet → cold) and
        # snippet-loop check_patch runs (just generated → hot) measure under
        # different VRAM/CPU contention conditions. Load-sensitive tests
        # (hypothesis @settings(deadline=...) etc.) flipped between hot/cold,
        # which combined with the failure_count drift to silently null
        # end_runtime_avg. Cost: each chains.invoke pays a re-load (~10-15s
        # for a 6.6 GB Q4 model from disk), but every pytest now sees the
        # same cold environment as the baseline.
        keep_alive=0,
    )


def count_tokens(text: str) -> int:
    """Approximate Qwen token count via cl100k_base. Close enough for hard caps."""
    return len(ENCODING.encode(text))
