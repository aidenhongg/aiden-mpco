import os

import tiktoken
from langchain_openai import ChatOpenAI

MAX_INPUT_TOKENS = 6000
SEED = 42

ENCODING = tiktoken.get_encoding("cl100k_base")


def local_llm() -> ChatOpenAI:
    """ChatOpenAI pointed at the local Ollama OpenAI-compat endpoint.

    Used for both target and meta-prompt roles. `temperature=0` + `seed` give the
    best reproducibility we can get; CUDA non-determinism may still produce rare
    token-level drift.
    """
    return ChatOpenAI(
        model=os.environ["LOCAL_LLM_MODEL"],
        base_url=os.environ["LOCAL_LLM_BASE_URL"],
        api_key="ollama",
        temperature=0,
        max_tokens=2048,
        model_kwargs={"seed": SEED},
    )


def count_tokens(text: str) -> int:
    """Approximate Qwen token count via cl100k_base. Close enough for hard caps."""
    return len(ENCODING.encode(text))
