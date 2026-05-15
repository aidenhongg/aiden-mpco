"""ChatOllama smoke test — direct + structured output."""
import os, time
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class OptimizedCode(BaseModel):
    code: str = Field(description="Return ONLY the optimized code.")


PROMPT = """Optimize this Python function for runtime performance.

def _check_requirements(self, skill_meta: dict) -> bool:
    requires = skill_meta.get("requires", {})
    bins = requires.get("bins", [])
    env_vars = requires.get("env", [])
    return all(shutil.which(c) for c in bins) and all(os.environ.get(v) for v in env_vars)
"""


def make():
    return ChatOllama(
        model=os.environ["LOCAL_LLM_MODEL"],
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=2048,
        num_ctx=8192,
        repeat_penalty=1.0,
        top_p=1.0,
        top_k=0,
        seed=42,
    )


print(f"model: {os.environ['LOCAL_LLM_MODEL']}")

print("\n=== T1: ChatOllama plain text ===")
llm = make()
t0 = time.time()
try:
    r = llm.invoke(PROMPT)
    dt = time.time() - t0
    meta = getattr(r, "usage_metadata", None)
    rmeta = getattr(r, "response_metadata", {}) or {}
    print(f"  wall={dt:.1f}s")
    print(f"  finish: {rmeta.get('done_reason')}")
    print(f"  usage: {meta}")
    print(f"  head: {r.content[:200]!r}")
except Exception as e:
    print(f"  FAIL after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")

print("\n=== T2: ChatOllama with_structured_output(include_raw=True) ===")
llm_struct = llm.with_structured_output(OptimizedCode, include_raw=True)
t0 = time.time()
try:
    r = llm_struct.invoke(PROMPT)
    dt = time.time() - t0
    raw = r.get("raw") if isinstance(r, dict) else None
    parsed = r.get("parsed") if isinstance(r, dict) else None
    meta = getattr(raw, "usage_metadata", None) if raw else None
    rmeta = getattr(raw, "response_metadata", {}) if raw else {}
    print(f"  wall={dt:.1f}s")
    print(f"  finish: {rmeta.get('done_reason')}")
    print(f"  usage: {meta}")
    print(f"  parsed: {parsed!r}"[:300])
except Exception as e:
    print(f"  FAIL after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")

print("\n=== DONE ===")
