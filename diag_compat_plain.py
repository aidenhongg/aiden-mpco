"""Test OpenAI-compat plain text vs structured output, against qwen3.5-greedy."""
import os, time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

PROMPT = """Optimize this Python function for runtime performance. Return ONLY the optimized code.

def _check_requirements(self, skill_meta: dict) -> bool:
    requires = skill_meta.get("requires", {})
    bins = requires.get("bins", [])
    env_vars = requires.get("env", [])
    return all(shutil.which(c) for c in bins) and all(os.environ.get(v) for v in env_vars)
"""

client = OpenAI(base_url=os.environ["LOCAL_LLM_BASE_URL"], api_key="ollama")
MODEL = os.environ["LOCAL_LLM_MODEL"]
print(f"model: {MODEL}")

# T1: plain text
print("\n=== T1: plain text completion ===")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":PROMPT}],
        temperature=0,
        max_tokens=2048,
        seed=42,
    )
    dt = time.time() - t0
    print(f"  wall={dt:.1f}s finish={r.choices[0].finish_reason} comp_tokens={r.usage.completion_tokens}")
    print(f"  head: {r.choices[0].message.content[:150]!r}")
except Exception as e:
    print(f"  FAIL after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")

# T2: structured via response_format json_schema
print("\n=== T2: response_format=json_schema ===")
t0 = time.time()
schema = {
    "name": "OptimizedCode",
    "schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Return ONLY the optimized code."}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}
try:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":PROMPT}],
        temperature=0,
        max_tokens=2048,
        seed=42,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    dt = time.time() - t0
    print(f"  wall={dt:.1f}s finish={r.choices[0].finish_reason} comp_tokens={r.usage.completion_tokens}")
    print(f"  head: {r.choices[0].message.content[:200]!r}")
except Exception as e:
    print(f"  FAIL after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")

print("\n=== DONE ===")
