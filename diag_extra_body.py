"""Verify extra_body forwarding by comparing OpenAI SDK direct vs LangChain ChatOpenAI."""
import os, time
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

PROMPT = """Optimize this Python function:
def _check_requirements(self, skill_meta: dict) -> bool:
    requires = skill_meta.get("requires", {})
    bins = requires.get("bins", [])
    env_vars = requires.get("env", [])
    return all(shutil.which(c) for c in bins) and all(os.environ.get(v) for v in env_vars)
"""

# Test 1: raw OpenAI client with extra_body
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
print("=== T1: Raw OpenAI client + extra_body ===")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="qwen3.5:9b-q4_K_M",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        max_tokens=2048,
        seed=42,
        extra_body={"options": {"repeat_penalty": 1.0, "top_p": 1.0, "top_k": 0, "num_predict": 2048}},
    )
    dt = time.time() - t0
    fr = r.choices[0].finish_reason
    ct = r.usage.completion_tokens
    print(f"  wall={dt:.1f}s finish={fr} completion_tokens={ct}")
    print(f"  content head: {r.choices[0].message.content[:200]!r}")
except Exception as e:
    dt = time.time() - t0
    print(f"  FAIL after {dt:.1f}s: {type(e).__name__}: {str(e)[:200]}")

print()
# Test 2: same prompt without extra_body (baseline runaway)
print("=== T2: Raw OpenAI client WITHOUT extra_body (will likely run away) ===")
print("  Skipping to avoid burning compute. The b6yrhjfe7 + b4v23dcz3 already showed runaway.")
