"""Pre-launch preflight: confirm
(a) local_llm() no longer runs away on a real-shaped snippet,
(b) OPENAI_KEY actually authenticates against GPT-4o (so the Ragas judge will work),
(c) Ollama keeps the model in VRAM (not spilled to RAM).
"""
import os
import time
import json

from dotenv import load_dotenv
load_dotenv()

import urllib.request

# (a) Qwen structured-output sanity check
from src.llm import local_llm
from src.chains.chain import OptimizedCode

PROMPT = """Optimize this Python function for runtime performance.

def _check_requirements(self, skill_meta: dict) -> bool:
    \"\"\"Check if skill requirements are met (bins, env vars).\"\"\"
    requires = skill_meta.get("requires", {})
    required_bins = requires.get("bins", [])
    required_env_vars = requires.get("env", [])
    return all(shutil.which(cmd) for cmd in required_bins) and all(
        os.environ.get(var) for var in required_env_vars
    )
"""

print("=" * 60)
print("(a) Qwen structured output: real snippet, neutralized params")
print("=" * 60)
llm = local_llm().with_structured_output(OptimizedCode, include_raw=True)
t0 = time.time()
result = llm.invoke(PROMPT)
dt = time.time() - t0
raw = result.get("raw")
parsed = result.get("parsed")
meta = getattr(raw, "usage_metadata", None) if raw else None
rmeta = getattr(raw, "response_metadata", {}) if raw else {}
print(f"  wall: {dt:.1f}s")
print(f"  finish_reason: {rmeta.get('finish_reason')}")
print(f"  usage_metadata: {meta}")
out_tokens = (meta or {}).get("output_tokens", -1)
ok_qwen = (rmeta.get("finish_reason") in ("stop", "tool_calls")
           and 0 < out_tokens < 4000
           and parsed is not None)
print(f"  -> Qwen OK: {ok_qwen}")

# (b) OPENAI_KEY check
print()
print("=" * 60)
print("(b) GPT-4o auth (Ragas judge will use this)")
print("=" * 60)
from langchain_openai import ChatOpenAI
gpt = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_KEY"], max_tokens=10)
try:
    r = gpt.invoke("Reply with the single word OK")
    print(f"  response: {r.content!r}")
    ok_judge = True
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    ok_judge = False
print(f"  -> Judge OK: {ok_judge}")

# (c) VRAM check
print()
print("=" * 60)
print("(c) Ollama model placement (VRAM vs RAM)")
print("=" * 60)
with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as resp:
    ps = json.loads(resp.read())
for m in ps.get("models", []):
    sz = m.get("size", 0)
    vram = m.get("size_vram", 0)
    pct = 100 * vram / sz if sz else 0
    print(f"  {m['name']}: total={sz/1e9:.2f}GB, vram={vram/1e9:.2f}GB ({pct:.0f}% on GPU)")
ok_vram = bool(ps.get("models")) and ps["models"][0].get("size_vram", 0) > 0.9 * ps["models"][0].get("size", 1)
print(f"  -> VRAM OK: {ok_vram}")

print()
print("=" * 60)
print(f"OVERALL: qwen={ok_qwen} judge={ok_judge} vram={ok_vram}")
print("=" * 60)
