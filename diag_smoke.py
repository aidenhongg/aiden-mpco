"""End-to-end smoke through the actual chain code paths."""
import os, time
from dotenv import load_dotenv
load_dotenv()

from src.chains.chain import OptimizedCode, RMPChain, build_chain, invoke as chains_invoke

PROMPT_CODE = """def _check_requirements(self, skill_meta: dict) -> bool:
    requires = skill_meta.get("requires", {})
    bins = requires.get("bins", [])
    env_vars = requires.get("env", [])
    return all(shutil.which(c) for c in bins) and all(os.environ.get(v) for v in env_vars)
"""

print("=== T1: build_chain('qwen3.5-9b-q4', 'base', 'nanobot') ===")
t0 = time.time()
chain = build_chain("qwen3.5-9b-q4", "base", "nanobot")
code_out, usage = chains_invoke(chain, PROMPT_CODE, scope=[])
print(f"wall={time.time()-t0:.1f}s")
print(f"usage: {usage}")
print(f"code head: {code_out[:200]!r}")
print()
print("=== DONE ===")
