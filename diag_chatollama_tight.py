"""Test ChatOllama with reasoning disabled."""
import os, time
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=os.environ["LOCAL_LLM_MODEL"],
    base_url="http://localhost:11434",
    temperature=0,
    seed=42,
    num_predict=512,
    num_ctx=8192,
    repeat_penalty=1.0,
    top_p=1.0,
    top_k=0,
    reasoning=False,   # disable Qwen3 thinking mode
)
print(f"model: {os.environ['LOCAL_LLM_MODEL']}, num_predict=512, reasoning=False")
t0 = time.time()
r = llm.invoke("Optimize this Python function. Reply only with code:\ndef f(x): return x*2")
dt = time.time() - t0
meta = getattr(r, "usage_metadata", None)
rmeta = getattr(r, "response_metadata", {}) or {}
print(f"wall={dt:.1f}s")
print(f"done_reason: {rmeta.get('done_reason')}")
print(f"usage: {meta}")
print(f"content: {r.content[:300]!r}")
print("DONE")
