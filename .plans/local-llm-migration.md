# Local LLM Migration

**Status:** IN_PROGRESS
**Created:** 2026-05-05
**Owner:** aidenhong77@gmail.com
**Last update:** 2026-05-05 — P1–P5 landed in commits 91ab9db, 4f4c0ca, b488df4, 02c1fb9, e32258f. P6 is the operator gate (resample + run).

## Goal

Replace cloud API calls (OpenAI / Anthropic / Google) with a single local Qwen3.5 9B Q4
served by Ollama on the same machine that runs the experiment (RTX 2070, 8GB VRAM —
no remote box, `LOCAL_LLM_BASE_URL` points at `http://localhost:11434/v1`). Drop
multi-model dupe trials. Set `temperature=0` (and `seed`) for reproducibility. Adjust
repo sampling to post-training-cutoff to prevent data leakage.

## Decisions locked in

- **Model:** Qwen3.5 9B Q4_K_M (released ~Apr 2026, training cutoff Dec 2025).
- **Server:** Ollama. OpenAI-compatible endpoint, returns `usage` for token accounting,
  reliable JSON mode for `OptimizedCode` / `GeneratedPrompt`. Run with
  `OLLAMA_KV_CACHE_TYPE=q4_0` + flash attention to extend usable context to ~8K on 8GB.
- **Judge:** Stays GPT-4o (Ragas + RMP convergence). Keeps eval integrity; cost is negligible.
- **Sampling cutoff:** `CREATED_AFTER = '2025-12-01'`. Existing `repos.json` (CorridorKey,
  Memori) is pre-cutoff and gets re-sampled. `MIN_STARS = 100` held; lower to 50 if
  `sample_main` runs dry.
- **Reproducibility:** `temperature=0`, `seed=42`. CUDA non-determinism may still produce
  rare token-level drift — documented, not fought.
- **Context budget:** `MAX_INPUT_TOKENS = 6000` (leaves ~2K output in 8K window). Truncations
  logged via the `Tee` in `main.py`.
- **Telemetry:** Drop LangSmith. Use `get_openai_callback` + `time.time()`. `ttft` becomes
  `None` (streaming + structured output don't compose cleanly).
- **Graphing:** Flat single-bar-per-prompt. Drop agent facet.

## File-by-file changes

### New: `src/llm.py`
Single factory returning a configured `ChatOpenAI` for both target and meta roles. Reads
`LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL` from env. ~20 lines. Also exposes
`count_tokens(text)` using `tiktoken` cl100k_base (close-enough for Qwen for hard caps).

### New: `src/chains/truncate.py`
- `truncate_head(text, n_tokens, *, label) -> str` — keep tail, drop head (for refiner
  prompts where the trailing instruction matters most).
- `truncate_tail(text, n_tokens, *, label) -> str` — keep head, drop tail (for code
  snippets where signature + early body are most important).
- Both log `[TRUNCATE] <label>: 9421 → 6000 tokens (-3421)` to stdout.

### `src/chains/chain.py`
- `AGENTS` collapses to `{"qwen3.5-9b-q4": local_llm}`.
- `RMPChain.__init__`: `self._llm = self._meta_llm = local_llm()`.
- Drop `langchain_anthropic`, `langchain_google_genai`, `langsmith` imports + `@traceable`.
- `invoke()`: tail-truncate `inputs["code"]` to budget before templating.
- `_refine_prompt()`: head-truncate `p_current` before each refiner call.
- `RMPChain.invoke()`: cap `self._cached_prompt` (head-truncate) before system message.

### `src/chains/evaluation.py`
Unchanged. Judge stays GPT-4o.

### `src/mainloop.py`
- Drop `for agent_name in chains.AGENTS` outer loop.
- `agent_name = next(iter(chains.AGENTS))` once at top.
- Combo key still `f"{agent_name}/{prompt_name}"` — preserves `results.json` schema.
- Replace `telemetry.fetch_run_stats(run_id)` call with `track_run()` context manager.

### `src/telemetry.py`
Rewrite: `track_run()` context manager wrapping `get_openai_callback` + timer. Returns
`{prompt_tokens, completion_tokens, total_latency, tokens_per_second, ttft: None}`.
Drop `langsmith.Client`.

### `graphing/graphing.py`
- Drop `AGENT_LABELS`, `AGENT_COLORS`.
- `_iter_combos` yields `(prompt_label, proj_data)` (2-tuple).
- `_aggregate_snippets` keys on prompt only.
- `_bar_chart` becomes single-color flat bars per prompt.

### `setup.py`
- `CREATED_AFTER = '2025-12-01'`.
- Note in commit: existing `repos.json` invalid; must re-run `python setup.py -s 10`. Old
  speedscope profiles + `repos/` dir get wiped.

### `prompts/considerations.json`
Shrink to one entry: `qwen3.5-9b-q4`. Text reflects model strengths (concise reasoning,
JSON-mode discipline, no thinking tags in code output).

### `tests/test_rmp.py`
- Update `_make_rmp_chain` mocks: meta_llm and target llm now share the same factory; one
  patch site instead of two.
- `AGENTS` patching uses single new key.
- `TestBuildChainRouting` updates accordingly.

### `requirements.txt`
- Remove: `langchain-anthropic`, `langchain-google-genai`, `langsmith`.
- Add: `tiktoken`.

### `.env`
- Add: `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`.
- Remove: `ANTHROPIC_KEY`, `GEMINI_KEY`, `LANGSMITH_API_KEY`.
- Keep: `OPENAI_KEY` (judge), `GITHUB_KEY` (setup).

## Phases

- [x] **P1 — Scaffolding.** Created `src/llm.py`, `src/chains/truncate.py`. Updated
  `requirements.txt`. Live Ollama smoke-test deferred to operator (no Ollama in code env).
- [x] **P2 — Wire chain + telemetry.** Rewrote `chain.py`, `telemetry.py`, dropped the deps
  and `@traceable`. Updated `tests/test_rmp.py` mocks (and fixed a pre-existing patch-lifetime
  bug in the test helper). All 12 tests pass.
- [x] **P3 — Mainloop + truncation.** Dropped agent loop, integrated `track_run`, applied
  truncation at the three sites. Tests still green. End-to-end smoke deferred to P6.
- [x] **P4 — Considerations + graphing.** Shrunk `prompts/considerations.json` to single
  qwen entry; flattened `graphing/graphing.py` to one bar per prompt.
- [x] **P5 — Resample repos (code-only).** Bumped `CREATED_AFTER` to `2025-12-01`. Operator
  follow-up: wipe `repos.json` / `repos/` / `src/profiler/profiles/`, run
  `python setup.py -s 10`. If yield < 5, drop `MIN_STARS` to 50 and retry.
- [ ] **P6 — Full run (operator).** Set `LOCAL_LLM_BASE_URL` + `LOCAL_LLM_MODEL` in `.env`,
  start Ollama on the RTX box with `OLLAMA_KV_CACHE_TYPE=q4_0` + flash attention, then
  `python main.py` and regenerate graphs. Sanity-check `results.json` schema is intact and
  metrics are populated.

## Open questions / risks

- **Ollama JSON mode reliability at Q4.** If `with_structured_output` fails frequently for
  the `OptimizedCode` schema, fall back to grammar-based llama.cpp server. Will know in P2.
- **Context overflow on long bottleneck functions.** If truncation logs show frequent code
  truncation, the optimizer is being asked to optimize a code stub — undermines the
  experiment. Mitigation: in P3, also log the original-token-count distribution across
  bottlenecks during baseline profiling so we can see this upfront.
- **5-month repo window yield.** Flagged. If `sample_main` returns < 5 viable repos, drop
  `MIN_STARS` first; if still dry, surface to user.
- **CUDA non-determinism.** Even with temp=0 + seed, ~rare token drift possible. Acceptable
  for this experiment; document in final write-up.

## Out of scope

- Streaming-based TTFT recovery (acceptable metric loss).
- Per-snippet token-budget profiling beyond a simple histogram.
- Migrating Ragas to a local judge.
- Re-baselining old experiments under the new model — fresh study only.
