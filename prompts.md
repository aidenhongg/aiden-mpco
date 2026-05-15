# Comprehensive Prompt Logging — Plan

**Status:** PROPOSED
**Scope:** Capture every prompt sent and every response received (success or failure)
into `src/results.json`, with enough fidelity to reconstruct exactly what the local
LLM saw and produced on every retry of every snippet of every strategy.

---

## 1. Problem statement

Today `src/results.json` records the *outcome* of each snippet but very little about
the *journey*. Concretely:

- The rendered system + human messages sent to the LLM are **never** stored — only
  template files (`prompts/*.json`) plus, for RMP, the cached `_cached_prompt`.
- The raw LLM response (pre structured-output parse) is **never** stored — the parse
  result is, but only when parsing succeeded.
- A snippet that exhausts `MAX_RETRIES = 10` (`src/mainloop.py:13`) currently gets
  a stub record like `{"failed_regenerations": 10, "runtime_diff": 0.0}` with **zero
  diagnostic content** — no prompts, no responses, no error messages, no failure
  category. There are 5 such stubs in the current `src/results.json` for `nanobot`
  alone (lines 38, 60, 67, 74, 110).
- RMP refinement traces (`chain._refinement_trace`) are written onto the snippet
  record only **after a successful exec call** (`src/mainloop.py:107-112`). When all
  10 attempts fail, the trace is silently dropped.
- The catch-all `except Exception as e` at `src/mainloop.py:119` swallows three
  categorically different failure modes (`TestRegressionError`, pydantic parse
  errors, generic LLM errors) into the same opaque `failures += 1` counter
  (`src/mainloop.py:129`).
- Patch no-ops (`Patch.apply_patch()` returns `False` at `src/patches/patch.py:31`)
  also bump the failure counter without recording why.

Net effect: when an experiment goes sideways, the only forensic surface is the
stdout teed by `main.py` into `logs/run_*.log` — unstructured, hard to grep, and
irreversibly entangled with progress messages.

---

## 2. Goals

| # | Goal | How it's verified |
|---|---|---|
| G1 | Every LLM call's rendered system+human messages are stored in `results.json`. | Open the JSON; locate any snippet; every attempt has non-null `system_prompt` + `human_prompt` strings. |
| G2 | Every LLM call's raw response is stored, even when structured-output parsing fails. | Parse-error attempts have non-null `raw_response` and null `parsed_code`. |
| G3 | Every failure carries a **category** + **detail**, not just a counter. | Each attempt has a `status` ∈ `{success, no_op_patch, regression, parse_error, llm_error, patch_apply_error}` and, when applicable, `error_type` / `error_message` / `error_traceback`. |
| G4 | RMP intermediate prompts (proposer + each refiner iteration) are captured **on every attempt**, including failed ones. | Failed RMP snippet records contain `meta_meta_prompt`, `proposer_output`, and a `refinement` list with rendered refiner prompts even when `failed_regenerations == 10`. |
| G5 | Token/latency stats are captured per call, not only for the final exec call. | Each proposer call and each refiner iteration has its own `usage` block. |
| G6 | Existing graphing pipeline (`graphing/`) keeps working without code changes. | Successful snippets retain `original_code`, `optimized_code`, scores, `prompt_tokens`, `total_latency`, etc. at the same depth they live today. |

Non-goals: streaming TTFT (`telemetry.py:29` already documents this as
unrecoverable with structured-output), prompt replay tooling, log compression.

---

## 3. Failure-path inventory (what we have to capture)

Found by tracing `src/mainloop.py:79-129` and the chain layer:

| Where it fails | Symptom in code | Currently logged as |
|---|---|---|
| Structured-output schema rejection (model returned malformed tool call) | `pydantic.ValidationError` raised by `with_structured_output` | swallowed in `except Exception` |
| Model returned valid JSON but identical to original code | `Patch.apply_patch()` → `False` (`patch.py:31`) | `failures += 1`, `continue` (no diagnostic) |
| Patch applied but tests now fail more than baseline | `TestRegressionError` raised by `profile.check_patch()` (`profile.py:137`) | swallowed in `except Exception`; patch reverted |
| Ollama daemon error / network blip / OOM | generic exception bubbling out of `chain.invoke` | swallowed in `except Exception` |
| RMP refiner iteration itself crashes | already caught at `chain.py:85`, recorded into `_refinement_trace` with `error` key | preserved in trace, but trace is only written on outer success |
| Snippet exhausts MAX_RETRIES | loop falls through; `record["failed_regenerations"] = failures` | stub record only |
| Baseline profiling failed for a project | `RuntimeError` caught at `mainloop.py:34` | `print` only, project skipped |

We will categorize the first four explicitly, surface the fifth on every attempt
(not only successful ones), turn the sixth into a structured per-attempt log, and
leave the seventh alone — no LLM call, out of scope.

---

## 4. Schema additions to `results.json`

Backwards-compatible: existing keys stay where they are at the same depth, so
`graphing/` and any downstream readers keep working. New keys are additive.

### 4.1 Per-snippet additions

```jsonc
{
  // ... all existing keys retained: original_code, optimized_code,
  //     project_name, failed_regenerations, runtime_diff,
  //     cognitive_complexity, significance, dependency_usage,
  //     prompt_tokens, completion_tokens, ttft, total_latency,
  //     tokens_per_second, generated_prompt, meta_meta_prompt,
  //     refinement_trace, converged, refinement_iterations ...

  "snippet_id": "nanobot::build_messages::ContextBuilder",  // new — stable id
  "scope": "class ContextBuilder > def build_messages",     // new — what was passed as {scope}
  "final_status": "success" | "max_retries_exhausted",      // new — terminal verdict
  "attempts": [ <AttemptRecord>, ... ]                      // new — one entry per try
}
```

### 4.2 `AttemptRecord` (always present, success or failure)

```jsonc
{
  "attempt_idx": 0,                          // 0-based; equals failures count at entry
  "regenerate": false,                        // true on attempts ≥ 1; triggers RMP re-refinement
  "timestamp": "2026-05-06T14:32:11.428Z",

  // The actual LLM exec call
  "exec": {
    "system_prompt": "<full rendered system message after head-truncation>",
    "human_prompt": "Enclosing scope: ...\n\nObject to be optimized: ...",
    "raw_response": "<the assistant's full response text>",  // null only if call errored before any tokens
    "parsed_code": "<optimized code string>",                // null on parse_error
    "usage": {
      "prompt_tokens": 894,
      "completion_tokens": 349,
      "total_latency": 54.89,
      "tokens_per_second": 6.36,
      "ttft": null
    }
  },

  // RMP-only: the prompt-generation half of the call (absent for base/few_shot/cot)
  "rmp": {
    "meta_meta_prompt": "<full rendered proposer system message>",
    "proposer_output": "<the optimization prompt the L3 model generated>",
    "proposer_usage": { "prompt_tokens": ..., "completion_tokens": ..., "latency": ... },
    "refinement": [
      {
        "iteration": 1,
        "input_prompt": "<head-truncated p_current fed in>",
        "refiner_system_prompt": "<full rendered refiner system message>",
        "output_prompt": "<refined prompt>",                 // null on iteration error
        "convergence_score": 0.93,                            // null when not computable
        "converged": false,
        "usage": { "prompt_tokens": ..., "completion_tokens": ..., "latency": ... },
        "error": null                                         // string when iteration crashed
      },
      // ... up to MAX_REFINEMENT_ITERS = 3
    ],
    "cached_prompt_used": "<the final p_current that became the exec system prompt>"
  },

  // What happened after the LLM returned
  "patch_applied": true,            // false → no_op or apply_error
  "runtime_after_patch": 122.762,    // null when patch didn't apply or check failed
  "runtime_diff": 12.06,             // null when not computable

  // Outcome categorization
  "status": "success",
  // one of:
  //   success          — patch applied, runtime check passed
  //   no_op_patch      — Patch.apply_patch() returned False (model returned identical code)
  //   regression       — TestRegressionError (more pytest failures than baseline)
  //   parse_error      — pydantic.ValidationError from structured output
  //   llm_error        — any other exception from chain.invoke (Ollama down, OOM, ...)
  //   patch_apply_error — Patch.apply_patch raised (file IO, encoding, ...)

  "error": {
    "type": "TestRegressionError",         // exception class name; null on success
    "message": "Patch increased failures for nanobot: 0 -> 3 ...",
    "traceback": "<full traceback>",        // captured via traceback.format_exc()
    "regression_failures": [                // only on status=regression
      { "testcase": "...", "classname": "...", "message": "..." }
    ]
  }
}
```

### 4.3 Top-level project record

No structural change. The existing `meta_meta_prompt`, `generated_prompt`,
`refinement_trace`, `converged`, `refinement_iterations` keys at project level
(`src/mainloop.py:138-143`) keep their meaning: snapshot of the **last successful
chain state** for that project. The per-attempt RMP trace inside each snippet's
`attempts[*].rmp` is the new, finer-grained source of truth.

---

## 5. Capture mechanism

Three layers, in order of reliability:

### Layer A (floor) — explicit capture in `chains.invoke()` and `RMPChain`

This always works because we already use `include_raw=True` in
`with_structured_output(...)` (`chain.py:42, 50, 59`), which gives us the raw
`AIMessage` regardless of whether the parser succeeded.

In `chains/chain.py`:

- `invoke(chain, code, scope, *, regenerate=False)` already returns
  `(parsed.code, usage)`. Change return signature to also surface a
  `CallTrace` dict containing the rendered messages and raw response. To avoid
  rippling, attach it as a third tuple element OR (cleaner) write it onto a
  caller-supplied `trace_sink` dict. **Choice: trace_sink** because it composes
  with RMP's multi-call shape without nested tuples.

- `RMPChain.invoke()` and `RMPChain._refine_prompt()` similarly accept a
  `trace_sink` and write proposer / refiner / cached-prompt entries into it.

- The rendered system + human strings can be captured by formatting the
  `ChatPromptTemplate` before invocation: `prompt.format_messages(**inputs)` →
  list of `BaseMessage`; we serialize their `.content`.

### Layer B (polish) — `PromptCaptureCallback(BaseCallbackHandler)`

A new file `src/prompt_capture.py` registers a callback that hooks
`on_chat_model_start` / `on_llm_end` / `on_llm_error`. We pass it through
`chain.invoke(inputs, config={"callbacks": [cb], "metadata": {...}})`. The
callback writes one line per LLM call into `src/prompt_log.jsonl` (sidecar,
append-only) for forensic scrolling without re-reading the multi-MB
`results.json`.

This layer is **additive**: even if langchain-ollama's structured-output codepath
fires the callback inconsistently, Layer A already has the data — the callback
log is for human grep convenience, not as the source of truth for
`results.json`.

### Layer C (already exists) — keep `telemetry.track_run()` as-is

The existing context manager (`src/telemetry.py:7-33`) stays unchanged. It feeds
`prompt_tokens` / `completion_tokens` / `total_latency` into the attempt's
`usage` block. Its `get_openai_callback()` integration is retained.

### What gets passed where

```
mainloop.py
  └─ for each snippet:
       └─ for each attempt:
            ├─ trace_sink = {}
            ├─ chains.invoke(chain, code, scope, regenerate=…, trace_sink=trace_sink)
            │       └─ writes exec.system_prompt / exec.human_prompt / exec.raw_response / exec.parsed_code
            │       └─ if RMPChain: also writes rmp.meta_meta_prompt / rmp.proposer_output /
            │            rmp.refinement[*] / rmp.cached_prompt_used
            ├─ try patch.apply_patch() / profile.check_patch()
            ├─ classify outcome → status
            ├─ build AttemptRecord from trace_sink + telemetry stats + outcome
            └─ record["attempts"].append(AttemptRecord)
```

---

## 6. File-by-file changes

### 6.1 `src/prompt_capture.py` (new, ~80 LOC)

- `class PromptCaptureCallback(BaseCallbackHandler)` — callbacks fire on
  `on_chat_model_start`, `on_llm_end`, `on_llm_error`. Each fired event writes
  one line to a JSONL file path passed at construction.
- Helper `serialize_messages(messages: list[BaseMessage]) -> list[dict]` returning
  `[{"role": "system", "content": ...}, {"role": "human", "content": ...}]`.
- Helper `make_attempt_record(trace_sink, telemetry_stats, outcome) -> dict` —
  the canonical builder used by `mainloop.py`.

### 6.2 `src/chains/chain.py` (edit)

- `invoke(chain, code, scope, *, regenerate=False, trace_sink=None)` — render the
  prompt template manually, log into `trace_sink["exec"]`, run the chain, and
  log raw + parsed back into the same dict.
- `RMPChain.invoke(inputs, *, regenerate=False, trace_sink=None)` — pass
  `trace_sink` down to `_refine_prompt()`. After refinement, write
  `trace_sink["rmp"]["cached_prompt_used"] = self._cached_prompt` (post-truncation).
- `RMPChain._refine_prompt(trace_sink=None)` — write proposer's rendered system
  message + output + usage into `trace_sink["rmp"]`. For each iteration, write
  the input `p_current_capped`, the rendered refiner system message
  (`self._refiner_template.format_messages(p_current=p_current_capped)`), the
  refined output, convergence score, usage, and any caught error. The existing
  `self._refinement_trace` field stays (it's used by per-snippet success path
  at `mainloop.py:108-112`); the new structured trace is the trace_sink one.

### 6.3 `src/mainloop.py` (edit — biggest change)

- Replace the inner `try/except Exception` (lines 79-129) with a categorized
  block:
  ```python
  trace_sink: dict = {}
  outcome = {"status": None, "error": None, "patch_applied": False,
             "runtime_after_patch": None, "runtime_diff": None}
  try:
      with telemetry.track_run() as stats:
          optimized, usage = chains.invoke(
              chain, original_code, snippet.scope,
              regenerate=failures > 0, trace_sink=trace_sink,
          )
          if usage:
              stats["prompt_tokens"]     = usage.get("input_tokens", 0)
              stats["completion_tokens"] = usage.get("output_tokens", 0)

      patch = Patch(...)
      if not patch.apply_patch():
          outcome["status"] = "no_op_patch"
      else:
          outcome["patch_applied"] = True
          try:
              new_runtime = profile.check_patch()
          except TestRegressionError as e:
              outcome["status"] = "regression"
              outcome["error"]  = {"type": type(e).__name__, "message": str(e),
                                   "traceback": traceback.format_exc(),
                                   "regression_failures": e.failures}
              patch.revert_patch()
          else:
              outcome["status"] = "success"
              outcome["runtime_after_patch"] = new_runtime
              outcome["runtime_diff"]        = new_runtime - last_runtime
              # ... existing success bookkeeping (record.update, patch_stack.push, etc.)
  except ValidationError as e:                          # pydantic
      outcome["status"] = "parse_error"
      outcome["error"]  = {"type": "ValidationError", "message": str(e),
                           "traceback": traceback.format_exc()}
  except Exception as e:
      outcome["status"] = "llm_error"
      outcome["error"]  = {"type": type(e).__name__, "message": str(e),
                           "traceback": traceback.format_exc()}
      if patch is not None:
          patch.revert_patch()

  attempt = make_attempt_record(
      attempt_idx=failures, regenerate=failures > 0,
      trace_sink=trace_sink, telemetry_stats=stats, outcome=outcome,
  )
  record.setdefault("attempts", []).append(attempt)

  if outcome["status"] == "success":
      success = True
      break
  failures += 1
  ```

- After the retry loop, set `record["final_status"] = "success" if success else "max_retries_exhausted"`.
- After the project loop, `record["snippet_id"]` and `record["scope"]` get
  populated from `snippet`.
- Wire `_save()` to use atomic write: `json.dump` to `RESULTS_PATH.with_suffix(".tmp")`,
  then `os.replace(...)` to the real path. Prevents half-written JSON on Ctrl+C.

### 6.4 `src/chains/__init__.py` (edit)

Re-export the new tracer types if they're imported from `src.chains` elsewhere
(currently nothing imports them externally — leave alone unless needed).

### 6.5 `.gitignore` (edit)

Add `src/prompt_log.jsonl` — the sidecar can grow large; we don't want it in git.
`results.json` stays committed.

---

## 7. Write strategy

- **`results.json`:** keep the existing whole-file rewrite via `_save()`
  (`mainloop.py:16-18`). It runs once per snippet, the file is small enough
  (worst-case ~200 MB only in adversarial scenarios; realistic ~10 MB), and
  whole-file writes are crash-resistant when paired with tempfile-rename.
  Add `os.replace()` for atomicity.
- **`src/prompt_log.jsonl`:** append one line per LLM call, fsync optional.
  This is the high-frequency forensic stream; never rewritten.
- **Encoding:** UTF-8 explicit; `json.dump(..., ensure_ascii=False, indent=2,
  default=str)`. The `default=str` is already there and handles edge cases like
  `Path` objects from `Patch.code_object`.

### Size math (worst case)

- System prompt ≤ 4000 tokens × ~4 chars = 16 KB
- Human prompt ≤ 1500 tokens × ~4 chars = 6 KB
- Raw response ≤ 2048 tokens × ~4 chars = 8 KB
- One exec attempt ≈ 30 KB
- One RMP attempt adds proposer (~2 KB) + 3 refiner iters × 10 KB ≈ 32 KB extra
- Worst-case fully-failed RMP snippet (10 attempts, all max-size): 10 × 62 KB ≈ 620 KB
- 10 projects × 10 snippets × 4 strategies = 400 snippets
- Worst case ~248 MB; realistic case (most succeed in ≤ 2 attempts, RMP only on
  one strategy) ~5–15 MB. Acceptable.

---

## 8. Implementation order

1. `src/prompt_capture.py` — `PromptCaptureCallback` + `serialize_messages` +
   `make_attempt_record`.
2. `src/chains/chain.py` — thread `trace_sink` through `invoke()`, `RMPChain.invoke()`,
   `RMPChain._refine_prompt()`. Render and capture system+human messages.
3. `src/mainloop.py` — categorized exception handling + `record["attempts"]`
   construction + atomic `_save()`.
4. `.gitignore` — add `src/prompt_log.jsonl`.
5. **Smoke test:** run `python main.py` with `repos.json` trimmed to one project
   (e.g. `nanobot`) and `prompt_names = ["base"]` only. Verify:
   - First-attempt successes have `attempts[0].status == "success"` and full
     prompts.
   - Force a parse error by temporarily renaming the `code` field in
     `OptimizedCode` to confirm `status == "parse_error"` and `raw_response` is
     populated.
   - Force a regression by writing a clearly-broken patch to confirm
     `status == "regression"` with `regression_failures` populated.
6. **Full run:** restore full `repos.json` and `prompt_names`. Spot-check the
   resulting `results.json`: every snippet has `attempts`, every failed-snippet
   group has prompts captured, RMP entries have refinement traces.
7. Update `.plans/local-llm-migration.md` (or a new `.plans/prompt-logging.md`)
   with the landed status.

---

## 9. Risks and open questions

| Risk | Mitigation |
|---|---|
| `with_structured_output(method="function_calling")` fires the LangChain callback inconsistently across versions of `langchain-ollama`. | Layer A (explicit capture in `invoke()`) is the source of truth. Callback is sidecar only. |
| Pydantic `ValidationError` is raised inside `with_structured_output`'s parser — the raw `AIMessage` may not be reachable through `result["raw"]` because the parser threw before returning. | Catch `ValidationError` from the chain invoke, then call the LLM directly without the parser to retrieve the offending raw text. Cost: one extra small call per parse failure. Alternative: use `include_raw=True` and `with_structured_output(..., method="json_mode")` which returns the raw even on parse failure — already enabled. Verify before relying. |
| `traceback.format_exc()` inside the `except` may be huge (several KB per regression). | Acceptable — these are diagnostic by design. If it explodes, cap to last N lines. |
| RMP failure where the proposer itself errors (rare): trace_sink may have only `meta_meta_prompt` + an error key. | Schema explicitly allows `proposer_output: null` and an `error` field at the rmp level. Document it. |
| `prompt_log.jsonl` grows unboundedly across runs. | Rotate per-run: name `src/logs/prompts_{run_id}.jsonl` keyed off `main.py`'s existing `datetime.now()` log naming. |
| Existing `results.json` (with old schema) gets overwritten when the next run starts. | Same as today — `_save()` rewrites the whole dict. If preservation matters, the operator can copy `results.json` aside before re-running. (Already true; not a regression.) |
| Graphing in `graphing/` reads keys we move/rename. | We don't move/rename anything. Audit `graphing/` quickly before merge to confirm. |

---

## 10. Out of scope (deliberately)

- Streaming time-to-first-token. Documented as `None` in `telemetry.py:29` —
  fundamentally hard with structured-output + Ollama. Not chasing.
- Replay tooling (re-run a captured prompt against a different model). The data
  is now sufficient to build this later without code changes here.
- Compression / external storage. File sizes are well within local-disk reason.
- Capturing prompts for runs that fail before the first LLM call (baseline
  profile failures at `mainloop.py:34`). Those don't involve the LLM.
- Schema migration tool for old `results.json`. The new schema is additive; old
  files are still readable, just lacking `attempts[]`. No migration needed.

---

## 11. Acceptance checklist (definition of done)

- [ ] `src/prompt_capture.py` exists with `PromptCaptureCallback` and `make_attempt_record`.
- [ ] `chains.invoke()` and `RMPChain` accept and populate `trace_sink`.
- [ ] `mainloop.py` builds `record["attempts"]` for every snippet, including
      max-retries-exhausted snippets.
- [ ] `_save()` writes via tempfile + `os.replace`.
- [ ] `.gitignore` excludes `src/prompt_log.jsonl`.
- [ ] Smoke run on one project produces `attempts[]` arrays containing rendered
      prompts and raw responses for both success and forced-failure cases.
- [ ] `graphing/` runs unmodified against the new `results.json` and produces
      the same charts as before.
- [ ] One representative failed-RMP snippet in the new `results.json` has
      `attempts[k].rmp.refinement[*]` populated for `k` in 0..9.
