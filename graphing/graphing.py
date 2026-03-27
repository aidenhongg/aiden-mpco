import json
from pathlib import Path
from collections import defaultdict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = _DIR / "graphs"
RESULTS_PATH = _DIR.parent / "src" / "results.json"

AGENT_LABELS = {
    "gemini-2.5-pro": "Gemini Pro 2.5",
    "o4-mini": "OpenAI GPT-4o",
    "claude-sonnet-4": "Claude Sonnet 4.0",
}
PROMPT_LABELS = {
    "meta": "Metaprompted",
    "few_shot": "Few-shot",
    "cot": "Chain-of-thought",
    "base": "Base content",
}
AGENT_COLORS = {
    "Gemini Pro 2.5": "#4285F4",
    "OpenAI GPT-4o": "#10A37F",
    "Claude Sonnet 4.0": "#D97706",
}


def _load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _iter_combos(results: dict):
    """Yield (prompt_label, agent_label, proj_data) for valid combos."""
    for combo_key, projects in results.items():
        agent_key, prompt_key = combo_key.split("/", 1)
        agent_label = AGENT_LABELS.get(agent_key)
        prompt_label = PROMPT_LABELS.get(prompt_key)
        if agent_label and prompt_label:
            for proj_data in projects.values():
                yield prompt_label, agent_label, proj_data


def _aggregate_snippets(
    results: dict,
    extract: Callable[[dict], float | None],
    reduce: Callable[[list[float]], float] = np.mean,
) -> dict[tuple[str, str], float]:
    """Collect per-snippet values via `extract`, then `reduce` per (prompt, agent)."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for prompt, agent, proj_data in _iter_combos(results):
        for s in proj_data.get("snippets", []):
            val = extract(s)
            if val is not None:
                buckets[(prompt, agent)].append(val)
    return {k: reduce(v) for k, v in buckets.items()}


def _bar_chart(
    scores: dict[tuple[str, str], float],
    ylabel: str,
    title: str,
    filename: str,
):
    prompts = list(PROMPT_LABELS.values())
    agents = list(AGENT_LABELS.values())
    bar_width = 0.22
    x = np.arange(len(prompts))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        vals = [scores.get((p, agent), 0.0) for p in prompts]
        ax.bar(x + i * bar_width, vals, bar_width,
               label=agent, color=AGENT_COLORS[agent])

    ax.set_xlabel("Prompt Type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(agents) - 1) / 2)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPHS_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Graph definitions
# ---------------------------------------------------------------------------

GRAPHS: list[dict] = [
    {
        "ylabel": "Average % Optimization",
        "title": "Average % Optimization by Prompt Type and Optimizer",
        "filename": "avg_optimization.png",
        "source": "project",  # special: uses project-level data
    },
    {
        "extract": lambda s: s.get("failed_regenerations"),
        "reduce": sum,
        "ylabel": "Total Failed Attempts",
        "title": "Total Failed Attempts by Prompt Type and Optimizer",
        "filename": "total_failed_attempts.png",
    },
    {
        "extract": lambda s: 1 if s.get("failed_regenerations") == 10 else 0,
        "reduce": sum,
        "ylabel": "Total Failed Revisions",
        "title": "Total Failed Revisions by Prompt Type and Optimizer",
        "filename": "total_failed_revisions.png",
    },
    {
        "extract": lambda s: s.get("cognitive_complexity"),
        "ylabel": "Average Cognitive Complexity",
        "title": "Average Cognitive Complexity by Prompt Type and Optimizer",
        "filename": "avg_cognitive_complexity.png",
    },
    {
        "extract": lambda s: s.get("runtime_diff"),
        "ylabel": "Average Runtime Diff (s)",
        "title": "Average Runtime Diff by Prompt Type and Optimizer",
        "filename": "avg_runtime_diff.png",
    },
    {
        "extract": lambda s: s.get("significance"),
        "ylabel": "Average Significance Score",
        "title": "Average Significance Score by Prompt Type and Optimizer",
        "filename": "avg_significance.png",
    },
    {
        "extract": lambda s: s.get("dependency_usage"),
        "ylabel": "Average Dependency Usage Score",
        "title": "Average Dependency Usage Score by Prompt Type and Optimizer",
        "filename": "avg_dependency_usage.png",
    },
    {
        "extract": lambda s: (
            (s["prompt_tokens"] + s["completion_tokens"])
            if s.get("prompt_tokens") is not None and s.get("completion_tokens") is not None
            else None
        ),
        "ylabel": "Average Total Tokens",
        "title": "Average Total Tokens by Prompt Type and Optimizer",
        "filename": "avg_total_tokens.png",
    },
    {
        "extract": lambda s: s.get("total_latency"),
        "ylabel": "Average Total Latency (s)",
        "title": "Average Total Latency by Prompt Type and Optimizer",
        "filename": "avg_total_latency.png",
    },
    {
        "extract": lambda s: s.get("tokens_per_second"),
        "ylabel": "Average Tokens per Second",
        "title": "Average Tokens per Second by Prompt Type and Optimizer",
        "filename": "avg_tokens_per_second.png",
    },
]


def _compute_avg_optimization(results: dict) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for prompt, agent, proj_data in _iter_combos(results):
        start = proj_data.get("start_runtime_avg")
        end = proj_data.get("end_runtime_avg")
        if start and end and start > 0:
            scores[(prompt, agent)].append((start - end) / start * 100)
    return {k: np.mean(v) for k, v in scores.items()}


def plot_all():
    results = _load_results()
    for g in GRAPHS:
        if g.get("source") == "project":
            scores = _compute_avg_optimization(results)
        else:
            scores = _aggregate_snippets(
                results, g["extract"], g.get("reduce", np.mean)
            )
        _bar_chart(scores, g["ylabel"], g["title"], g["filename"])


if __name__ == "__main__":
    plot_all()
