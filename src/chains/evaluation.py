import os
import asyncio
import difflib

from ragas.metrics import SimpleCriteriaScore
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from langchain_openai import ChatOpenAI

_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_KEY"])
)

_METRICS = [
    SimpleCriteriaScore(
        name="cognitive_complexity",
        definition=(
            "Compare the optimized code (response) against the original code (user_input). "
            "Evaluate whether the optimized code maintains or improves cognitive complexity "
            "and maintainability. Consider nesting depth, cyclomatic complexity, readability, "
            "and code clarity. Score 1 if significantly harder to understand, 5 if significantly "
            "more readable and maintainable."
        ),
        llm=_llm,
    ),
    SimpleCriteriaScore(
        name="significance",
        definition=(
            "Compare the optimized code (response) against the original code (user_input). "
            "Evaluate whether the optimization is a significant, meaningful improvement or "
            "merely rewrites the same logic differently. Score 1 if essentially the same logic "
            "rewritten, 5 if it introduces genuinely better algorithms, data structures, or approaches."
        ),
        llm=_llm,
    ),
    SimpleCriteriaScore(
        name="dependency_usage",
        definition=(
            "Compare the optimized code (response) against the original code (user_input). "
            "Evaluate the appropriateness of dependency and library usage. Consider whether "
            "unnecessary dependencies are introduced or existing ones are used more effectively. "
            "Score 1 if inappropriate dependencies are added, 5 if dependency usage is optimal."
        ),
        llm=_llm,
    ),
]


async def _score(original: str, optimized: str) -> dict:
    sample = SingleTurnSample(user_input=original, response=optimized)
    scores = await asyncio.gather(
        *(m.single_turn_ascore(sample) for m in _METRICS)
    )
    return {m.name: s for m, s in zip(_METRICS, scores)}


def score(original: str, optimized: str) -> dict:
    return asyncio.run(_score(original, optimized))


# ---------------------------------------------------------------------------
# IsConverged — judge-independent prompt convergence via difflib
# ---------------------------------------------------------------------------

CONVERGENCE_THRESHOLD = 0.95   # similarity ratio: >= 0.95 means converged


def is_converged(p_current: str, p_refined: str) -> tuple[bool, float]:
    """Return (converged, ratio).

    Convergence is judged by structural similarity (difflib.SequenceMatcher.ratio,
    0.0 = totally different, 1.0 = identical). This implements the paper's
    "no-op detection + max iterations" interpretation (see meta_prompting_paper_analysis.md).
    Judge-independent: no LLM call, deterministic, free.
    """
    if p_current == p_refined:
        return True, 1.0
    ratio = difflib.SequenceMatcher(None, p_current, p_refined).ratio()
    return ratio >= CONVERGENCE_THRESHOLD, ratio
