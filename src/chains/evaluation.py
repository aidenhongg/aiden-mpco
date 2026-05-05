import os
import asyncio

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
# IsConverged — Ragas-based semantic difference detection for RMP
# ---------------------------------------------------------------------------

_convergence_metric = SimpleCriteriaScore(
    name="prompt_difference",
    definition=(
        "Compare the refined prompt (response) against the previous prompt (user_input). "
        "Evaluate how DIFFERENT they are in structure, content, specificity, and reasoning "
        "guidance. Score 1 if the prompts are essentially identical or nearly unchanged. "
        "Score 5 if they are substantially different in structure, approach, or content."
    ),
    llm=_llm,
)

CONVERGENCE_THRESHOLD = 2.0


async def _is_converged(p_current: str, p_refined: str) -> tuple[bool, float]:
    if p_current == p_refined:
        return True, 0.0
    sample = SingleTurnSample(user_input=p_current, response=p_refined)
    score = await _convergence_metric.single_turn_ascore(sample)
    return score <= CONVERGENCE_THRESHOLD, score


def is_converged(p_current: str, p_refined: str) -> tuple[bool, float]:
    return asyncio.run(_is_converged(p_current, p_refined))
