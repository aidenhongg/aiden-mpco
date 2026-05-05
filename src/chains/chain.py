import time

from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback

from langchain_core.prompts import ChatPromptTemplate

from ..llm import local_llm, MAX_INPUT_TOKENS
from .prompts import PROMPTS, rmp_proposer, rmp_refiner, GeneratedPrompt, _HUMAN
from .evaluation import is_converged
from .truncate import truncate_head, truncate_tail


class OptimizedCode(BaseModel):
    code: str = Field(description="Return ONLY the optimized code. Include only executable code, and exclude any comments, explanations, markdown formatting, or additional text.")


class RMPChain:
    """Recursive Meta Prompting: generate, refine, execute."""

    MAX_REFINEMENT_ITERS = 3

    def __init__(self, agent_name: str, proj_name: str):
        self._llm = self._meta_llm = local_llm()
        self._proposer = rmp_proposer(proj_name, agent_name)
        self._refiner_template = rmp_refiner(proj_name, agent_name)
        self._cached_prompt: str | None = None
        self._refinement_trace: list[dict] = []
        self._meta_meta_prompt: str | None = None
        self._converged: bool = False

    def invoke(self, inputs: dict, *, regenerate: bool = False) -> OptimizedCode:
        if self._cached_prompt is None or regenerate:
            self._refine_prompt()

        capped = truncate_head(self._cached_prompt, MAX_INPUT_TOKENS, label="rmp.cached_prompt")
        prompt = ChatPromptTemplate.from_messages([
            ("system", _escape(capped)),
            ("human", _HUMAN),
        ])
        return (prompt | self._llm.with_structured_output(OptimizedCode)).invoke(inputs)

    def _refine_prompt(self):
        self._meta_meta_prompt = self._proposer.messages[0].prompt.template
        self._converged = False

        gen_chain = self._proposer | self._meta_llm.with_structured_output(GeneratedPrompt)
        p_current = gen_chain.invoke({}).prompt
        self._refinement_trace = [{"iteration": 0, "prompt": p_current, "convergence_score": None}]

        for i in range(1, self.MAX_REFINEMENT_ITERS + 1):
            try:
                t0 = time.time()
                p_current_capped = truncate_head(p_current, MAX_INPUT_TOKENS, label=f"rmp.refiner.iter{i}")
                refine_chain = self._refiner_template | self._meta_llm.with_structured_output(GeneratedPrompt)
                with get_openai_callback() as cb:
                    p_refined = refine_chain.invoke({
                        "p_current": p_current_capped
                    }).prompt
                refine_latency = time.time() - t0

                converged, conv_score = is_converged(p_current, p_refined)
                self._refinement_trace.append({
                    "iteration": i,
                    "prompt": p_refined,
                    "convergence_score": conv_score,
                    "refine_prompt_tokens": cb.prompt_tokens,
                    "refine_completion_tokens": cb.completion_tokens,
                    "refine_latency": refine_latency,
                })

                if converged:
                    p_current = p_refined
                    self._converged = True
                    break
                p_current = p_refined
            except Exception as e:
                print(f"Refinement iteration {i} failed: {e}")
                self._refinement_trace.append({
                    "iteration": i,
                    "prompt": None,
                    "convergence_score": None,
                    "error": str(e),
                })
                break

        self._cached_prompt = p_current


AGENTS: dict[str, callable] = {
    "qwen3.5-9b-q4": local_llm,
}


def build_chain(agent_name: str, prompt_name: str, proj_name: str):
    if prompt_name == "rmp":
        return RMPChain(agent_name, proj_name)
    llm = AGENTS[agent_name]()
    prompt = PROMPTS[prompt_name](proj_name, agent_name)
    return prompt | llm.with_structured_output(OptimizedCode)


def _escape(text: str) -> str:
    """Escape curly braces so LangChain doesn't treat them as template vars."""
    return text.replace("{", "{{").replace("}", "}}")


def invoke(chain, code: str, scope: list[dict], *, regenerate: bool = False) -> str:
    scope_str = (
        " > ".join(f"{s['type']} {s['name']}" for s in scope)
        if scope else "module-level"
    )
    code = truncate_tail(code, MAX_INPUT_TOKENS, label="invoke.code")
    inputs = {"code": code, "scope": scope_str}
    if isinstance(chain, RMPChain):
        return chain.invoke(inputs, regenerate=regenerate).code
    return chain.invoke(inputs).code
