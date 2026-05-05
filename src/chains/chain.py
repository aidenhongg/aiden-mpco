import os
import time
from uuid import UUID

from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_community.callbacks import get_openai_callback

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import PROMPTS, rmp_proposer, rmp_refiner, GeneratedPrompt, _HUMAN
from .evaluation import is_converged


class OptimizedCode(BaseModel):
    code: str = Field(description="Return ONLY the optimized code. Include only executable code, and exclude any comments, explanations, markdown formatting, or additional text.")

class RMPChain:
    """Recursive Meta Prompting: generate, refine, execute."""

    MAX_REFINEMENT_ITERS = 3

    def __init__(self, agent_name: str, proj_name: str):
        self._llm = AGENTS[agent_name]()
        self._meta_llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_KEY"])
        self._proposer = rmp_proposer(proj_name, agent_name)
        self._refiner_template = rmp_refiner(proj_name, agent_name)
        self._cached_prompt: str | None = None
        self._refinement_trace: list[dict] = []
        self._meta_meta_prompt: str | None = None
        self._converged: bool = False

    @traceable(name="RMPChain")
    def invoke(self, inputs: dict, *, regenerate: bool = False, langsmith_extra: dict | None = None) -> OptimizedCode:
        if self._cached_prompt is None or regenerate:
            self._refine_prompt()

        prompt = ChatPromptTemplate.from_messages([
            ("system", _escape(self._cached_prompt)),
            ("human", _HUMAN),
        ])
        return (prompt | self._llm.with_structured_output(OptimizedCode)).invoke(inputs)

    @traceable(name="RMPChain.refine_prompt")
    def _refine_prompt(self):
        self._meta_meta_prompt = self._proposer.messages[0].prompt.template
        self._converged = False

        gen_chain = self._proposer | self._meta_llm.with_structured_output(GeneratedPrompt)
        p_current = gen_chain.invoke({}).prompt
        self._refinement_trace = [{"iteration": 0, "prompt": p_current, "convergence_score": None}]

        for i in range(1, self.MAX_REFINEMENT_ITERS + 1):
            try:
                t0 = time.time()
                refine_chain = self._refiner_template | self._meta_llm.with_structured_output(GeneratedPrompt)
                with get_openai_callback() as cb:
                    p_refined = refine_chain.invoke({
                        "p_current": p_current
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
    "o4-mini": lambda: ChatOpenAI(model="o4-mini", api_key=os.environ["OPENAI_KEY"], max_tokens=16192),
    "claude-sonnet-4": lambda: ChatAnthropic(model="claude-sonnet-4-20250514", api_key=os.environ["ANTHROPIC_KEY"], max_tokens=16192),
    "gemini-2.5-pro": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.environ["GEMINI_KEY"], max_output_tokens=16192),
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


def invoke(chain, code: str, scope: list[dict], *, regenerate: bool = False, run_id: UUID | None = None) -> str:
    scope_str = (
        " > ".join(f"{s['type']} {s['name']}" for s in scope)
        if scope else "module-level"
    )
    inputs = {"code": code, "scope": scope_str}
    config = {"run_id": run_id} if run_id else {}
    if isinstance(chain, RMPChain):
        extra = {"run_id": run_id} if run_id else {}
        return chain.invoke(inputs, regenerate=regenerate, langsmith_extra=extra).code
    return chain.invoke(inputs, config=config).code
