import os
from uuid import UUID

from pydantic import BaseModel, Field
from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import PROMPTS, meta_generator, GeneratedPrompt, _HUMAN


class OptimizedCode(BaseModel):
    code: str = Field(description="Return ONLY the optimized code. Include only executable code, and exclude any comments, explanations, markdown formatting, or additional text.")

class MetaChain:
    """Two-stage chain: gpt-4o generates a prompt, then the target agent uses it."""

    def __init__(self, agent_name: str, proj_name: str):
        self._llm = AGENTS[agent_name]()
        self._meta_llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_KEY"])
        self._meta_prompt = meta_generator(proj_name, agent_name)
        self._cached_prompt: str | None = None

    @traceable(name="MetaChain")
    def invoke(self, inputs: dict, *, regenerate: bool = False, langsmith_extra: dict | None = None) -> OptimizedCode:
        if self._cached_prompt is None or regenerate:
            gen_chain = self._meta_prompt | self._meta_llm.with_structured_output(GeneratedPrompt)
            self._cached_prompt = gen_chain.invoke({}).prompt

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._cached_prompt),
            ("human", _HUMAN),
        ])
        return (prompt | self._llm.with_structured_output(OptimizedCode)).invoke(inputs)


AGENTS: dict[str, callable] = {
    "o4-mini": lambda: ChatOpenAI(model="o4-mini", api_key=os.environ["OPENAI_KEY"], max_tokens=16192),
    "claude-sonnet-4": lambda: ChatAnthropic(model="claude-sonnet-4-20250514", api_key=os.environ["ANTHROPIC_KEY"], max_tokens=16192),
    "gemini-2.5-pro": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.environ["GEMINI_KEY"], max_output_tokens=16192),
}


def build_chain(agent_name: str, prompt_name: str, proj_name: str):
    if prompt_name == "meta":
        return MetaChain(agent_name, proj_name)
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
    inputs = {"code": _escape(code), "scope": _escape(scope_str)}
    config = {"run_id": run_id} if run_id else {}
    if isinstance(chain, MetaChain):
        extra = {"run_id": run_id} if run_id else {}
        return chain.invoke(inputs, regenerate=regenerate, langsmith_extra=extra).code
    return chain.invoke(inputs, config=config).code
