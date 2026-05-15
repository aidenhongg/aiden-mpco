import json
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_project_info: dict = json.loads((_PROMPTS_DIR / "project_info.json").read_text())
_considerations: dict = json.loads((_PROMPTS_DIR / "considerations.json").read_text())
_task_info: dict = json.loads((_PROMPTS_DIR / "task_info.json").read_text())


class GeneratedPrompt(BaseModel):
    prompt: str = Field(description="A system prompt that instructs an LLM to optimize Python code for runtime. Plain text, no markdown.")

def _context(proj_name: str, agent_name: str) -> dict:
    proj = _project_info.get(proj_name, {})
    return dict(
        objective=_task_info["objective"],
        p_name=proj_name,
        p_desc=proj.get("description", ""),
        p_lang=proj.get("languages", ""),
        t_desc=_task_info["description"],
        t_cons=_task_info["considerations"],
        llm_name=agent_name,
        llm_cons=_considerations.get(agent_name, ""),
    )

def _base_system(objective, p_name, p_desc, p_lang,
                 t_desc, t_cons, llm_name, llm_cons) -> str:
    return dedent(f"""\
        Optimize the given code for {objective}.

        Project: {p_name} ({p_lang}) — {p_desc}
        Task: {t_desc}
        Optimization levers: {t_cons}
        Model ({llm_name}) notes: {llm_cons}

        Output: only the optimized code, preserving the original signature. No markdown, comments, or prose.""")


_HUMAN = "Scope: {scope}\nCode:\n{code}"

def _base_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", _HUMAN),
    ])


def _few_shot_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    few_shot = dedent("""\
        Examples (Original -> Optimized):

        1) Loop:
        for i in range(len(arr)):
            if arr[i] > threshold: result.append(arr[i])
        ->
        result = [x for x in arr if x > threshold]

        2) Algorithm:
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0: count += 1
        ->
        count = np.sum(matrix > 0)

        3) Data structure:
        items = []
        for x in data: items.append(x)
        return sorted(items)
        ->
        return sorted(data)

        Apply the same pattern. Emit only the optimized code.""")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", few_shot + "\n\n" + _HUMAN),
    ])


def _cot_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    cot = dedent("""\
        Before emitting code, work through these steps internally (do not write them out):
        1. Locate the bottleneck (loops, redundant work, bad data structure, O(n^2) where O(n) exists).
        2. Pick one optimization (vectorization, comprehension, builtin, better structure, hoisted invariant).
        3. Verify the rewrite preserves the signature and behavior.

        Emit only the optimized code. No reasoning, no steps, no prose.""")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", cot + "\n\n" + _HUMAN),
    ])


def rmp_proposer(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    """L3 meta-meta-prompt: instructs the proposer LLM HOW to design an optimization prompt."""
    ctx = _context(proj_name, agent_name)
    system = dedent(f"""\
        Write a system prompt that instructs an LLM to optimize Python functions for {ctx['objective']} while preserving the original signature and behavior.

        Inputs to weave in:
        - Project: {ctx['p_name']} — {ctx['p_desc']} (langs: {ctx['p_lang']})
        - Target model: {ctx['llm_name']}; constraints: {ctx['llm_cons']}
        - Optimization levers to mention: {ctx['t_cons']}

        The prompt you produce must contain these sections, in order, each on its own line(s):
        Objective: <one sentence stating the goal>
        Steps:
        1) Identify the runtime bottleneck.
        2) Apply the strongest applicable optimization (vectorization, comprehension, builtin, better data structure, reduced complexity).
        3) Preserve the function signature and external behavior.
        Output: only the optimized code, no markdown, no comments, no prose.

        Keep the prompt under 200 words. Make it general for any Python function in this project, not tied to one snippet. Emit only the prompt text.""")
    # A human turn is required: without one, Qwen3.5-9B-Q4 emits the answer as
    # plain assistant text and never produces the tool_call that
    # with_structured_output(method="function_calling") expects, leaving
    # parsed=None at chain.py:52 and breaking every RMP retry.
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Generate the optimization prompt now."),
    ])


def rmp_refiner(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    """Refinement prompt: asks the refiner LLM to verify or edit an existing optimization prompt."""
    ctx = _context(proj_name, agent_name)
    system = dedent(f"""\
        Verify the prompt below against this checklist:
        [A] Has an "Objective:" line naming runtime/{ctx['objective']} as the goal.
        [B] Has a numbered "Steps:" list (>= 2 steps) covering bottleneck identification and at least one concrete optimization technique.
        [C] Has an "Output:" line restricting output to code only with no markdown or prose.
        [D] Mentions preserving the original function signature/behavior.
        [E] Is general for any Python function in {ctx['p_name']}, not tied to a single snippet.

        Decision rule:
        - If every item [A]-[E] is present and concrete, return the prompt UNCHANGED, byte-for-byte.
        - Otherwise, return a revised prompt that adds the missing items, keeping all existing concrete content.

        Do not add prose around the prompt. Do not wrap it in markdown. Emit only the prompt text.

        Prompt to verify:
        {{p_current}}""")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Verify or revise the prompt now per the checklist."),
    ])

PROMPTS: dict[str, callable] = {
    "base": _base_prompt,
    "few_shot": _few_shot_prompt,
    "cot": _cot_prompt,
}
