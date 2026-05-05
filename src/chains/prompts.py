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
    prompt: str = Field(description="The optimization system prompt to send to the agent")

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
        You are an expert in code optimization. Please optimize the provided code for {objective}. Consider the project context, task context, and adapt your optimization approach accordingly.

        ## Project Context
        Project Name: {p_name}
        Project Description: {p_desc}
        Primary Languages: {p_lang}

        ## Task Context
        - Description: {t_desc}
        - Considerations: {t_cons}

        ## Target LLM Context
        - Target Model: {llm_name}
        - Considerations: {llm_cons}""")


_HUMAN = "Enclosing scope: {scope}\n\nObject to be optimized: {code}"

def _base_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", _HUMAN),
    ])


def _few_shot_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    few_shot = dedent("""\
        Here are examples of code optimization:
        Example 1 - Loop optimization:
        Original: for i in range(len(arr)): if arr[i] > threshold: result.append(arr[i])
        Optimized: result = [x for x in arr if x > threshold]

        Example 2 - Algorithm optimization:
        Original: for i in range(n): for j in range(n): if matrix[i][j] > 0: count += 1
        Optimized: count = np.sum(matrix > 0)

        Example 3 - Data structure optimization:
        Original: items = []; for x in data: items.append(x); return sorted(items)
        Optimized: return sorted(data)

        Now optimize the code for better runtime performance, then provide only the final optimized code.""")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", few_shot + "\n\n" + _HUMAN),
    ])


def _cot_prompt(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    system = _base_system(**_context(proj_name, agent_name))
    cot = dedent("""\
        Let's optimize the following code step by step:

        Please follow these reasoning steps:
        1. First, analyze the current code to identify performance bottlenecks
        2. Consider different optimization strategies (algorithmic, data structure, loop optimization, etc.)
        3. Evaluate the trade-offs of each approach
        4. Select the best optimization strategy
        5. Implement the optimized version

        Think through each step, then provide only the final optimized code.""")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", cot + "\n\n" + _HUMAN),
    ])


def rmp_proposer(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    """L3 meta-meta-prompt: instructs GPT-4o HOW to design optimization prompts."""
    ctx = _context(proj_name, agent_name)
    system = dedent(f"""\
        Task: Design a Code Optimization Prompt

        1. Context Analysis:
           - Project: {ctx['p_name']} — {ctx['p_desc']}
           - Languages: {ctx['p_lang']}
           - Target model: {ctx['llm_name']}
           - Objective: {ctx['objective']}

        2. Task Interpretation:
           - The prompt must instruct the target LLM to optimize Python functions
             for runtime performance while preserving correctness.
           - Considerations: {ctx['t_cons']}
           - Model-specific considerations: {ctx['llm_cons']}

        3. Prompt Design:
           - Design a structured system prompt for the target LLM.
           - Include: optimization objectives, step-by-step reasoning guidance,
             relevant techniques (algorithmic complexity, data structures,
             Python-specific optimizations like list comprehensions, generators,
             built-in functions, numpy vectorization).
           - The prompt should be general-purpose for any Python function in
             this project, not tied to a specific code snippet.

        4. Output: Generate the system prompt.""")
    return ChatPromptTemplate.from_messages([("system", system)])


def rmp_refiner(proj_name: str, agent_name: str) -> ChatPromptTemplate:
    """Refinement prompt: asks GPT-4o to improve an existing optimization prompt."""
    ctx = _context(proj_name, agent_name)
    system = dedent(f"""\
        You are refining a code optimization prompt. Your goal is to make it more
        specific, structured, and effective for the target model.

        Project: {ctx['p_name']} — {ctx['p_desc']}
        Target model: {ctx['llm_name']}
        Objective: {ctx['objective']}

        Review the current prompt below and refine it. Consider:
        - Is the reasoning structure clear and actionable?
        - Are optimization techniques specific enough for Python?
        - Does it account for the target model's strengths and tendencies?
        - Is the output format unambiguous?

        If the prompt is already optimal, return it unchanged.

        Current prompt to refine:
        {{p_current}}""")
    return ChatPromptTemplate.from_messages([("system", system)])

PROMPTS: dict[str, callable] = {
    "base": _base_prompt,
    "few_shot": _few_shot_prompt,
    "cot": _cot_prompt,
}
