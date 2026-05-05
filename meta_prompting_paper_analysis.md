# Meta Prompting: A Framework for Agentic and Compositional Reasoning

**Venue:** Under review at TMLR (anonymous)
**Base model used:** Qwen-72B (base, not instruction-tuned)

---

## Core Idea

Instead of giving an LLM **examples** of solved problems (few-shot), give it a **structural template** describing *how to think* — the reasoning procedure itself. This is "Meta Prompting" (MP). They then extend it to "Recursive Meta Prompting" (RMP), where the LLM generates and refines *its own* meta prompts.

---

## Agent Graph (from Figure 6 + Algorithm 1)

```
┌─────────────────────────────────────────────────────────────┐
│                   RECURSIVE META PROMPTING                  │
│                                                             │
│  ┌──────────┐                                               │
│  │   Task   │ (e.g. "Solve MATH problem X")                 │
│  │   T_0    │                                               │
│  └────┬─────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │  Meta-Meta-     │───▶│  META PROMPT     │                │
│  │  Prompt         │    │  PROPOSER (LLM)  │                │
│  │  (P_meta)       │    │                  │                │
│  │                 │    │  "Given this task │                │
│  │  High-level     │    │   category, design│                │
│  │  instructions   │    │   a structured    │                │
│  │  for HOW to     │    │   prompt..."      │                │
│  │  design prompts │    └────────┬─────────┘                │
│  └─────────────────┘             │                          │
│                                  ▼                          │
│                     ┌────────────────────────┐              │
│                     │  Generated Meta Prompt │              │
│                     │  (P_current)           │              │
│                     └────────────┬───────────┘              │
│                                  │                          │
│              ┌───────────────────┼───────────────┐          │
│              │  REFINEMENT LOOP (up to N iters)  │          │
│              │                                   │          │
│              │   P_refined = LLM(P_meta, P_curr) │          │
│              │                                   │          │
│              │   if IsConverged(P_refined, P_curr)│         │
│              │       break                       │          │
│              │   else                            │          │
│              │       P_curr = P_refined          │          │
│              │       loop                        │          │
│              └───────────────────┬───────────────┘          │
│                                  │                          │
│                                  ▼                          │
│                     ┌────────────────────────┐              │
│                     │  META PROMPT EXECUTOR  │              │
│                     │  (LLM)                 │              │
│                     │                        │              │
│                     │  Takes final refined   │              │
│                     │  meta prompt + task    │              │
│                     │  and solves it         │              │
│                     └────────────┬───────────┘              │
│                                  │                          │
│                                  ▼                          │
│                          ┌──────────────┐                   │
│                          │  Solved Task │                   │
│                          │  T_solved    │                   │
│                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm 1 (RMP) — Pseudocode

```
Input: Task T_0, Meta-Meta-Prompt P_meta, LLM L

P_current ← InitialPrompt(T_0)          # Generate a basic prompt
for i = 1 to N_max_iterations do
    P_refined ← L(P_meta, P_current)    # Refine the prompt
    if IsConverged(P_refined, P_current) then
        break
    end if
    P_current ← P_refined
end for
Solution ← L(P_current, T_0)            # Solve task with final prompt
return Solution
```

#### What is `IsConverged`?

The paper does not specify an explicit implementation — it's a conceptual fixed-point check. In the monadic framework, convergence means the prompt has reached a **fixed point** of the endofunctor `M_P`, i.e., `M_P(P) ≈ P` — refining the prompt no longer changes it meaningfully.

In practice, this could be implemented as any of:

| Strategy | How it works |
|----------|-------------|
| **String/structural diff** | Measure edit distance or structural similarity between `P_refined` and `P_current`. Converged if below a threshold (e.g., <5% token-level change). |
| **Semantic similarity** | Embed both prompts and compute cosine similarity. Converged if above a threshold (e.g., >0.98). |
| **LLM-as-judge** | Ask the LLM itself: *"Has this prompt meaningfully improved? If not, stop."* The refinement call naturally returns the same prompt when it can't improve further. |
| **No-op detection** | If `P_refined == P_current` (exact string match), the LLM has decided it can't improve the prompt and returned it unchanged. |
| **Max iterations fallback** | The outer `for` loop caps at `N_max_iterations` regardless, preventing infinite refinement. |

The most natural implementation (given the paper uses a single LLM for both proposing and refining) is **no-op detection + max iterations**: the LLM is asked to refine `P_current` using `P_meta` as guidance. When the prompt is already good, the LLM returns it unchanged or nearly unchanged, triggering the break. The max iteration cap is the safety net.

---

## The Three Prompt Layers

The system has a **3-level prompt hierarchy**:

| Layer | Name | Role | Who writes it |
|-------|------|------|--------------|
| **L3** | Meta-Meta-Prompt | Instructions for *how to design prompts* | Human (once) |
| **L2** | Meta Prompt | Structural template for *how to solve a task category* | LLM (Proposer) |
| **L1** | Task + Solution | The actual problem and its answer | LLM (Executor) |

---

## Concrete Prompt Examples from the Paper

### 1. Meta-Meta-Prompt (L3) — Figure 7

This is the top-level human-authored prompt that tells the LLM how to *design* meta prompts:

```
Task: Meta Prompting for In-Context Prompt Design

1. Document Analysis:
   - Input: [Complex document (e.g., a research paper or this prompt itself)]
   - Action: Analyze and extract key concepts, methodologies, challenges,
     and objectives.

2. Task Interpretation:
   - Action: Synthesize the extracted information to define the core
     problem or task.
   - Considerations: Identify constraints, goals, or requirements.

3. Prompt Design:
   - Objective: Develop a structured prompt for problem-solving, including
     clear instructions, a step-by-step approach, and relevant background
     information.

4. Optional – Direct Solution Proposal:
   - Objective: Propose initial steps or a complete solution strategy,
     ensuring feasibility and practicality.

5. Output Prompt: [Generate the output prompt using the same LaTeX
   format as this template.]
```

### 2. Meta Prompt for MATH problems (L2) — Figure 1 (JSON format)

```json
{
  "Problem": "[question to be answered]",
  "Solution": {
    "Step 1": "Begin the response with 'Let's think step by step.'",
    "Step 2": "Follow with the reasoning steps, ensuring the solution
               process is broken down clearly and logically.",
    "Step 3": "End the solution with the final answer encapsulated in
               a LaTeX-formatted box, ... for clarity and emphasis."
  },
  "Final Answer": "[final answer to the problem]"
}
```

### 3. Meta Prompt for MATH (L2) — Figure 2 (Markdown format)

```markdown
**Problem Statement:**
- **Problem**: [question to be answered]

**Solution Structure:**
1. Begin the response with "Let's think step by step."
2. Follow with the reasoning steps, ensuring the solution process is
   broken down clearly and logically.
3. End the solution with the final answer encapsulated in a
   LaTeX-formatted box, [...], for clarity and emphasis.
4. Finally, state "The answer is [final answer].", with the final
   answer presented in LaTeX notation.
```

### 4. Meta Prompt for Quadratic Equations (L2) — Figure 4

```json
{
  "Problem": "Solve the quadratic equation ax^2 + bx + c = 0 for x.",
  "Solution": {
    "Step 1": "Identify the coefficients a, b, and c from the equation.",
    "Step 2": "Compute the discriminant using Delta = b^2 - 4ac.",
    "Step 3": "Determine the nature of the roots by checking if Delta > 0,
               Delta = 0, or Delta < 0.",
    "Step 4": "If Delta > 0, calculate the two distinct real roots using
               x_1,2 = (-b +/- sqrt(Delta)) / 2a.",
    "Step 5": "If Delta = 0, calculate the single real root using
               x = -b / 2a.",
    "Step 6": "If Delta < 0, calculate the complex roots using
               x_1,2 = (-b +/- i*sqrt(|Delta|)) / 2a.",
    "Step 7": "Conclude by summarizing the roots in a LaTeX-formatted
               box."
  },
  "Final Answer": "Depending on the value of Delta, the final answer is
                    provided by x_1,2."
}
```

### 5. Meta Prompt for Complex Reasoning (L2) — Figure 9

```markdown
<syntax>

## Problem: [problem]

Solution: Let's think step by step. [initial interpretation]

### Preliminary Content
- **Prelim 1**: [preliminary content 1]
- **Prelim 2**: [preliminary content 2]

### Hints
- **Hint 1**: [useful hint 1]
- **Hint 2**: [useful hint 2]

### Intermediate Steps: Question->Answer, Sketch->Code, Output, Answer Pairs

#### Question 1: [the first sub-question]
- **Answer Sketch**: [sketch of the answer for question 1]

##### Code for Question 1
[execute code interpreter to verify and refine your answer sketch]

#### Answer for Question 1
- **Answer**: [final answer for question 1, based on code results]

#### Question 2: [the second sub-question]
- **Answer Sketch**: [sketch of the answer for question 2]

##### Code for Question 2
[execute code interpreter to verify and refine]

#### Answer for Question 2
- **Answer**: [final answer for question 2]

### Final Solution

Recall the original problem: <MathP> [original problem] </MathP>.

Let's think step by step.

#### Solution Sketch
[provide an overall sketch for the final solution]

#### Code for Final Solution
[execute code interpreter to verify and finalize the solution]

#### Final Answer
[present the final answer in a LaTeX-formatted box]

</syntax>
```

### 6. Generic System Meta Prompt (L2) — Figure 8

```
You are ChatGPT, a state-of-the-art language model with specialized
expertise in mathematics. Your strengths include tackling complex
mathematical challenges using intricate reasoning and delivering
solutions via methodical problem-solving.

Your primary objective is to:
1. Clearly interpret and understand the problem statement.
2. Decompose the problem into manageable components, if necessary.
3. Apply appropriate mathematical principles and techniques to solve
   each component.
4. Synthesize the component solutions into a comprehensive answer.
5. Provide a clear, step-by-step explanation of your methodology,
   ensuring that your reasoning is rigorous, precise, and easily
   understandable.
```

### 7. Game of 24 — User Prompt (Figure 11)

```
User:
Task Step 1: Recall the definition of the Game of 24 (allowed
operations: '+', '-', '*', '/', '(', ')'; note that intermediate
results may be fractional), then provide a detailed plan using
code interpreter to solve the following problem: a, b, c, d
(e.g., 3, 3, 7, 7).

Task Step 2: [uploaded 24.csv] I have a file containing over 1k
Game of 24 puzzles. Please batch-process them. Verify whether the
first five samples are solved correctly, and then compute the overall
success rate.

Task Step 3: Reply with the output file.
Assistant:
[solving the tasks]
```

---

## Key Distinction: Meta Prompting vs Few-Shot

| | Few-Shot | Meta Prompting |
|---|----------|----------------|
| **Gives the model** | Solved examples (content) | A reasoning template (structure) |
| **Teaches** | *What* has been thought | *How* to think |
| **Token cost** | High (full examples) | Low (just the skeleton) |
| **Example-agnostic** | No | Yes |

---

## Results Highlights

| Benchmark | Model | Method | Accuracy |
|-----------|-------|--------|----------|
| MATH | Qwen-72B (base) | Meta Prompting | **46.3%** PASS@1 |
| MATH | GPT-4 (2023-0314) | CoT | 42.5% |
| GSM8K | Qwen-72B (base) | Meta Prompting | **83.5%** |
| Game of 24 | MP-CR Agent | Meta Prompting | **100%** (N=1362) |

The Game of 24 result is particularly striking: the MP-CR Agent used effectively `1/N` LLM sessions (one batch call generating Python code that solves all puzzles), costing ~$0.0003 total vs $0.74 for Tree-of-Thought.

---

## Mathematical Formalization

- **MP as Functor**: `M: T -> P` maps the category of tasks to the category of prompts, preserving composition: `M(g . f) = M(g) . M(f)`
- **RMP as Monad**: The triple `(M_P, eta, mu)` where:
  - `M_P`: endofunctor (refines a prompt into a better prompt)
  - `eta` (unit): lifts a raw task description into a structured meta-prompt
  - `mu` (multiplication/join): flattens nested refinements — a "prompt about refining a prompt" collapses into a single refined prompt
- **Stability guarantee**: The monad's associativity law ensures refinement order doesn't matter for the final result
