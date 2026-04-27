# Architecture

This document explains how the system is built, what design decisions were made, and what trade-offs they involve. It complements the [README](../README.md) — read that first if you haven't.

## Table of contents

- [Overview](#overview)
- [State and data flow](#state-and-data-flow)
- [Agents](#agents)
- [LLM client and provider abstraction](#llm-client-and-provider-abstraction)
- [Retrieval (Hybrid RAG)](#retrieval-hybrid-rag)
- [Calibration](#calibration)
- [Error handling and retries](#error-handling-and-retries)
- [Why these choices](#why-these-choices)

## Overview

The system is structured as a **state machine over LLM calls**, implemented with [LangGraph](https://github.com/langchain-ai/langgraph). Each turn of an interview is a single graph invocation:

```
expert  →  manager  →  interviewer  →  END
```

Between graph invocations, the CLI captures user input. The graph itself does not loop — looping is the CLI's responsibility. This separation makes the graph testable in isolation (one invocation at a time) and the CLI simple (a `while` loop).

State is a single `InterviewState` Pydantic model. Each agent reads the full state but **writes** only its specialized fields. LangGraph merges the partial dict each node returns with the existing state.

## State and data flow

```python
class InterviewState(BaseModel):
    candidate: CandidateInfo
    messages: list[Message]
    log: list[TurnLog]
    expert_analysis: ExpertAnalysis | None = None
    manager_decision: ManagerDecision | None = None
    current_turn: int = 1
    is_active: bool = True
```

Three observations matter:

**Pydantic everywhere.** Every structured handoff between agents is a Pydantic model. The expert's output is `ExpertAnalysis`, the manager's is `ManagerDecision`, the final report is `FinalFeedback`. This means:
- LLM responses are validated immediately. A malformed JSON throws at the boundary, not three layers deep.
- The IDE and mypy understand the shape of every field everywhere.
- The same models serve as the prompt's expected output schema (via `model_json_schema()`).

**Messages are append-only.** New messages are added via `[*state.messages, new_msg]`. Pydantic models in this state are not strictly frozen, but the discipline holds — and LangGraph's merge semantics replace the field wholesale anyway.

**Optional fields use `Field(default=...)` everywhere.** Pylance has trouble inferring defaults from positional `Field(...)` calls; explicit keyword arguments avoid spurious "missing argument" errors.

## Agents

Each agent is a **factory function** that returns a node closure. The factory accepts dependencies (LLM client, RAG system); the closure has the LangGraph node signature `(state) -> dict`.

```python
def make_expert_node(llm: LLMClient, rag: RagSystem | None = None):
    def expert_node(state: InterviewState) -> dict[str, ExpertAnalysis]:
        #  use llm, rag, state
        return {"expert_analysis": analysis}
    return expert_node
```

Why factories instead of classes or globals?

- **Globals don't compose.** A test wants to inject a mock LLM. A class would carry per-call state we don't need.
- **Closures are the natural Python idiom** for "function with bound dependencies."
- **The graph node signature is fixed by LangGraph.** A class with `__call__` would work, but adds nothing.

### Expert

Reads the candidate's last answer, optionally retrieves context from the RAG knowledge base, and asks the LLM to evaluate technical correctness, identify knowledge gaps, recommend follow-up questions, and adjust difficulty.

Output: `ExpertAnalysis` (Pydantic).

On the very first turn there's no candidate answer yet — the expert returns an empty dict, leaving `state.expert_analysis` as `None`. This skip is important: running the expert on no data would waste an LLM call and produce hallucinated analysis.

### Manager

Reads the full recent dialogue and the expert's analysis. Decides on:
- Soft skills score (0-10)
- Strategic direction for the next question
- Whether to end the interview

Output: `ManagerDecision`.

The manager's decision is **not** trusted blindly. After the LLM returns:

```python
if state.current_turn < settings.min_turns_before_end:
    should_end = False
if state.current_turn >= settings.max_turns:
    should_end = True

if state.current_turn == 1:
    decision.soft_skills_score = 0
```

These are deterministic policies on top of a non-deterministic decision. This is a recurring pattern in the system: the LLM proposes, the code disposes.

### Interviewer

Reads the recent dialogue, the expert's recommendations, and the manager's strategy. Generates the next question for the candidate.

Output: plain text (not structured — questions are free-form).

There's a defensive `_strip_json_wrapper` post-processing step. Despite explicit "return plain text only" instructions, the model occasionally wraps the question in `{"question": "..."}`, especially when the conversation context is rich with structured outputs from other agents. The wrapper is detected and unwrapped without complaining — the user never sees raw JSON.


### Structured output (`complete_structured`)

Given a Pydantic schema, the client:
1. Builds the JSON schema via `schema.model_json_schema()`.
2. Builds a **concrete example** dict via `_example_from_schema()` — using each field's default, default factory, or a type-appropriate dummy.
3. Sends a prompt that includes both the schema (for structure) and the example (for shape).
4. Receives the response with `response_format={"type": "json_object"}` requested.
5. Validates with `schema.model_validate_json(raw)`.

The example dramatically improves reliability. Without it, the model occasionally returns the JSON Schema itself (an object with `properties`, `required`, etc.) instead of data matching it. With it, this failure mode disappears almost entirely. This is a simple form of [few-shot prompting] tailored to structured output.

A [`field_validator(mode="before")`](https://docs.pydantic.dev/latest/concepts/validators/) on `ManagerDecision.direction` handles the remaining edge case: when the prompt mentions "bullet points" the model sometimes returns a list. The validator coerces lists into newline-separated strings.

## Retrieval (Hybrid RAG)

When CSV knowledge bases are present, retrieval combines three signals:

```
Query
  ├─→ BM25 (lemmatized) ─────┐
  ├─→ Bi-encoder embedding ──┼─→ Score fusion ─→ Top-K candidates ─→ Cross-encoder rerank ─→ Final results
  └──────────────────────────┘
```

**BM25 with lemmatization (rank-bm25 + pymorphy3).** Russian morphology is rich; "программист", "программирование", and "программисту" should match the same lemma. `pymorphy3` reduces them to a single normal form before BM25 indexing. Without lemmatization, BM25 misses ~30% of relevant matches in Russian.

**Bi-encoder (sentence-transformers).** A multilingual encoder produces dense embeddings; cosine similarity in FAISS gives semantic recall.

**Cross-encoder reranker.** Top-K candidates from the lexical+dense fusion are re-scored with a cross-encoder. Cross-encoders see the query and document together (vs encoding them separately), giving much higher precision at the cost of speed. Reranking only the top-K keeps the cost bounded.

This is the **standard production-grade RAG architecture**, sometimes called "hybrid search with reranking." It outperforms pure vector search significantly, especially on technical content where exact terminology matters.

The `Encoder` and `Reranker` are defined as Protocols rather than concrete classes — any object with the right method signatures works. This keeps the retriever testable with mock objects.

## Calibration

Calibration is the most subtle part of the system, and the part with the most failure modes. It addresses three problems:

### Problem 1: Positivity bias

RLHF-trained models default to encouraging language. Without calibration, the system would issue "Strong Hire 90% confidence" for almost any candidate who shows up.

**Fix**: Explicit framing in the final-feedback prompt:

> You are a calibrated technical hiring assessor. Generate a structured final feedback report. Be honest, not encouraging — this is a high-stakes hiring decision, NOT a coaching conversation.

Plus a calibration table:

```
confidence_score (0-100):
  - 90-100: only when 8+ turns and answers are consistently strong/weak
  - 70-89: when 6+ turns and pattern is clear
  - 50-69: when 4-6 turns and signal is mixed or partial
  - 30-49: when 2-4 turns or contradictory evidence
  - 0-29: when fewer than 2 substantive answers — too little data
```

### Problem 2: Behavioral signals get averaged out

The most subtle issue: a candidate gives two strong technical answers, then drifts off-topic. The naive verdict averages everything to "Borderline." But in real interviews, off-topic deflections are **disqualifying**, not just minor.

**Fix**: A separate `behavioral_red_flags: list[str]` field in `FinalFeedback`. The prompt explicitly enumerates what counts (off-topic anecdotes mid-answer, evasions, fundamental errors not recovered after multiple chances). After the LLM responds, code-level logic downgrades the recommendation if any flags are present:

```python
if feedback.behavioral_red_flags and recommendation in (STRONG_HIRE, HIRE):
    feedback.hiring_recommendation = HiringRecommendation.BORDERLINE
```

This is a **gate**, not a soft hint. The technical content can be good; if behavior was unprofessional, the recommendation cannot be Hire.

## Error handling and retries

Three error categories, three responses:

**Network and rate-limit errors → retry with exponential backoff.** A custom `_should_retry` predicate accepts `ConnectionError`, `TimeoutError`, `OSError`, HTTP 429, and HTTP 5xx. Other errors (validation, auth) fall through immediately.

**Logging during retries → custom `before_sleep` callback.** Tenacity's built-in `before_sleep_log` calls loguru with `str.format`-style placeholders. If the exception text contains `{...}` (e.g. a JSON error body), loguru tries to substitute the braces and crashes with `KeyError`. The custom callback uses positional `{}` and passes `repr(exception)` instead of the raw message.

**LLM returns malformed structured output → defense in depth.** Three layers, in order: explicit example in prompt → field validators in schema → post-processing strip in interviewer. Each layer catches a different failure mode.
