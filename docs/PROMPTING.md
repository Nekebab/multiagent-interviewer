# Prompt engineering

This document describes how prompts are structured, what techniques are used, and which failure modes each technique addresses.

The three agent prompts live in `src/multiagent_interviewer/prompts/`:
- `expert.j2` — analyzes the candidate's last answer
- `manager.j2` — strategic decisions
- `interviewer.j2` — formulates the next question

Plus the final-feedback prompt is built in `src/multiagent_interviewer/feedback.py`.

## Why Jinja2 templates and not f-strings

f-strings are tempting for short prompts but fall apart as soon as you need:
- Conditional sections (`{% if last_answer %} ... {% else %} ... {% endif %}`)
- Loops over messages
- Stripping whitespace at template boundaries (`{%- ... -%}`)
- Catching missing variables before send (with `StrictUndefined`)

Jinja2 with `StrictUndefined` is non-negotiable — it raises `UndefinedError` at render time if you forget to pass a variable, instead of silently substituting an empty string. We've all seen LLMs respond to "Hello { } I am a ..." with confused output. `StrictUndefined` makes this impossible.

## Structured output technique

For Expert and Manager, the LLM must return JSON matching a Pydantic schema. Three things go into the prompt:

1. The actual instruction ("analyze the candidate's last answer for technical correctness").
2. The JSON Schema, generated automatically from the Pydantic model via `schema.model_json_schema()`.
3. A **concrete example** of valid output, built dynamically from `model_fields`.

The example matters. Without it, the model occasionally returns the schema itself — an object with `properties`, `required`, etc. — instead of data matching it. The example removes ambiguity:

```
SCHEMA (for reference):
{
  "type": "object",
  "properties": {
    "technical_correctness": {"type": "string"},
    "knowledge_gaps": {"type": "array", "items": {"type": "string"}},
    ...
  },
  "required": ["technical_correctness", ...]
}

EXAMPLE (the format you must return — replace values with real ones):
{
  "technical_correctness": "...",
  "knowledge_gaps": [],
  "recommended_follow_ups": [],
  "difficulty_adjustment": "same"
}
```

Building the example dynamically (rather than hard-coding it per schema) means it stays in sync with the schema automatically. See `_example_from_schema` in `llm/client.py`.

## Defense against common failure modes

### "You mentioned X" on the first turn

**Symptom**: The interviewer says "You mentioned CRUD" on turn 1, when the candidate hasn't said anything yet.

**Cause**: The manager's `direction` field sometimes contains a fully-formed sample question. The interviewer reads this as part of the conversation and tries to "continue" from it.

**Fix** (in `interviewer.j2`):
```jinja
{% if last_answer -%}
CANDIDATE'S LAST ANSWER:
"{{ last_answer }}"
{%- else -%}
=== THIS IS THE FIRST QUESTION OF THE INTERVIEW ===
The candidate has NOT said anything yet.
You must NOT reference any previous statements from them.
You must NOT use phrases like "you mentioned", "you said", "вы упомянули", ...
{%- endif %}

EXPERT'S NOTES (your internal guidance — do NOT echo these to the candidate):
{{ expert_recommendations }}

MANAGER'S STRATEGY (your internal guidance — do NOT echo these to the candidate):
{{ manager_direction }}

NEVER reference instructions from expert/manager as if the candidate said them.
NEVER pretend the candidate already mentioned a topic if they haven't.
```


### Manager generating sample questions instead of strategy

**Symptom**: `direction` contains "Кирилл, расскажите о...". This works as a question, but turns the interviewer's prompt into "ask this question that you already asked."

**Cause**: The original prompt asked for "guidance for the interviewer" without specifying what form that guidance should take.

**Fix** (in `manager.j2`):
```jinja
CRITICAL RULES for the `direction` field:
- Describe the STRATEGY, not the exact wording of a question.
- Do NOT write any sentence that starts with "Кирилл,..." or addresses the
  candidate directly. The interviewer will phrase the actual question.
- Use phrases like "ask about X", "explore the topic of Y", "verify
  understanding of Z" — not "Расскажите...", "Объясните...", "Как бы вы...".
- Keep the strategy concise — 2-3 sentences, written as one continuous string.
- Do NOT use bullet points or arrays in the JSON output.
```

The "do NOT use bullet points or arrays" line is important: an earlier version asking for "2-3 bullet points" caused the model to return a JSON array, breaking validation. (A `field_validator(mode="before")` was added as a safety net.)

### Plain-text response wrapped in JSON

**Symptom**: The interviewer's question arrives as `{"question": "..."}` instead of plain text.

**Cause**: Surrounded by structured outputs from Expert and Manager, the model "imitates" the structured pattern.

**Fix**: Two layers.

In the prompt:
```jinja
Respond with ONLY the question text for the candidate as plain Russian text.
Do NOT use JSON, markdown, code blocks, or any structured format.
Do NOT prefix your response with labels like "Question:" or "Interviewer:".
Just the question itself.
```

In the code (`agents/interviewer.py`):
```python
def _strip_json_wrapper(text: str) -> str:
    """If the LLM returned JSON despite our instructions, extract the
    message text. Returns the input unchanged if it isn't JSON."""
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    for key in ("question", "message", "content", "text", "response"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            return value
    return text
```

## Calibration in the final feedback

The final-feedback prompt is the longest, because it has the most policy. Here's the core of it:

```
You are a calibrated technical hiring assessor. Generate a structured final
feedback report. Be honest, not encouraging — this is a high-stakes hiring
decision, NOT a coaching conversation.

INTERVIEW STATISTICS:
- Total turns conducted: {{ current_turn - 1 }}
- Turns with candidate answers: {{ answered_turns }}
- Knowledge gaps flagged by expert (cumulative): {{ flagged_gaps }}
- Turns with fact errors flagged by expert: {{ fact_errors }}

CALIBRATION RULES — FOLLOW STRICTLY:

confidence_score (0-100) — how SURE you are in your verdict:
  - 90-100: only when 8+ turns and answers are consistently strong/weak
  - 70-89: when 6+ turns and pattern is clear
  - 50-69: when 4-6 turns and signal is mixed or partial
  - 30-49: when 2-4 turns or contradictory evidence
  - 0-29: when fewer than 2 substantive answers — too little data

KEY HEURISTICS:
  - If the candidate said "I don't know" or "не помню" on basic questions for
  their stated level → cannot be Strong Hire, likely Borderline or below.
  - If the expert flagged factual errors in basics → recommendation cannot be
  above Borderline.
  - If interview was cut short ({state.current_turn - 1} turns), confidence
  should be lower; do not pretend you have full picture.
  - If candidate's experience claim seems inflated relative to demonstrated
  knowledge → flag in soft_skills_summary, lower confidence.
  - Don't pad confirmed_skills with generic items — only list skills the
  candidate actually demonstrated in answers (with concrete evidence).

  DISQUALIFYING SIGNALS (any single one of these → recommendation cannot be
  above No Hire, regardless of how strong other answers were):
  - Off-topic personal anecdotes inserted into a technical answer
  (e.g. "I baked bread last week" mid-explanation)
  - Avoiding a direct question with a deflection ("I missed it, repeat please"
  after going off on a tangent)
  - Confused fundamental concepts (mistaking variance for mean, etc.)
  after multiple chances to clarify
  - Dishonest or evasive behavior
```

Three observations:

**Statistics are passed in.** The model isn't asked to count turns or gaps — those are computed in Python and inserted into the prompt. This keeps the model from miscounting.

**Calibration table is explicit.** "How sure should I be?" is a calibration question. The prompt answers it directly with quantitative thresholds tied to data volume.

**Disqualifying signals are framed as overrides.** Most of the prompt asks the model to weigh evidence; this section says "any of these alone is enough." This matches how human interviewers actually work — a single off-topic deflection is a hard "no" regardless of technical knowledge.

## What I'd do differently with more time

A few patterns worth adopting in larger systems:

**Versioned prompts.** Each agent prompt should be stamped with a version (`expert.v3.j2`). When you change a prompt, bumping the version preserves a record. This becomes critical when running A/B tests on prompt changes against an eval set.

**Prompt-as-config.** Externalize prompts to YAML/JSON if they need to be tuned without redeploying. For this project, having them as Jinja2 files in the repo is fine — the tradeoff is reproducibility (everything in git) over flexibility.

**Prompt eval framework.** Synthetic interview transcripts paired with expected verdicts. Run them through the system, measure agreement. Without this, prompt changes are guesswork.

**Few-shot example pool.** Currently the structured-output example is auto-generated from the schema (zero-shot). Real production systems maintain a small pool of high-quality input/output pairs and pick the most relevant ones at prompt time. This typically halves the error rate on edge cases.

These are out of scope for this version, but on the [Roadmap](../README.md#roadmap).
