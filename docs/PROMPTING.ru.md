# Prompt engineering

> [English version](PROMPTING.md)

Документ описывает, как структурированы промпты, какие техники используются и какие failure modes каждая техника решает.

Три промпта агентов лежат в `src/multiagent_interviewer/prompts/`:
- `expert.j2` — анализирует последний ответ кандидата
- `manager.j2` — стратегические решения
- `interviewer.j2` — формулирует следующий вопрос

Промпт финального фидбэка строится в `src/multiagent_interviewer/feedback.py`.

## Почему Jinja2-шаблоны, а не f-strings

f-strings хороши для коротких промптов, но разваливаются, как только нужно:
- Условные секции (`{% if last_answer %} ... {% else %} ... {% endif %}`)
- Циклы по сообщениям
- Удаление пробелов на границах шаблона (`{%- ... -%}`)
- Ловить отсутствующие переменные до отправки (с `StrictUndefined`)

## Техника structured output

Для Expert и Manager LLM должен вернуть JSON, соответствующий Pydantic-схеме. В промпт идут три вещи:

1. Сама инструкция ("проанализируй последний ответ кандидата на техническую корректность").
2. JSON Schema, генерируемая автоматически из Pydantic-модели через `schema.model_json_schema()`.
3. **Конкретный пример** валидного output, построенный динамически из `model_fields`.

Пример важен. Без него модель иногда возвращает саму JSON Schema — объект с `properties`, `required` и т.д. — вместо данных, соответствующих ей. Пример убирает неоднозначность:

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

Динамическое построение примера (вместо хардкода под каждую схему) означает, что он автоматически синхронизируется со схемой. См. `_example_from_schema` в `llm/client.py`.

## Защита от типичных failure modes

### "Вы упомянули X" на первом ходу

**Симптом**: Интервьюер говорит "Вы упомянули CRUD" на ходу 1, когда кандидат ещё ничего не сказал.

**Причина**: Поле `direction` менеджера иногда содержит полностью сформулированный пример вопроса. Интервьюер читает это как часть разговора и пытается "продолжить" с него.

**Фикс** (в `interviewer.j2`):
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

Фрейминг "internal guidance — do NOT echo" говорит модели, что эти поля для **планирования**, а не для **контента**. Это значительно снижает проблему.

### Менеджер генерирует готовые вопросы вместо стратегии

**Симптом**: `direction` содержит "Кирилл, расскажите о...". Это работает как вопрос, но превращает промпт интервьюера в "задай этот вопрос, который ты уже задал".

**Причина**: Изначальный промпт просил "guidance for the interviewer" без уточнения, в какой форме это guidance должно быть.

**Фикс** (в `manager.j2`):
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

Строка "do NOT use bullet points or arrays" важна: ранняя версия с просьбой "2-3 bullet points" заставляла модель возвращать JSON-массив, ломая валидацию. (`field_validator(mode="before")` был добавлен как страховка.)

### Plain-text ответ обёрнут в JSON

**Симптом**: Вопрос интервьюера приходит как `{"question": "..."}` вместо plain text.

**Причина**: Окружённая structured outputs от Expert и Manager модель "имитирует" структурированный паттерн.

**Фикс**: Два слоя.

В промпте:
```jinja
Respond with ONLY the question text for the candidate as plain Russian text.
Do NOT use JSON, markdown, code blocks, or any structured format.
Do NOT prefix your response with labels like "Question:" or "Interviewer:".
Just the question itself.
```

В коде (`agents/interviewer.py`):
```python
def _strip_json_wrapper(text: str) -> str:
    """Если LLM вернул JSON несмотря на инструкции, извлекаем
    текст сообщения. Возвращает вход без изменений, если это не JSON."""
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

## Калибровка в финальном фидбэке

Промпт финального фидбэка — самый длинный. Вот его основа:

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

Три наблюдения:

**Статистика передаётся внутрь.** Модель не просят считать ходы или пробелы — это вычисляется в Python и вставляется в промпт. Это не даёт модели обсчитываться.

**Калибровочная таблица явная.** "Насколько я должен быть уверен?" — это калибровочный вопрос. Промпт отвечает на него напрямую количественными порогами, привязанными к объёму данных.

**Дисквалифицирующие сигналы фреймятся как override-ы.** Большая часть промпта просит модель взвешивать данные; этот раздел говорит "любой одиночный из них — этого достаточно".

## Что бы я сделал со временем

Несколько паттернов, которые стоит применить в крупных системах:

**Версионирование промптов.** Каждый промпт агента должен иметь версию (`expert.v3.j2`). Когда меняешь промпт, бамп версии сохраняет историю. Это становится критичным при A/B-тестах изменений промптов на eval-наборе.

**Prompt-as-config.** Вынести промпты в YAML/JSON, если их нужно тюнить без редеплоя. Для этого проекта Jinja2-файлы в репо — trade-off в пользу воспроизводимости против гибкости.

**Пул few-shot примеров.** Сейчас пример для structured output генерируется автоматически из схемы (zero-shot). Реальные production-системы поддерживают небольшой пул высококачественных пар input/output и подбирают самые релевантные на этапе формирования промпта. Это обычно вдвое снижает error rate на edge cases.

Это вне scope этой версии, но в [Roadmap](../README.ru.md#roadmap).
