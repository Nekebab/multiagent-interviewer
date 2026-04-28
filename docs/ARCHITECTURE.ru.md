# Архитектура

> [English version](ARCHITECTURE.md)

Документ описывает, как устроена система, какие были приняты решения и какие у них trade-offs. Это дополнение к [README](../README.ru.md).

## Содержание

- [Обзор](#обзор)
- [State и поток данных](#state-и-поток-данных)
- [Агенты](#агенты)
- [Retrieval (Hybrid RAG)](#retrieval-hybrid-rag)
- [Калибровка](#калибровка)
- [Обработка ошибок и retry](#обработка-ошибок-и-retry)

## Обзор

Система устроена как **state machine поверх LLM-вызовов**, реализованная на [LangGraph](https://github.com/langchain-ai/langgraph). Каждый ход интервью — одно обращение к графу:

```
expert  →  manager  →  interviewer  →  END
```

Между обращениями к графу CLI собирает ввод пользователя. Сам граф **не зацикливается** — цикл это ответственность CLI. Это разделение делает граф тестируемым в изоляции (по одному вызову за раз), а CLI — простым (просто `while`-цикл).

State — это одна Pydantic-модель `InterviewState`. Каждый агент читает весь state, но **пишет** только в свои специализированные поля. LangGraph мёрджит частичный dict, который возвращает каждый узел, с существующим state.

## State и поток данных

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

Три важных наблюдения:

**Pydantic везде.** Каждая структурированная передача данных между агентами — Pydantic-модель. Output эксперта — `ExpertAnalysis`, менеджера — `ManagerDecision`, финальный отчёт — `FinalFeedback`. Это значит:
- Ответы LLM валидируются немедленно. Битый JSON падает сразу, а не тремя слоями глубже.
- IDE и mypy понимают форму каждого поля везде.
- Те же модели используются как expected output schema для промпта (через `model_json_schema()`).

**Сообщения append-only.** Новые сообщения добавляются через `[*state.messages, new_msg]`. Pydantic-модели в state не строго иммутабельны, но структура соблюдается — а merge-семантика LangGraph всё равно заменяет поле целиком.

**Опциональные поля используют `Field(default=...)` везде.** У Pylance проблемы с интерпретацией defaults через позиционные аргументы `Field(...)`; явные keyword-аргументы устраняют ложные ошибки "missing argument".

## Агенты

Каждый агент — это **factory-функция**, возвращающая closure-узел. Factory принимает зависимости (LLM-клиент, RAG-систему); closure имеет сигнатуру узла LangGraph: `(state) -> dict`.

```python
def make_expert_node(llm: LLMClient, rag: RagSystem | None = None):
    def expert_node(state: InterviewState) -> dict[str, ExpertAnalysis]:
        # ... используем llm, rag, state ...
        return {"expert_analysis": analysis}
    return expert_node
```

Почему factories вместо классов или глобальных переменных?

- **Глобальные переменные не композируются.** Тест хочет инжектить mock LLM. Класс носил бы per-call state, который нам не нужен.
- **Closure — естественная Python-идиома** для "функция со связанными зависимостями".
- **Сигнатура узла графа фиксирована LangGraph.** Класс с `__call__` сработал бы, но ничего не добавляет.

### Expert

Читает последний ответ кандидата, опционально извлекает контекст из RAG-базы знаний и просит LLM оценить техническую корректность, выявить пробелы в знаниях, предложить follow-up вопросы и скорректировать сложность.

### Manager

Читает полный недавний диалог и анализ эксперта. Решает:
- Soft skills score (0-10)
- Стратегическое направление следующего вопроса
- Завершать ли интервью

Output: `ManagerDecision`.

Решение менеджера **не** принимается на веру. После ответа LLM:

```python
if state.current_turn < settings.min_turns_before_end:
    should_end = False
if state.current_turn >= settings.max_turns:
    should_end = True

if state.current_turn == 1:
    decision.soft_skills_score = 0
```

Это детерминированные политики поверх недетерминированного решения. Это повторяющийся паттерн в системе: **LLM предлагает, код решает**.

### Interviewer

Читает недавний диалог, рекомендации эксперта и стратегию менеджера. Генерирует следующий вопрос для кандидата.

Output: plain text (не структурированный — вопросы свободной формы).

Есть защитный шаг post-processing — `_strip_json_wrapper`. Несмотря на явные инструкции "верни только текст", модель иногда оборачивает вопрос в `{"question": "..."}`, особенно когда в контексте много structured outputs от других агентов. Wrapper детектируется и распаковывается без жалоб — пользователь никогда не видит сырой JSON.


### Structured output (`complete_structured`)

Получив Pydantic-схему, клиент:
1. Строит JSON Schema через `schema.model_json_schema()`.
2. Строит **конкретный пример** dict через `_example_from_schema()` — используя default поля, default factory или type-appropriate dummy.
3. Шлёт промпт, включающий и схему (для структуры), и пример (для формы).
4. Получает ответ с запросом `response_format={"type": "json_object"}`.
5. Валидирует через `schema.model_validate_json(raw)`.

Пример имеет значение. Без него модель иногда возвращает саму JSON Schema (объект с `properties`, `required` и т.д.) вместо данных, соответствующих ей. С примером это failure mode почти исчезает. Это простая форма [few-shot prompting](https://www.promptingguide.ai/techniques/fewshot), адаптированная под structured output.

[`field_validator(mode="before")`](https://docs.pydantic.dev/latest/concepts/validators/) на `ManagerDecision.direction` обрабатывает оставшийся edge case: когда промпт упоминает "bullet points", модель иногда возвращает list. Validator приводит lists к строкам через newline-join.

## Retrieval (Hybrid RAG)

Когда CSV-базы знаний доступны, retrieval комбинирует три сигнала:

```
Query
  ├─→ BM25 (lemmatized) ─────┐
  ├─→ Bi-encoder embedding ──┼─→ Score fusion ─→ Top-K candidates ─→ Cross-encoder rerank ─→ Final results
  └──────────────────────────┘
```

**BM25 с лемматизацией (rank-bm25 + pymorphy3).** Русская морфология богатая; "программист", "программирование" и "программисту" должны матчиться по одной лемме. `pymorphy3` приводит их к единой нормальной форме перед BM25-индексированием. Без лемматизации BM25 пропускает ~30% релевантных матчей в русском языке.

**Bi-encoder (sentence-transformers).** Многоязычный encoder выдаёт dense embeddings; cosine similarity в FAISS даёт семантический recall.

**Cross-encoder reranker.** Top-K кандидатов из lexical+dense fusion переоцениваются cross-encoder'ом. Cross-encoder видит query и document вместе (в отличие от их раздельного кодирования), даёт значительно более высокую precision ценой скорости. Reranking только top-K держит стоимость ограниченной.

Это **стандартная production-grade RAG-архитектура**, иногда называемая "hybrid search with reranking". Она значительно превосходит чистый vector search, особенно на технических текстах, где точная терминология имеет значение.

`Encoder` и `Reranker` определены как Protocols, а не как конкретные классы — любой объект с правильными сигнатурами методов работает. Это сохраняет retriever тестируемым с mock-объектами.

## Калибровка

Калибровка — самая тонкая часть системы и часть с наибольшим количеством failure modes. Она решает три проблемы:

### Проблема 1: Positivity bias

Модели, обученные через RLHF, по умолчанию используют поощрительный язык. Без калибровки система выдавала бы "Strong Hire 90% confidence" почти любому пришедшему кандидату.

**Фикс**: Явный фрейминг в промпте финального фидбэка:

> You are a calibrated technical hiring assessor. Generate a structured final feedback report. Be honest, not encouraging — this is a high-stakes hiring decision, NOT a coaching conversation.

Плюс калибровочная таблица:

```
confidence_score (0-100):
  - 90-100: только при 8+ ходах и стабильно сильных/слабых ответах
  - 70-89: при 6+ ходах и явном паттерне
  - 50-69: при 4-6 ходах и смешанном или частичном сигнале
  - 30-49: при 2-4 ходах или противоречивых данных
  - 0-29: при менее чем 2 содержательных ответах — слишком мало данных
```


### Проблема 2: Поведенческие сигналы усредняются

Самая тонкая проблема: кандидат даёт два сильных технических ответа, потом уходит в off-topic. Наивный вердикт усредняет всё до "Borderline". Но в реальных собеседованиях off-topic-уходы — **дисквалифицирующие**, а не просто незначительные минусы.

**Фикс**: Отдельное поле `behavioral_red_flags: list[str]` в `FinalFeedback`. Промпт явно перечисляет, что в него входит (off-topic анекдоты посреди ответа, уклонения, фундаментальные ошибки, не исправленные после нескольких попыток). После ответа LLM код-уровень понижает рекомендацию, если флаги есть:

```python
if feedback.behavioral_red_flags and recommendation in (STRONG_HIRE, HIRE):
    feedback.hiring_recommendation = HiringRecommendation.BORDERLINE
```

Это **gate**, не мягкая подсказка. Технический контент может быть хорошим; если поведение было непрофессиональным, рекомендация не может быть Hire.

## Обработка ошибок и retry

Три категории ошибок, три ответа:

**Сетевые и rate-limit ошибки → retry с exponential backoff.** Кастомный `_should_retry`- предикат принимает `ConnectionError`, `TimeoutError`, `OSError`, HTTP 429 и HTTP 5xx. Другие ошибки (валидация, авторизация) пробрасываются сразу.

**Логирование при retry → кастомный `before_sleep` callback.** Встроенный в Tenacity `before_sleep_log` зовёт loguru с `str.format`-style placeholders. Если текст исключения содержит `{...}` (например, JSON-тело ошибки), loguru пытается подставить скобки и крашится с `KeyError`. Кастомный callback использует позиционные `{}` и передаёт `repr(exception)` вместо raw-сообщения.

**LLM возвращает плохо сформированный structured output → defense in depth.** Три слоя по порядку: явный пример в промпте → field validators в схеме → post-processing strip в interviewer. Каждый слой ловит разный failure mode.
