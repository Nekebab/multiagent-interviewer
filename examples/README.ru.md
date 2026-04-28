# Примеры

> [English version](README.md)

Образцы отчётов интервью, генерируемых системой.

## Именование файлов

```
interview_<имя_кандидата>_<YYYYMMDD>_<HHMMSS>.json
```

## Структура

Каждый файл содержит:

```jsonc
{
  "candidate": {
    "name": "...",
    "position": "...",
    "grade": "Junior | Middle | Senior",
    "experience": "..."
  },
  "turns": [
    {
      "turn_id": 1,
      "interviewer_message": "заданный вопрос",
      "timestamp": "ISO datetime",
      "candidate_message": "ответ (null на первом ходу)",
      "expert_analysis": { ... },
      "manager_decision": { ... }
    }
  ],
  "final_feedback": {
    "grade_assessment": "...",
    "hiring_recommendation": "Strong Hire | Hire | Borderline | No Hire | Insufficient Data",
    "confidence_score": 0-100,
    "confirmed_skills": [...],
    "knowledge_gaps": [...],
    "behavioral_red_flags": [...],
    "soft_skills_summary": "...",
    "learning_roadmap": [...],
    "suggested_resources": [...]
  }
}
```

## Подсказки по чтению

- У `turn_id: 1` всегда `candidate_message: null` — интервьюер задаёт вопрос первым.
- `expert_analysis` — `null` на ходу 1 (анализировать ещё нечего).
- `behavioral_red_flags` пустой для нормальных интервью; заполняется, когда кандидат уклоняется, уходит off-topic или показывает неконсистентность.
- `confidence_score` должен быть откалиброван по числу ходов. Ожидай `< 60%` на интервью из < 5 содержательных ходов.
