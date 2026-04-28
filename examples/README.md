# Examples
> [Русская версия](README.ru.md)

Sample interview reports produced by the system.
## File naming

```
interview_<candidate_name>_<YYYYMMDD>_<HHMMSS>.json
```

## Structure

Each file contains:

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
      "interviewer_message": "the question asked",
      "timestamp": "ISO datetime",
      "candidate_message": "answer (null on first turn)",
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

## Reading tips

- `turn_id: 1` always has `candidate_message: null` — the interviewer asks first.
- `expert_analysis` is `null` on turn 1 (nothing to analyze yet).
- `behavioral_red_flags` is empty for normal interviews; populated when the candidate deflects, drifts off-topic, or shows inconsistency.
- `confidence_score` should be calibrated against turn count. Look for `< 60%` on interviews of < 5 substantive turns.
