# Skill — faithful extraction

When extracting an answer from evidence:

- Prefer verbatim spans over paraphrase.
- Preserve units, signs, and notation: `$1.8M`, `12%`, `Q3 2023`, `-4.1`.
- When multiple candidates exist, pick the one most directly tied to the
  prompt's named entity, period, or qualifier.
- Never fill in numbers, dates, or names that do not appear in the evidence.
- If the prompt asks for a single value, output a single value, not a
  sentence.
