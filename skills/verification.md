# Skill — evidence-grounded verification

When verifying a candidate against evidence:

- Treat literal string match in the evidence as the strongest signal.
- Numerical answers must match every relevant qualifier (period, units,
  scope, sign).
- Downgrade confidence if the candidate generalises or paraphrases a span
  beyond what the evidence supports.
- Surface specific concerns in the `concerns` list (what is missing or
  mismatched), not vague worries.
- Prefer "uncertain" over a confident guess.
