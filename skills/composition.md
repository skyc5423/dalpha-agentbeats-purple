# Skill — direct answer composition

Final answers should be:

- As short as the question allows. A "what is X" question with a single
  number deserves a single-token answer, not a sentence.
- Free of meta commentary about the agent, the pipeline, or confidence
  scores.
- Faithful: never add caveats that the evidence does not justify, and never
  invent caveats to hedge.
- Format-matching: if the prompt asks for "only the number", or "in one
  word", obey exactly.
