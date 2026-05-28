# Skill — tool use

When the controller picks a tool to call:

- Call one tool per turn. Do not chain multiple actions in a single response.
- Pass only the arguments declared in the tool's `arg_schema`. Omit unknown
  keys.
- Prefer reading observations already on the transcript over re-running the
  same tool with identical args.
- If a tool fails or returns no useful evidence twice in a row, switch to a
  different tool rather than retrying.
- Treat `finish` (and the `final` action) as the only ways to commit an
  answer; do not paste the answer into another tool's args to "preview" it.
- Never invent tool names, benchmark identifiers, or task IDs.
