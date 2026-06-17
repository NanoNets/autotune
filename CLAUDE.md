# CLAUDE.md

Build-time guidance for working in this project. This file is **advisory** — it nudges,
it doesn't enforce.

---

## Coding guidelines

1. **Think before coding.** State assumptions, surface tradeoffs. Unclear → ask before
   implementing. Multiple interpretations → present them, don't pick silently. Plan before
   every major change.
2. **Simplicity first.** Minimum code that solves the problem. No speculative abstractions,
   no unrequested config knobs, no error handling for impossible cases. If 200 lines could
   be 50, rewrite.
3. **Surgical changes.** Touch only what you must. No drive-by refactors, no reformatting
   untouched code. Remove orphans your change created; leave pre-existing dead code alone
   (mention it, don't delete it). Every changed line traces to the stated goal.
4. **Goal-driven execution.** Turn tasks into verifiable goals ("write the test, then make
   it pass"). State a brief plan for multi-step work; check off as you go.
5. **Explain before coding.** Say what you're building, why, and how it works conceptually.
   Then code.
6. **Concise code docs.** Comments and docstrings are short one-liners; detailed explanation
   goes in chat, not source. After each change, update this file so it matches the codebase.

---

## Process docs

Three files carry the working state. Only this file (`CLAUDE.md`) loads automatically;
read the others on their trigger.

| File | Read it when… | Write it when… |
|---|---|---|
| `CLAUDE.md` | always loaded — the standing guidance | a coding rule or working agreement changes |
| `PLAN.md` | session start | scope changes, a task completes, session end |
| `HANDOFF.md` | first thing, every session | session end — overwrite with current state |

### What each file owns

- **CLAUDE.md** — the coding guidelines and working agreements. Durable.
- **PLAN.md** — the work queue: scope, current task, next. Empties as work finishes.
- **HANDOFF.md** — the save-state: enough context to resume cold next session.

---

## Session flow

```
start  →  read HANDOFF.md  →  read PLAN.md  →  work the queue
work   →  plan → build → verify (build/typecheck/lint/tests) → commit
end    →  update PLAN.md  →  overwrite HANDOFF.md
```
