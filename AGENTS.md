# AGENTS.md

## Workflow and approval

- Work design-first. Before changing code, briefly state the problem, likely cause when relevant, affected files, smallest viable change, and key risks or assumptions.
- Implement only after an explicit request such as “implement it,” “make the change,” “go ahead and code it,” or “apply the fix.” Silence, ambiguity, and requests to investigate, analyze, review, debug, explain, or propose a fix are not approval; stop after analysis/design.
- Read-only terminal commands need no permission. Before any command that may change files, dependencies, configuration, the environment, or other state, explain why it is needed, what it does, and what it affects, then obtain explicit permission. A general investigation or fix request is not command approval.
- Prefer the least invasive command. Clearly label commands as read-only or state-changing and recommend the safest small option first.

## Change scope

- Make the smallest simple, local change that satisfies the approved request.
- Preserve existing behavior and edit existing code where possible.
- Do not modify unrelated files, add unjustified abstractions or edge-case handling, broadly refactor, or rename/move/restructure files unless required or explicitly requested.
- After implementation, summarize exactly what changed and why; mention only necessary side effects or follow-up work.

## Python execution environment

- Use the Conda environment `FingerControl` through `cmd.exe`, not PowerShell or the Microsoft Store Python alias.
- Activate it with: `call C:\Users\ynaga\anaconda3\condabin\conda.bat activate FingerControl`
- For one-off commands from the current shell, use: `cmd.exe /d /s /c 'call "C:\Users\ynaga\anaconda3\condabin\conda.bat" activate FingerControl && python <arguments>'`
- The expected interpreter is `C:\Users\ynaga\anaconda3\envs\FingerControl\python.exe`. Do not create or seek another environment unless this one fails.
- Give users `cmd.exe` commands to run after activation unless they explicitly request PowerShell. Use `python -m pip`, never bare `pip`.
