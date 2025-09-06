# Project Guidelines

This document defines working rules for assistants on this repo.

## Goals
- Keep changes minimal, focused, and reversible.
- Prefer clarity and simplicity over cleverness.
- Update documentation alongside code changes.

## Workflow
- Discuss intent, then propose a short plan for multi-step tasks.
- Make small, focused patches with clear summaries.
- Validate locally when possible; surface follow-ups succinctly.

## Coding Standards
- Python 3.12; format with Black (line length 88).
- Lint with Ruff; type-check with mypy where practical.
- Avoid one-letter variable names; keep functions small and cohesive.

## Docs
- MkDocs with pages in `docs/`. Default dev address: `127.0.0.1:8001`.
- Keep `mkdocs.yml` nav in sync with added pages.

## Approvals & Safety
- Ask before adding dependencies or running destructive commands.
- Never include secrets in code or config.

## Commit & PR Hygiene (if applicable)
- Descriptive titles, imperative mood.
- Include rationale and user-visible impact.

## Open Questions
- Confirm preferred docs theme and structure.
- Confirm testing/linting stack (defaults noted above).

