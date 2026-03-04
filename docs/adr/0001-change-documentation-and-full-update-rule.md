# ADR 0001: Change Documentation and Relevant File Update Rule

- Status: Accepted
- Date: 2026-02-28

## Context

The LRC project requires fast iteration across architecture, research workflow, and implementation. Untracked process decisions and partial file updates increase drift between design intent, code behavior, and user documentation.

## Decision

Every change must be documented, and all relevant files must be updated together before merge.

The minimum required updates for any non-trivial change are:

1. Code changes in the affected modules.
2. Tests that validate the changed behavior.
3. User-facing documentation updates (README and/or docs).
4. Configuration or examples when interfaces change.

## Consequences

- Improves traceability and review quality.
- Reduces stale docs and hidden behavior changes.
- Adds a small overhead per change, offset by lower rework cost.

## Compliance Rule

A pull request is incomplete if it changes behavior without corresponding updates to tests and relevant documentation.