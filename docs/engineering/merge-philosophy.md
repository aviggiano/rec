# Merge philosophy

## Why this model

`rec` follows a throughput-first merge philosophy: waiting is expensive, and
corrections are cheap when pull requests stay small.

This is inspired by OpenAI's
[Harness Engineering write-up (February 11, 2026)](https://openai.com/index/harness-engineering/).

The important tradeoff is context-dependent:

- In high-throughput environments, long waits at merge gates reduce output.
- In low-throughput environments, heavy gates can still be appropriate.
- Here, we optimize for fast iteration with explicit follow-up ownership.

## Repository policy

1. Keep PRs short-lived and narrow in scope.
2. Merge once deterministic safety checks pass.
3. Run slower or flakier checks as advisory signals.
4. Re-run suspected flakes, then track them with issues instead of blocking indefinitely.
5. Prefer quick follow-up PRs to recover or improve behavior.

## Escalation rules

Move a check from advisory to blocking when one of these is true:

- It has become deterministic and fast.
- It repeatedly catches high-impact regressions.
- Its false-positive and flake rate is low enough for reliable gating.

Move a check from blocking to advisory when it becomes flaky or too slow for
the normal PR path.
