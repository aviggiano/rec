# CI policy

## Goal

CI should protect correctness without turning every PR into a waiting queue.

## Check types

Blocking checks:

- Fast
- Deterministic
- Directly tied to correctness or repository integrity

Advisory checks:

- Potentially flaky (network, external service variance, etc.)
- Slower quality signals
- Useful for detection, but not always reliable merge gates

## Current checks

Blocking:

- `repo-sanity` (from `.github/workflows/merge-gate.yml`)

Advisory:

- `quality-checks` and `smoke-matrix` from `.github/workflows/ci.yml`
- `markdown-lint` and `links` from `.github/workflows/advisory-checks.yml`

## Flake handling

1. Re-run the failed advisory check once.
2. If it passes on rerun, continue.
3. If it fails again but appears flaky, do not block merge when blocking checks are green.
4. Open a flake-tracking issue using `.github/ISSUE_TEMPLATE/ci-flake.yml`.

## Follow-up SLA

- Open the flake issue at merge time.
- Schedule a fix or mitigation quickly (target: within 48 hours).
- Promote to blocking only after reliability is good.
