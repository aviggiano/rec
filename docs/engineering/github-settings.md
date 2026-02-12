# GitHub settings

Apply these settings to the `main` branch so repository behavior matches the
throughput-first CI policy.

## Branch protection

- Require a pull request before merging.
- Require status checks to pass before merging.
- Set only one required check: `repo-sanity` (from `.github/workflows/merge-gate.yml`).
- Do not make advisory checks required.
- Keep force pushes and branch deletions disabled on `main`.

## Merge options

- Enable squash merge.
- Enable auto-merge.
- Enable "delete head branch after merge".
- Keep merge queue optional unless PR throughput requires it.

## Operational result

Merges are gated by one deterministic safety check, while slower or flaky checks
remain visible and actionable without blocking delivery.
