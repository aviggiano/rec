# Branch protection

This repository uses branch protection for `main` and requires CI checks before merge.

## Required checks

- `repo-sanity` (from `.github/workflows/merge-gate.yml`)

Advisory/non-blocking checks still run for signal quality:

- `.github/workflows/ci.yml` (`quality-checks`, `smoke-matrix`)
- `.github/workflows/advisory-checks.yml` (`markdown-lint`, `links`)

## Review rules

- No approving reviews required
- Dismiss stale approvals on new commits
- Require conversation resolution before merge

## Apply protection

Use GitHub encrypted credentials via `gh auth login` and then run:

```bash
.github/scripts/apply_branch_protection.sh aviggiano/rec main
```

The configuration is stored in `.github/branch-protection.json`.
