# Branch protection

This repository uses branch protection for `main` and requires CI checks before merge.

## Required checks

- `quality-checks`
- `smoke-matrix (ubuntu-latest)`
- `smoke-matrix (macos-latest)`

## Review rules

- At least 1 approving review
- Dismiss stale approvals on new commits
- Require conversation resolution before merge

## Apply protection

Use GitHub encrypted credentials via `gh auth login` and then run:

```bash
.github/scripts/apply_branch_protection.sh aviggiano/rec main
```

The configuration is stored in `.github/branch-protection.json`.
