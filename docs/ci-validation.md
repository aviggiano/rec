# CI/release validation checklist

## CI enforcement validation

1. Open a temporary PR with an intentional lint or test failure.
2. Confirm `quality-checks` fails and merge is blocked by branch protection.
3. Revert the intentional failure and confirm checks pass.

## Release workflow validation

1. Trigger `Release` workflow via `workflow_dispatch` with `prerelease=true`.
2. Confirm wheel/sdist artifacts are produced and attached to the prerelease.
3. Trigger release by pushing a temporary test tag (for example `v0.0.0-rc1`) if needed.
4. Remove test tag/release after validation.

## Secrets policy

- External provider/API credentials must be stored only in GitHub encrypted secrets.
- No credentials are committed in repository files or workflow definitions.
