# CI/release validation checklist

## CI enforcement validation

1. Open a temporary PR with an intentional `repo_sanity.sh` failure (for example, add a temporary required file entry that does not exist).
2. Confirm `repo-sanity` fails and merge is blocked by branch protection.
3. Revert the intentional failure and confirm `repo-sanity` passes.
4. Optionally trigger an advisory failure (lint/link) and confirm merge is still allowed when `repo-sanity` is green.

## Release workflow validation

1. Trigger `Release` workflow via `workflow_dispatch` with:
   - `tag_name` set to a temporary prerelease tag (for example `v0.0.0-rc1`)
   - `prerelease=true`
2. Confirm wheel/sdist artifacts are produced and attached to the prerelease.
3. Optionally validate tag-push release flow by pushing the same temporary test tag.
4. Remove temporary tag/release after validation.

## Secrets policy

- External provider/API credentials must be stored only in GitHub encrypted secrets.
- No credentials are committed in repository files or workflow definitions.
