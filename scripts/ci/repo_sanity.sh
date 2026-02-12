#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "README.md"
  "LICENSE"
  "AGENTS.md"
  "docs/engineering/merge-philosophy.md"
  "docs/engineering/ci-policy.md"
  "docs/engineering/github-settings.md"
)

missing=0
for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Missing required file: ${file_path}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n --hidden --glob '!.git/**' '^(<<<<<<<|=======|>>>>>>>)' .; then
    echo "Merge conflict markers found."
    exit 1
  fi
else
  if grep -RIn --exclude-dir=.git -E '^(<<<<<<<|=======|>>>>>>>)' .; then
    echo "Merge conflict markers found."
    exit 1
  fi
fi

echo "Repo sanity checks passed."
