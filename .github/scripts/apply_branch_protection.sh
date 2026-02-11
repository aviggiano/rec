#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-aviggiano/rec}"
BRANCH="${2:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../branch-protection.json"

echo "Applying branch protection for ${REPO}:${BRANCH} using ${CONFIG_FILE}"

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input "${CONFIG_FILE}" >/dev/null

echo "Branch protection applied."
