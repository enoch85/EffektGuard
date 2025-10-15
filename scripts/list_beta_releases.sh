#!/bin/bash
# Helper script to show beta releases for a version

VERSION="$1"

if [[ -z "$VERSION" ]]; then
    echo "Usage: bash scripts/list_beta_releases.sh <version>"
    echo "Example: bash scripts/list_beta_releases.sh v1.3.4"
    exit 1
fi

# Remove 'v' prefix if present, then add it back
VERSION="${VERSION#v}"
VERSION="v${VERSION}"

echo "=== Beta Tags for ${VERSION} ==="
git tag -l "${VERSION}-beta*" | sort -V || echo "No beta tags found"

echo ""
echo "=== Beta Releases for ${VERSION} ==="
if command -v gh &> /dev/null; then
    gh release list --limit 100 | grep "${VERSION}-beta" || echo "No beta releases found"
else
    echo "GitHub CLI (gh) not installed - cannot list releases"
    echo "Install from: https://cli.github.com/"
fi
