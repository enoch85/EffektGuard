#!/bin/bash
# EffektGuard Home Assistant integration release script
# Inspired by ge-spot release process

set -e

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Integration naming constants
MAIN_NAME="EffektGuard"
REPO_NAME="EffektGuard"
INTEGRATION_PATH="custom_components/effektguard"
MANIFEST_PATH="${INTEGRATION_PATH}/manifest.json"

# Release flags
IS_PRERELEASE=false
IS_PROD_RELEASE=false
DEBUG=true

# Debug function
function debug_log {
    if [[ "$DEBUG" == "true" ]]; then
        echo -e "${BLUE}DEBUG:${NC} $1" >&2
    fi
}

# Error function
function error_log {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

# Info function
function info_log {
    echo -e "${YELLOW}INFO:${NC} $1" >&2
}

# Success function 
function success_log {
    echo -e "${GREEN}SUCCESS:${NC} $1" >&2
}

# Display usage information
function show_usage {
    echo -e "${YELLOW}Usage:${NC} bash release.sh <version_tag> [options]"
    echo -e "${YELLOW}Example:${NC} bash release.sh v0.0.1"
    echo -e "${YELLOW}Example (beta):${NC} bash release.sh v0.0.1-beta.1"
    echo -e "${YELLOW}Example (prod):${NC} bash release.sh v0.0.1 --prod"
    echo ""
    echo "Options:"
    echo "  --prod        Production release mode (runs tests, cleans up beta tags/releases)"
    echo "  --pr-only     Create a PR for the release without pushing tags"
    echo "  --help        Show this help message"
    echo ""
    echo "The version tag must follow the format 'vX.Y.Z' where X, Y, Z are numbers."
    echo "For pre-releases, use the format 'vX.Y.Z-alpha.N' or 'vX.Y.Z-beta.N'."
    echo ""
    echo "The release process will:"
    echo "  1. Update version in manifest.json"
    echo "  2. Update version badge in README.md"
    echo "  3. Commit and push changes"
    echo "  4. Create and push git tag"
    echo "  5. Trigger GitHub Actions to create the release"
    echo ""
    echo "Production Release (--prod) workflow:"
    echo "  1. Switches to main branch and pulls latest changes"
    echo "  2. Runs pytest test suite (must pass)"
    echo "  3. Removes all beta tags for this version"
    echo "  4. Removes all beta releases for this version"
    echo "  5. Creates production release"
}

# Validate version tag format
function validate_version_tag {
    if [[ ! "${1}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.]+)?$ ]]; then
        error_log "Invalid version tag format!"
        echo "The version tag must follow the format 'vX.Y.Z' or 'vX.Y.Z-suffix'"
        show_usage
        exit 1
    fi
    
    # Check if this is a pre-release
    if [[ "${1}" =~ -(alpha|beta|rc) ]]; then
        IS_PRERELEASE=true
        info_log "Pre-release version detected: ${1} - Will be marked as a pre-release"
    fi
}

# Check if we're on main or initial-implementation branch
function check_branch {
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if [[ "$current_branch" != "main" ]] && [[ "$current_branch" != "initial-implementation" ]]; then
        if [[ "$IS_PRERELEASE" == "true" ]]; then
            info_log "Pre-release detected - allowing release from branch: $current_branch"
        else
            error_log "You are not on the main or initial-implementation branch!"
            echo "Current branch: $current_branch"
            echo "Please switch to the main or initial-implementation branch before creating a release."
            exit 1
        fi
    fi
}

# Check for uncommitted changes
function check_uncommitted_changes {
    if ! git diff-index --quiet HEAD --; then
        info_log "You have uncommitted changes."
        read -p "Do you want to continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting release process."
            exit 1
        fi
    fi
}

# Check if tag already exists
function check_tag_exists {
    if git tag -l | grep -q "^${1}$"; then
        error_log "Tag ${1} already exists!"
        echo "Please use a different version tag."
        exit 1
    fi
}

# Check if manifest.json exists and is valid
function check_manifest {
    if [[ ! -f "$MANIFEST_PATH" ]]; then
        error_log "Manifest file not found at: $MANIFEST_PATH"
        echo "Please make sure you're running this script from the project root."
        exit 1
    fi
    
    # Check if manifest.json is valid JSON
    if ! command -v jq &> /dev/null || ! cat "$MANIFEST_PATH" | jq . &>/dev/null; then
        error_log "Manifest file is not valid JSON: $MANIFEST_PATH"
        exit 1
    fi
    
    debug_log "Manifest file found and valid: $MANIFEST_PATH"
    return 0
}

# Update version in manifest.json
function update_manifest {
    local version_tag="$1"
    # Keep the full version tag with 'v' prefix (like ge-spot does: "v1.5.0")
    local manifest_changed=false
    
    # Check if manifest exists and is valid JSON
    check_manifest
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Update Manifest Version
    local current_manifest_version=$(grep -o '"version": *"[^"]*"' "$MANIFEST_PATH" | cut -d'"' -f4)
    if [[ "$current_manifest_version" == "$version_tag" ]]; then
        info_log "Version is already set to ${version_tag} in manifest.json"
    else
        debug_log "Updating version in: $MANIFEST_PATH"
        if [[ "$(uname)" == "Darwin" ]]; then
            sed -i '' "s/\\\"version\\\": *\\\"[^\\\"]*\\\"/\\\"version\\\": \\\"${version_tag}\\\"/g" "$MANIFEST_PATH"
        else
            sed -i "s/\\\"version\\\": *\\\"[^\\\"]*\\\"/\\\"version\\\": \\\"${version_tag}\\\"/g" "$MANIFEST_PATH"
        fi
        if [[ $? -eq 0 ]]; then
            success_log "Version updated to ${version_tag} in manifest.json"
            manifest_changed=true
        else
            error_log "Failed to update version in manifest.json"
            return 1
        fi
    fi
    
    # Return 0 if changes were made, 1 otherwise
    if [[ "$manifest_changed" == true ]]; then
        return 0
    else
        return 1
    fi
}

# Update version badge in README.md
function update_readme_badge {
    local version_tag="$1"
    # Remove 'v' prefix for the badge (e.g., v0.0.52-beta1 -> 0.0.52-beta1)
    local version_no_v="${version_tag#v}"
    local readme_path="README.md"
    local badge_changed=false
    
    if [[ ! -f "$readme_path" ]]; then
        error_log "README.md not found"
        return 1
    fi
    
    # Check current badge version (matches everything between "version-" and "-blue")
    local current_badge_version=$(grep -oP 'version-\K[^)]+(?=-blue)' "$readme_path" | head -1)
    
    if [[ "$current_badge_version" == "$version_no_v" ]]; then
        info_log "Version badge is already set to ${version_no_v} in README.md"
    else
        debug_log "Updating version badge in: $readme_path"
        # Use a more robust pattern that handles versions with dashes (e.g., beta1)
        if [[ "$(uname)" == "Darwin" ]]; then
            sed -i '' -E "s/version-[^)]+(-blue)/version-${version_no_v}\1/g" "$readme_path"
        else
            sed -i -E "s/version-[^)]+(-blue)/version-${version_no_v}\1/g" "$readme_path"
        fi
        if [[ $? -eq 0 ]]; then
            success_log "Version badge updated to ${version_no_v} in README.md"
            badge_changed=true
        else
            error_log "Failed to update version badge in README.md"
            return 1
        fi
    fi
    
    # Return 0 if changes were made, 1 otherwise
    if [[ "$badge_changed" == true ]]; then
        return 0
    else
        return 1
    fi
}

# Pull latest changes from remote
function pull_latest_changes {
    info_log "Pulling latest changes..."
    if ! git pull --rebase; then
        error_log "Failed to pull latest changes."
        echo "Please fix the error, then try again."
        exit 1
    fi
    success_log "Latest changes pulled"
    return 0
}

# Run pytest tests
function run_tests {
    info_log "Running pytest test suite..."
    
    # Check if pytest script exists
    if [[ -f "scripts/run_pytest.sh" ]]; then
        if ! bash scripts/run_pytest.sh; then
            error_log "Tests failed! Please fix failing tests before releasing."
            exit 1
        fi
    else
        # Fallback to direct pytest
        if command -v pytest &> /dev/null; then
            if ! pytest tests/ -v; then
                error_log "Tests failed! Please fix failing tests before releasing."
                exit 1
            fi
        else
            info_log "pytest not found, skipping tests"
        fi
    fi
    
    success_log "All tests passed"
    return 0
}

# Remove all beta tags for a version
function remove_beta_tags {
    local base_version="$1"
    
    info_log "Searching for beta tags for version ${base_version}..."
    
    # List all beta tags for this version
    local beta_tags=$(git tag -l "${base_version}-beta*")
    
    if [[ -z "$beta_tags" ]]; then
        info_log "No beta tags found for ${base_version}"
        return 0
    fi
    
    echo -e "${YELLOW}Found beta tags:${NC}"
    echo "$beta_tags"
    
    # Delete local tags
    info_log "Deleting local beta tags..."
    for tag in $beta_tags; do
        if git tag -d "$tag"; then
            debug_log "Deleted local tag: $tag"
        else
            error_log "Failed to delete local tag: $tag"
        fi
    done
    
    # Delete remote tags
    info_log "Deleting remote beta tags..."
    for tag in $beta_tags; do
        if git push origin ":refs/tags/$tag" 2>/dev/null; then
            debug_log "Deleted remote tag: $tag"
        else
            info_log "Remote tag $tag doesn't exist or already deleted"
        fi
    done
    
    success_log "Beta tags removed"
    return 0
}

# Remove all beta releases for a version
function remove_beta_releases {
    local base_version="$1"
    
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        error_log "GitHub CLI (gh) is not installed!"
        echo "Please install gh to use production release mode: https://cli.github.com/"
        exit 1
    fi
    
    info_log "Searching for beta releases for version ${base_version}..."
    
    # List all releases and filter for beta versions using JSON output for reliable parsing
    # The default tabular output has variable-width columns making awk unreliable
    local beta_releases=$(gh release list --limit 100 --json tagName --jq '.[].tagName' | grep "${base_version}-beta" || true)
    
    if [[ -z "$beta_releases" ]]; then
        info_log "No beta releases found for ${base_version}"
        return 0
    fi
    
    echo -e "${YELLOW}Found beta releases:${NC}"
    echo "$beta_releases"
    
    # Delete each beta release
    info_log "Deleting beta releases..."
    for release in $beta_releases; do
        if gh release delete "$release" -y; then
            debug_log "Deleted release: $release"
        else
            error_log "Failed to delete release: $release"
        fi
    done
    
    success_log "Beta releases removed"
    return 0
}

# Switch to main branch
function switch_to_main {
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if [[ "$current_branch" == "main" ]]; then
        info_log "Already on main branch"
    else
        info_log "Switching to main branch..."
        if ! git checkout main; then
            error_log "Failed to switch to main branch"
            exit 1
        fi
        success_log "Switched to main branch"
    fi
    
    return 0
}

# Creates and pushes a tag
function create_and_push_tag {
    local version_tag="$1"
    local tag_message="Release ${version_tag}"
    
    # For pre-release versions, add a note in the tag message
    if [[ "$IS_PRERELEASE" == true ]]; then
        tag_message="${tag_message} (Pre-release)"
    fi
    
    info_log "Creating and pushing tag ${version_tag}..."
    if ! git tag -a "${version_tag}" -m "${tag_message}"; then
        error_log "Failed to create tag ${version_tag}"
        return 1
    fi
    
    if ! git push origin "${version_tag}"; then
        error_log "Failed to push tag ${version_tag}"
        return 1
    fi
    
    if [[ "$IS_PRERELEASE" == true ]]; then
        success_log "Pre-release tag ${version_tag} created and pushed"
    else
        success_log "Tag ${version_tag} created and pushed"
    fi
    return 0
}

# Production release workflow
function run_production_release {
    local version_tag="$1"
    
    info_log "Starting PRODUCTION release process for ${MAIN_NAME} ${version_tag}..."
    
    # Step 1: Switch to main branch
    switch_to_main
    
    # Step 2: Pull latest changes
    pull_latest_changes
    
    # Step 3: Run tests
    run_tests
    
    # Step 4 & 5: Clean up beta tags and releases
    # Extract base version (e.g., v1.3.4 from v1.3.4)
    remove_beta_tags "$version_tag"
    remove_beta_releases "$version_tag"
    
    # Step 6: Create production release
    info_log "Creating production release..."
    run_release_process "$version_tag"
    
    success_log "Production release ${version_tag} completed successfully!"
    return 0
}

# Main execution function
function run_release_process {
    local version_tag="$1"
    
    info_log "Starting release process for ${MAIN_NAME} ${version_tag}..."
    
    # Initial validation checks
    validate_version_tag "$version_tag"
    check_branch
    check_uncommitted_changes
    check_tag_exists "$version_tag"
    
    # Pull latest changes (unless already pulled in prod workflow)
    if [[ "$IS_PROD_RELEASE" != "true" ]]; then
        pull_latest_changes
    fi
    
    # Update manifest.json
    update_manifest "$version_tag"
    manifest_changed=$?
    
    # Update README.md badge
    update_readme_badge "$version_tag"
    readme_changed=$?
    
    # If changes were made, commit and push
    if [[ "$manifest_changed" -eq 0 ]] || [[ "$readme_changed" -eq 0 ]]; then
        info_log "Staging changes..."
        
        # Stage manifest if it changed
        if [[ "$manifest_changed" -eq 0 ]]; then
            git add "$MANIFEST_PATH"
        fi
        
        # Stage README if it changed
        if [[ "$readme_changed" -eq 0 ]]; then
            git add "README.md"
        fi
        
        info_log "Committing changes..."
        if ! git commit -m "Release ${version_tag} of ${MAIN_NAME}"; then
            error_log "Failed to commit changes"
            exit 1
        fi
        success_log "Changes committed"
        
        info_log "Pushing to current branch..."
        local current_branch=$(git rev-parse --abbrev-ref HEAD)
        if ! git push origin "${current_branch}"; then
            error_log "Failed to push changes"
            exit 1
        fi
        success_log "Changes pushed"
    fi
    
    # Create and push tag
    if ! create_and_push_tag "$version_tag"; then
        error_log "Failed to create and push tag"
        exit 1
    fi
    
    success_log "GitHub release will be created automatically by GitHub Actions when tag is pushed"
    echo -e "${BLUE}Monitor the process at:${NC} https://github.com/enoch85/${REPO_NAME}/actions"
    
    if [[ "$IS_PRERELEASE" == true ]]; then
        success_log "${MAIN_NAME} ${version_tag} (PRE-RELEASE) successfully prepared!"
    else
        success_log "${MAIN_NAME} ${version_tag} successfully prepared!"
    fi
    
    return 0
}

# Main execution starts here

# Parse arguments
VERSION_TAG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prod) IS_PROD_RELEASE=true; shift ;;
        --help) show_usage; exit 0 ;;
        -h) show_usage; exit 0 ;;
        --pr-only) info_log "--pr-only flag noted (not yet implemented)"; shift ;;
        -*) echo "Unknown option: $1"; show_usage; exit 1 ;;
        *) VERSION_TAG="$1"; shift ;;
    esac
done

# Check if version tag is provided
if [ -z "${VERSION_TAG}" ]; then
    error_log "You forgot to add a release tag!"
    show_usage
    exit 1
fi

# Validate version format
validate_version_tag "$VERSION_TAG"

# Production releases cannot have beta/alpha/rc suffixes
if [[ "$IS_PROD_RELEASE" == "true" ]] && [[ "$IS_PRERELEASE" == "true" ]]; then
    error_log "Production releases cannot have pre-release suffixes (alpha/beta/rc)"
    echo "Please use a clean version tag like 'v1.3.4' for production releases"
    exit 1
fi

# Start the appropriate release process
if [[ "$IS_PROD_RELEASE" == "true" ]]; then
    run_production_release "$VERSION_TAG"
else
    run_release_process "$VERSION_TAG"
fi
