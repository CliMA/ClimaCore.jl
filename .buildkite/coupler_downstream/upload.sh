#!/bin/bash
# Uploads the ClimaCoupler downstream steps: all of them for a PR that changes
# the version of ClimaCore, and a subset otherwise.
set -euo pipefail

dir=$(dirname "$0")
steps="$dir/default.yml"

version() { git show "$1:Project.toml" | grep -m1 '^version'; }

if [[ "${BUILDKITE_PULL_REQUEST:-false}" != false ]]; then
    git fetch --quiet origin "$BUILDKITE_PULL_REQUEST_BASE_BRANCH"
    # The merge base, not the tip of the base branch, so that a PR branched
    # before the last release does not look like a version change
    base=$(git merge-base HEAD FETCH_HEAD)
    old=$(version "$base")
    new=$(version HEAD)
    echo "base ($base): $old"
    echo "PR:   $new"
    if [[ "$old" != "$new" ]]; then
        steps="$dir/release.yml"
    fi
fi

echo "--- Uploading $steps"
buildkite-agent pipeline upload "$steps"
