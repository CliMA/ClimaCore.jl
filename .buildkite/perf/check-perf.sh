#!/usr/bin/env bash
#
# Compare a run's performance number against the last passing build of the
# default branch, failing when it moves the wrong way by more than
# $MIN_PERCENT_CHANGE percent. On the default branch the check records the new
# number as the baseline instead of failing.
#
#     bash check-perf.sh output/<job_id>/artifacts/sypd.txt
#
# The number is one that should grow: a rate, not a duration. A benchmark that
# reports a time should be turned into a rate before it gets here.
#
# The baseline is stored in Buildkite build metadata under one key per step, so
# steps benchmarking different models do not overwrite each other. $BASELINE is
# the fallback for when no passing build has recorded one yet; a step with
# neither reports its number and passes.

METRIC_FILE="$1"

if [[ -z "$METRIC_FILE" ]]; then
    echo "❌ Usage: check-perf.sh <path to a file holding one number>"
    exit 1
fi

if [[ ! -f "$METRIC_FILE" ]]; then
    echo "❌ Metric file not found: $METRIC_FILE"
    exit 1
fi

METRIC=$(cat "$METRIC_FILE" | tr -d '[:space:]')

if [[ -z "$METRIC" ]]; then
    echo "❌ Metric file is empty"
    exit 1
fi

METRIC_NAME=$(basename "$METRIC_FILE" .txt)
echo "$METRIC_NAME: $METRIC"

DEFAULT_BRANCH="${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-main}"
META_KEY="baseline_${METRIC_NAME}_${BUILDKITE_STEP_KEY:-default}"

FETCHED_BASELINE=$(curl -sf \
    -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
    "https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds?branch=${DEFAULT_BRANCH}&state=passed&per_page=1" \
    | jq -r ".[0].meta_data.${META_KEY} // empty")

if [[ -n "$FETCHED_BASELINE" ]]; then
    echo "📥 Using $META_KEY=$FETCHED_BASELINE from the last passing $DEFAULT_BRANCH build"
    BASELINE="$FETCHED_BASELINE"
else
    echo "⚠️  Could not fetch $META_KEY from $DEFAULT_BRANCH; using fallback BASELINE=$BASELINE"
fi

if [[ "${BUILDKITE_BRANCH}" == "$DEFAULT_BRANCH" ]]; then
    # Record the new baseline whether or not performance moved.
    buildkite-agent meta-data set "$META_KEY" "$METRIC"
    echo "📌 Stored $META_KEY=$METRIC in build metadata"
    exit 0
fi

# A step that has never passed on the default branch has nothing to compare
# against, so it reports its number for the first default-branch build to record
# rather than failing. Give a step a fallback BASELINE once its spread is known,
# and this becomes a hard failure whenever the metadata fetch breaks.
if [[ -z "$BASELINE" ]]; then
    echo "⚠️  No baseline for $META_KEY yet; record $METRIC once this runs on $DEFAULT_BRANCH"
    exit 0
fi

PERCENT_CHANGE=$(echo "scale=2; (($METRIC - $BASELINE) / $BASELINE) * 100" | bc)

if (( $(echo "$PERCENT_CHANGE <= $MIN_PERCENT_CHANGE" | bc -l) )); then
    echo "❌ $METRIC_NAME changed by $PERCENT_CHANGE% (threshold: $MIN_PERCENT_CHANGE%)"
    exit 1
fi

echo "✅ $METRIC_NAME change ($PERCENT_CHANGE%) is okay (threshold: $MIN_PERCENT_CHANGE%)"
