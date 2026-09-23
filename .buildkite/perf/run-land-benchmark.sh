#!/usr/bin/env bash
#
# Run one of ClimaLand's own benchmark scripts, so that the model setup stays
# defined in ClimaLand rather than being duplicated here, and turn the mean
# simulation time it reports into a rate for check-perf.sh, which compares
# numbers that are better when larger.
#
#     bash run-land-benchmark.sh <ClimaLand checkout> <benchmark script name>

set -euo pipefail

LAND_PATH="$1"
SCRIPT="$2"
OUTDIR="${SCRIPT}_benchmark_gpu"

julia --color=yes --project="$LAND_PATH/.buildkite" \
    "$LAND_PATH/experiments/benchmarks/$SCRIPT.jl"

julia --project="$LAND_PATH/.buildkite" -e '
    import TOML
    timings = TOML.parsefile(joinpath(ARGS[1], "timings.toml"))
    write(
        joinpath(ARGS[1], "sims_per_second.txt"),
        string(1 / timings["average_timing_s"]),
    )
    @info "Benchmark timings" timings' "$OUTDIR"
