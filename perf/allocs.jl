EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
EXAMPLE_DIR in LOAD_PATH || pushfirst!(LOAD_PATH, EXAMPLE_DIR)

# Track allocations in ClimaCore.jl plus some important dependencies:
import ClimaCore
import SciMLBase
import DiffEqBase
import OrdinaryDiffEq
import DiffEqOperators

dirs_to_monitor = [
    pkgdir(ClimaCore),
    pkgdir(SciMLBase),
    pkgdir(DiffEqBase),
    pkgdir(OrdinaryDiffEq),
    pkgdir(DiffEqOperators),
]

# (filename, ARGs passed to script)
#! format: off
all_cases = [
    (joinpath(EXAMPLE_DIR, "sphere", "shallow_water.jl"), "barotropic_instability", ""),
    (joinpath(EXAMPLE_DIR, "hybrid", "plane", "bubble_2d.jl"), "", ""),
    (joinpath(EXAMPLE_DIR, "hybrid", "box", "bubble_3d.jl"), "", ""),
    (joinpath(EXAMPLE_DIR, "hybrid", "driver.jl"), "", "sphere/baroclinic_wave_rhoe"),
    (joinpath(EXAMPLE_DIR, "hybrid", "driver.jl"), "", "sphere/baroclinic_wave_rhotheta"),
]
#! format: on

import ReportMetrics

for (case, args, test_name) in all_cases
    ENV["ALLOCATION_CASE_NAME"] = case
    ENV["TEST_NAME"] = test_name
    ReportMetrics.report_allocs(;
        job_name = "$(test_name == "" ? basename(case) : test_name) $args",
        run_cmd = `$(Base.julia_cmd()) --project=perf/ --track-allocation=all perf/allocs_per_case.jl $args`,
        dirs_to_monitor = dirs_to_monitor,
        n_unique_allocs = 20,
        process_filename = function process_fn(fn)
            fn = "ClimaCore.jl/" * last(split(fn, "climacore-ci/"))
            fn = last(split(fn, "depot/cpu/packages/"))
            return fn
        end,
    )
end
