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
    (joinpath(EXAMPLE_DIR, "hybrid", "bubble_2d.jl"), ""),
    (joinpath(EXAMPLE_DIR, "hybrid", "bubble_3d.jl"), ""),
    (joinpath(EXAMPLE_DIR, "3dsphere", "baroclinic_wave.jl"), "baroclinic_wave"),
    (joinpath(EXAMPLE_DIR, "sphere", "shallow_water.jl"), "barotropic_instability"),
    (joinpath(EXAMPLE_DIR, "3dsphere", "solid_body_rotation_3d.jl"), ""),
    (joinpath(EXAMPLE_DIR, "3dsphere", "baroclinic_wave_rho_etot.jl"), ""),
]
#! format: on

import ReportMetrics

for (case, args) in all_cases
    ENV["ALLOCATION_CASE_NAME"] = case
    ReportMetrics.report_allocs(;
        job_name = "$(basename(case)) $args",
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
