import Pkg
Pkg.develop(path = ".")

# Track allocations in ClimaCore.jl plus all _direct_ dependencies
exhaustive = "exhaustive=true" in ARGS
@show exhaustive

import ClimaCore

mod_dir(x) = dirname(dirname(pathof(x)))
pkg_dir = mod_dir(ClimaCore)

# (filename, ARGs passed to script)
#! format: off
all_cases = [
    (joinpath(pkg_dir, "examples", "hybrid", "bubble_2d.jl"), ""),
    (joinpath(pkg_dir, "examples", "hybrid", "bubble_3d.jl"), ""),
    (joinpath(pkg_dir, "examples", "3dsphere", "baroclinic_wave.jl"), "baroclinic_wave"),
    (joinpath(pkg_dir, "examples", "sphere", "shallow_water.jl"), "barotropic_instability"),
    (joinpath(pkg_dir, "examples", "3dsphere", "solid_body_rotation_3d.jl"), ""),
    (joinpath(pkg_dir, "examples", "3dsphere", "baroclinic_wave_rho_etot.jl"), ""),
]
#! format: on

# only one case for exhaustive alloc analysis
all_cases = if exhaustive
    [(joinpath(pkg_dir, "examples", "hybrid", "bubble_2d.jl"), "")]
else
    all_cases
end

import ReportMetrics

for (case, args) in all_cases
    ENV["ALLOCATION_CASE_NAME"] = case
    ReportMetrics.report_allocs(;
        job_name = "$(basename(case)) $args",
        run_cmd = `$(Base.julia_cmd()) --project=examples/ --track-allocation=all perf/allocs_per_case.jl $args`,
        dirs_to_monitor = [pkg_dir],
        n_unique_allocs = 20,
        process_filename = function process_fn(fn)
            fn = "ClimaCore.jl/" * last(split(fn, "climacore-ci/"))
            fn = last(split(fn, "depot/cpu/packages/"))
            return fn
        end,
    )
end
