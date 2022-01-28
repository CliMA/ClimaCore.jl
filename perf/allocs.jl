import Pkg
Pkg.develop(path = abspath(joinpath(@__DIR__, "..")))

# Track allocations in ClimaCore.jl plus some important dependencies:
import ClimaCore
import SciMLBase
import DiffEqBase
import OrdinaryDiffEq
import DiffEqOperators

mod_dir(x) = dirname(dirname(pathof(x)))
pkg_dir = mod_dir(ClimaCore)
dirs_to_monitor = [
    pkg_dir,
    mod_dir(SciMLBase),
    mod_dir(DiffEqBase),
    mod_dir(OrdinaryDiffEq),
    mod_dir(DiffEqOperators),
]

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
