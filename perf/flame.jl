import Pkg
Pkg.develop(path = ".")

mod_dir(x) = dirname(dirname(pathof(x)))
import Profile

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

import ClimaCore
pkg_dir = mod_dir(ClimaCore)

filename = joinpath(pkg_dir, "examples", "hybrid", "bubble_2d.jl")
# filename = joinpath(pkg_dir, "examples", "3dsphere", "baroclinic_wave_rho_etot.jl")

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
Profile.clear_malloc_data()
prof = Profile.@profile begin
    for _ in 1:5
        OrdinaryDiffEq.step!(integrator)
    end
end

import PProf
PProf.pprof()
# http://localhost:57599/ui/flamegraph?tf
