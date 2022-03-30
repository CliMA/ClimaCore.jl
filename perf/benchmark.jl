EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
EXAMPLE_DIR in LOAD_PATH || pushfirst!(LOAD_PATH, EXAMPLE_DIR)

import Profile
import BenchmarkTools
ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

import ClimaCore

filename = joinpath(EXAMPLE_DIR, "hybrid", "driver.jl")
ENV["TEST_NAME"] = "sphere/baroclinic_wave_rhoe"

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
trial = BenchmarkTools.@benchmark OrdinaryDiffEq.step!($integrator)
show(stdout, MIME("text/plain"), trial)
println()
