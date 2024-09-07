EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")

import Profile

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

import ClimaCore

filename =
    joinpath(EXAMPLE_DIR, "hybrid", "plane", "bubble_2d_invariant_rhoe.jl")

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

using SciMLBase: step!, solve!
step!(integrator) # compile first
Profile.clear_malloc_data()
prof = Profile.@profile begin
    for _ in 1:100
        step!(integrator)
    end
end

import ProfileCanvas

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    ProfileCanvas.html_file(joinpath(output_dir, "flame.html"))
else
    ProfileCanvas.view(Profile.fetch())
end
