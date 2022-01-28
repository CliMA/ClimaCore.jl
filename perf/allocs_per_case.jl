# Launch with `julia --project --track-allocation=user`
import Pkg
Pkg.develop(path = abspath(joinpath(@__DIR__, "..")))

import Profile

case_name = ENV["ALLOCATION_CASE_NAME"]
ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true
try
    include(case_name)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
Profile.clear_malloc_data()
OrdinaryDiffEq.step!(integrator)
