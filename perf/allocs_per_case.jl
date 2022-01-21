# Launch with `julia --project --track-allocation=user`
import Pkg
Pkg.develop(path = ".")

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

if @isdefined parameters
    rhs!(dYdt, Y, parameters, 0.0) # compile first
    Profile.clear_malloc_data()
    rhs!(dYdt, Y, parameters, 0.0)
else
    rhs!(dYdt, Y, nothing, 0.0) # compile first
    Profile.clear_malloc_data()
    rhs!(dYdt, Y, nothing, 0.0)
end

# Quit julia (which generates .mem files), then call
#=
import Coverage
allocs = Coverage.analyze_malloc("src")
=#
