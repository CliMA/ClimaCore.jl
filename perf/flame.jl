EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
EXAMPLE_DIR in LOAD_PATH || pushfirst!(LOAD_PATH, EXAMPLE_DIR)

import Profile

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

import ClimaCore

filename = joinpath(EXAMPLE_DIR, "hybrid", "plane", "bubble_2d_rhotheta.jl")

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

if !isempty(get(ENV, "CI_PERF_CPUPROFILE", ""))

    import ChromeProfileFormat
    output_path =
        mkpath(get(ENV, "CI_OUTPUT_DIR", joinpath(@__DIR__, "output")))
    testname = splitext(splitdir(filename)[end])[1]
    cpufile = testname * ".cpuprofile"
    ChromeProfileFormat.save_cpuprofile(joinpath(output_path, cpufile))

    if !isempty(get(ENV, "BUILDKITE", ""))
        import URIs

        print_link_url(url) = print("\033]1339;url='$(url)'\a\n")

        profiler_url(uri) = URIs.URI(
            "https://profiler.firefox.com/from-url/$(URIs.escapeuri(uri))",
        )

        # copy the file to the clima-ci bucket
        buildkite_pipeline = ENV["BUILDKITE_PIPELINE_SLUG"]
        buildkite_buildnum = ENV["BUILDKITE_BUILD_NUMBER"]
        buildkite_step = ENV["BUILDKITE_STEP_KEY"]

        profile_uri = "$buildkite_pipeline/build/$buildkite_buildnum/$buildkite_step/$cpufile"
        gs_profile_uri = "gs://clima-ci/$profile_uri"
        dl_profile_uri = "https://storage.googleapis.com/clima-ci/$profile_uri"

        #=
        # sync to bucket
        run(`gsutil cp $(joinpath(output_path, cpufile)) $gs_profile_uri`)

        # print link
        println("+++ Profiler link for '$profile_uri': ")
        print_link_url(profiler_url(dl_profile_uri))
        =#
    end
else
    import PProf
    PProf.pprof()
    # http://localhost:57599/ui/flamegraph?tf
end
