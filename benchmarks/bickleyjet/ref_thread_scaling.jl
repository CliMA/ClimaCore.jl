push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using BenchmarkTools, Plots

proj = Base.active_project()
julia_cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(proj)`
run_script = joinpath(@__DIR__, "run_ref_thread.jl")

thread_counts = [1, 2, 4, 8, 16]
runtimes = Float64[]
result = IOBuffer()
for n in thread_counts
    cmd = deepcopy(julia_cmd)
    append!(cmd.exec, ["--threads=$(n)", run_script])
    Base.run(pipeline(cmd, stdout = result); wait = true)
    push!(runtimes, parse(Float64, String(take!(result))))
end

using Plots
ENV["GKSwstype"] = "nul"

plt = plot(
    xlabel = "#threads",
    ylabel = "time (secs)",
    title = "Volume thread scaling",
)
plot!(plt, thread_counts, runtimes, label = "Runtimes w/threads")
png(plt, joinpath(@__DIR__, "vol_thread_scaling.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("benchmarks/bickleyjet/vol_thread_scaling.png", "Volume thread scaling")
