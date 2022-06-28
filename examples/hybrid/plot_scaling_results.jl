using Plots
using JLD2
using Plots.PlotMeasures

I = Int
FT = Float32

test_name = "sphere/held_suarez_rhotheta_scaling"
domain, case_name = split(test_name, "/")
FT_string = string(FT)
path = joinpath(@__DIR__, domain, "output", case_name, FT_string)

nprocs = Int[]
walltime = FT[]

for filename in readdir(path)
    if occursin("scaling_data_", filename)
        dict = load(joinpath(path, filename))
        push!(nprocs, I(dict["nprocs"]))
        push!(walltime, FT(dict["walltime"]))
    end
end

order = sortperm(nprocs)
nprocs, walltime = nprocs[order], walltime[order]
nprocs_string = [string(i) for i in nprocs]

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

figpath = joinpath(path, "Scaling.png")

Plots.GRBackend()
Plots.png(
    plot(
        log2.(nprocs),
        walltime,
        markershape = :circle,
        xticks = (log2.(nprocs), nprocs_string),
        xlabel = "nprocs",
        ylabel = "wall time (sec)",
        title = "Scaling data",
        label = "simulation time = 1 hour",
        legend = :topright,
        grid = :true,
        margin = 15mm,
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Data")
