using Plots
using Plots.PlotMeasures
using JLD2
@assert length(ARGS) ≥ 1
case = get(ARGS, 1, "shallow_water")
if occursin("hybrid3dcubedsphere", case)
    resolution = get(ARGS, 2, "low")
    if occursin("low", resolution)
        output_dir = joinpath(Base.@__DIR__, "hybrid", "dss_output_low_res")
        output_file = "dss_scaling_low_res_3dcs"
    else
        output_dir = joinpath(Base.@__DIR__, "hybrid", "dss_output_high_res")
        output_file = "dss_scaling_high_res_3dcs"
    end
    @show resolution
else
    output_dir = joinpath(Base.@__DIR__, "sphere", "output")
    output_file = "dss_scaling_shallow_water"
end
poly = (2, 4)
npoly = length(poly)
# read ClimaAtmos scaling data
I, FT = Int, Float64
nprocs = map(i -> Vector{I}(undef, 0), zeros(I, npoly))
walltime_dss_full = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))
walltime_dss2_full = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))
walltime_dss_internal = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))
walltime_dss_comms = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))
walltime_dss_comms_fsb = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))
walltime_dss_comms_other = map(i -> Vector{FT}(undef, 0), zeros(I, npoly))

for filename in readdir(output_dir)
    if occursin("dss_scaling_data", filename)
        np = parse(I, split(split(filename, "npoly_")[end], ".")[1])
        loc = findfirst(poly .== np)
        if !isnothing(loc)
            dict = load(joinpath(output_dir, filename))
            push!(nprocs[loc], I(dict["nprocs"]))
            push!(walltime_dss_full[loc], FT(dict["walltime_dss_full"]))
            push!(walltime_dss2_full[loc], FT(dict["walltime_dss2_full"]))
            push!(walltime_dss_internal[loc], FT(dict["walltime_dss_internal"]))
            push!(walltime_dss_comms[loc], FT(dict["walltime_dss_comms"]))
            push!(
                walltime_dss_comms_fsb[loc],
                FT(dict["walltime_dss_comms_fsb"]),
            )
            push!(
                walltime_dss_comms_other[loc],
                FT(dict["walltime_dss_comms_other"]),
            )
        end
    end
end

log2nprocs = deepcopy(nprocs)
for i in 1:npoly
    order = sortperm(nprocs[i])
    nprocs[i] = nprocs[i][order]
    walltime_dss_full[i] = walltime_dss_full[i][order]
    walltime_dss2_full[i] = walltime_dss2_full[i][order]
    walltime_dss_internal[i] = walltime_dss_internal[i][order]
    walltime_dss_comms[i] = walltime_dss_comms[i][order]
    walltime_dss_comms_fsb[i] = walltime_dss_comms_fsb[i][order]
    walltime_dss_comms_other[i] = walltime_dss_comms_other[i][order]
    log2nprocs[i] .= log2.(nprocs[i])
end
xticksstr = (log2nprocs[1], [string(i) for i in nprocs[1]])
kwargs = (
    markershape = [:circle :square],
    markercolor = [:blue :green],
    label = ["npoly = 2" "npoly = 4"],
    xticks = xticksstr,
    yaxis = :log,
    ylabel = "walltime (sec)",
    xlabel = "# of procs",
    legendfontsize = 3,
    xlabelfontsize = 6,
    ylabelfontsize = 6,
    titlefontsize = 8,
    left_margin = 5mm,
    bottom_margin = 5mm,
    top_margin = 5mm,
)
plots = Plots.Plot{Plots.GRBackend}[
    plot(
        [log2nprocs[1] log2nprocs[2]],
        [walltime_dss_full[1] walltime_dss_full[2]],
        title = "Full DSS",
        legend = :topleft;
        kwargs...,
    ),
    plot(
        [log2nprocs[1] log2nprocs[2]],
        [walltime_dss_internal[1] walltime_dss_internal[2]],
        title = "Internal DSS",
        legend = :bottomleft;
        kwargs...,
    ),
    plot(
        [log2nprocs[1] log2nprocs[2]],
        [walltime_dss_comms[1] walltime_dss_comms[2]],
        title = "Communication only",
        legend = :bottom;
        kwargs...,
    ),
    plot(
        [log2nprocs[1] log2nprocs[2]],
        [walltime_dss_comms_fsb[1] walltime_dss_comms_fsb[2]],
        title = "fill send buffer",
        legend = :bottom;
        kwargs...,
    ),
    plot(
        [log2nprocs[1] log2nprocs[2]],
        [walltime_dss_comms_other[1] walltime_dss_comms_other[2]],
        title = "Start and end",
        legend = :topleft;
        kwargs...,
    ),
]
Plots.plot(plots..., layout = @layout([° °; ° °; ° _]))
savefig(joinpath(output_dir, "$output_file.png"))
savefig(joinpath(output_dir, "$output_file.pdf"))

kwargs = (
    markershape = :circle,
    markercolor = :blue,
    label = "Δt Np4/Δt Np2",
    xticks = xticksstr,
    ylim = (0.0, Inf),
    ylabel = "speedup",
    xlabel = "# of procs",
    legendfontsize = 3,
    legend = :bottom,
    xlabelfontsize = 6,
    ylabelfontsize = 6,
    titlefontsize = 8,
    left_margin = 5mm,
    bottom_margin = 5mm,
    top_margin = 5mm,
)

plots = Plots.Plot{Plots.GRBackend}[
    plot(
        log2nprocs[1],
        walltime_dss_full[2] ./ walltime_dss_full[1],
        title = "Full DSS";
        kwargs...,
    ),
    plot(
        log2nprocs[1],
        walltime_dss_internal[2] ./ walltime_dss_internal[1],
        title = "Internal DSS";
        kwargs...,
    ),
    plot(
        log2nprocs[1],
        walltime_dss_comms[2] ./ walltime_dss_comms[1],
        title = "Communication only";
        kwargs...,
    ),
    plot(
        log2nprocs[1],
        walltime_dss_comms_fsb[2] ./ walltime_dss_comms_fsb[1],
        title = "fill send buffer";
        kwargs...,
    ),
    plot(
        log2nprocs[1],
        walltime_dss_comms_other[2] ./ walltime_dss_comms_other[1],
        title = "Start and end";
        kwargs...,
    ),
]
output_file = replace(output_file, "scaling" => "comparison")
Plots.plot(plots..., layout = @layout([° °; ° °; ° _]))
savefig(joinpath(output_dir, "$output_file.png"))
savefig(joinpath(output_dir, "$output_file.pdf"))


kwargs = (
    xticks = xticksstr,
    ylim = (0.0, Inf),
    xlabel = "# of procs",
    legendfontsize = 6,
    xlabelfontsize = 6,
    ylabelfontsize = 6,
    titlefontsize = 8,
    left_margin = 5mm,
    bottom_margin = 5mm,
    top_margin = 5mm,
)

plots = Plots.Plot{Plots.GRBackend}[
    plot(
        [log2nprocs[1] log2nprocs[1]],
        [walltime_dss_full[1] walltime_dss2_full[1]],
        title = "DSS vs DSS2 (npoly = 2)",
        label = ["current DSS" "DSS2"],
        ylabel = "walltime (sec)",
        markershape = [:circle :square],
        markercolor = [:blue :green],
        legend = :topright;
        kwargs...,
    ),
    plot(
        [log2nprocs[2] log2nprocs[2]],
        [walltime_dss_full[2] walltime_dss2_full[2]],
        title = "DSS vs DSS2 (npoly = 4)",
        label = ["current DSS" "DSS2"],
        ylabel = "walltime (sec)",
        markershape = [:circle :square],
        markercolor = [:blue :green],
        legend = :topright;
        kwargs...,
    ),
    plot(
        log2nprocs[1],
        walltime_dss_full[1] ./ walltime_dss2_full[1],
        title = "Speedup DSS vs DSS2 (npoly = 2)",
        label = "walltime DSS/DSS2",
        ylabel = "walltime (DSS/DSS2)",
        markershape = :circle,
        markercolor = :blue,
        legend = :topleft;
        kwargs...,
    ),
    plot(
        log2nprocs[2],
        walltime_dss_full[2] ./ walltime_dss2_full[2],
        title = "Speedup DSS vs DSS2 (npoly = 4)",
        label = "walltime DSS/DSS2",
        ylabel = "walltime (DSS/DSS2)",
        markershape = :circle,
        markercolor = :blue,
        legend = :topleft;
        kwargs...,
    ),
]
output_file = replace(output_file, "comparison" => "compare_dss_dss2")
Plots.plot(plots..., layout = @layout([° °; ° °]))
savefig(joinpath(output_dir, "$output_file.png"))
savefig(joinpath(output_dir, "$output_file.pdf"))
