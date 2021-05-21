push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

include("bickleyjet_dg.jl")
include("bickleyjet_dg_reference.jl")

using BenchmarkTools, Plots

n1, n2 = 16, 16

Nqs = 2:7
volTs = Float64[]
volRs = Float64[]
faceTs = Float64[]
faceRs = Float64[]

for Nq in Nqs
    # setup core
    discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
    grid_topology = Topologies.GridTopology(discretization)
    quad = Meshes.Quadratures.GLL{Nq}()
    mesh = Meshes.Mesh2D(grid_topology, quad)

    y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))
    dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
    volume!(dydt, y0, (parameters,), 0.0)
    # TODO: move this to volume!
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, mesh.local_geometry.WJ)

    # setup reference
    X = coordinates(Val(Nq), n1, n2)
    Y0 = init_Y(X, Val(Nq), parameters)
    dYdt = similar(Y0)
    volume_ref!(dYdt, Y0, (parameters, Val(Nq)), 0.0)

    # check equivalent
    @assert Y0 ≈ reshape(parent(y0), (Nq, Nq, 4, n1, n2))
    @assert dYdt ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    # run benchmarks
    @info("Benchmark volume!", Nq)
    push!(volTs, @belapsed volume!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark volume_ref!", Nq)
    push!(
        volRs,
        @belapsed volume_ref!($dYdt, $Y0, ($parameters, $(Val(Nq))), 0.0)
    )

    # faces
    fill!(parent(dydt), 0.0)
    add_face!(dydt, y0, (parameters,), 0.0)
    # TODO: move this to volume!
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, mesh.local_geometry.WJ)

    fill!(dYdt, 0.0)
    add_face_ref!(dYdt, Y0, (parameters, Val(Nq)), 0.0)

    @assert dYdt ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    @info("Benchmark face!", Nq)
    push!(faceTs, @belapsed add_face!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark face_ref!", Nq)
    push!(
        faceRs,
        @belapsed add_face_ref!($dYdt, $Y0, ($parameters, $(Val(Nq))), 0.0)
    )
end


using Plots
ENV["GKSwstype"] = "nul"

plt = plot(
    ylims = (0, Inf),
    xlabel = "Nq",
    ylabel = "Time (ms)",
    title = "Volume",
)
plot!(plt, Nqs, 1e3 .* volTs, label = "ClimateMachineCore")
plot!(plt, Nqs, 1e3 .* volRs, label = "Reference")

png(plt, joinpath(@__DIR__, "volume.png"))


plt =
    plot(ylims = (0, Inf), xlabel = "Nq", ylabel = "Time (ms)", title = "Face")
plot!(plt, Nqs, 1e3 .* faceTs, label = "ClimateMachineCore")
plot!(plt, Nqs, 1e3 .* faceRs, label = "Reference")

png(plt, joinpath(@__DIR__, "face.png"))


function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("benchmarks/bickleyjet/volume.png", "Volume benchmarks")
linkfig("benchmarks/bickleyjet/face.png", "Face benchmarks")
