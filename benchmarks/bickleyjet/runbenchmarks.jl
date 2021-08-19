push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using BenchmarkTools, Plots
using CUDA

include("bickleyjet_dg.jl")
include("bickleyjet_dg_reference.jl")


n1, n2 = 16, 16

Nqs = 2:7
volTs = Float64[]
volRs = Float64[]
volRGPUs = Float64[]
faceTs = Float64[]
faceRs = Float64[]
faceRGPUs = Float64[]

DA = CUDA.has_cuda_gpu() ? CuArray : Array

for Nq in Nqs
    # setup core
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))
    dydt = Fields.Field(similar(Fields.field_values(y0)), space)
    volume!(dydt, y0, (parameters,), 0.0)
    # TODO: move this to volume!
    dydt_data = Fields.field_values(dydt)
    dydt_data .= RecursiveApply.rdiv.(dydt_data, space.local_geometry.WJ)

    # setup reference
    X = coordinates(Val(Nq), n1, n2)
    Y0 = init_Y(X, Val(Nq), parameters)
    dYdt = similar(Y0)
    volume_ref!(dYdt, Y0, (parameters, Val(Nq)), 0.0)

    # check equivalent
    @assert Y0 ≈ reshape(parent(y0), (Nq, Nq, 4, n1, n2))
    @assert dYdt ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    # setup GPU reference
    if DA === CuArray
        Y0_GPU = DA(Y0)
        dYdt_GPU = similar(Y0_GPU)
        volume_ref_CUDA!(dYdt_GPU, Y0_GPU, parameters, Nq)
        # check equivalent
        @assert Array(Y0_GPU) ≈ Y0
        @assert Array(dYdt_GPU) ≈ dYdt
    end
    # run benchmarks
    @info("Benchmark volume!", Nq)
    push!(volTs, @belapsed volume!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark volume_ref!", Nq)
    push!(
        volRs,
        @belapsed volume_ref!($dYdt, $Y0, ($parameters, $(Val(Nq))), 0.0)
    )
    if DA === CuArray
        @info("Benchmark volume_ref GPU!", Nq)
        push!(
            volRGPUs,
            @belapsed volume_ref_CUDA!($dYdt_GPU, $Y0_GPU, $parameters, $Nq)
        )
    end

    # faces
    fill!(parent(dydt), 0.0)
    add_face!(dydt, y0, (parameters,), 0.0)
    # TODO: move this to volume!
    dydt_data = Fields.field_values(dydt)
    dydt_data .= RecursiveApply.rdiv.(dydt_data, space.local_geometry.WJ)

    fill!(dYdt, 0.0)
    add_face_ref!(dYdt, Y0, (parameters, Val(Nq)), 0.0)

    @assert dYdt ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    # setup GPU face reference
    if DA === CuArray
        fill!(dYdt_GPU, 0.0)
        add_face_ref_CUDA!(dYdt_GPU, Y0_GPU, parameters, Nq)
        @assert Array(Y0_GPU) ≈ Y0
        @assert Array(dYdt_GPU) ≈ dYdt
    end
    @info("Benchmark face!", Nq)
    push!(faceTs, @belapsed add_face!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark face_ref!", Nq)
    push!(
        faceRs,
        @belapsed add_face_ref!($dYdt, $Y0, ($parameters, $(Val(Nq))), 0.0)
    )
    if DA === CuArray
        @info("Benchmark face_ref GPU!", Nq)
        push!(
            faceRGPUs,
            @belapsed add_face_ref_CUDA!($dYdt_GPU, $Y0_GPU, $parameters, $Nq)
        )
    end
end


using Plots
ENV["GKSwstype"] = "nul"

plt = plot(
    ylims = (0, Inf),
    xlabel = "Nq",
    ylabel = "Time (ms)",
    title = "Volume",
)
plot!(plt, Nqs, 1e3 .* volTs, label = "ClimaCore")
plot!(plt, Nqs, 1e3 .* volRs, label = "Reference")
if DA === CuArray
    plot!(plt, Nqs, 1e3 .* volRGPUs, label = "Reference GPU")
end

png(plt, joinpath(@__DIR__, "volume.png"))


plt =
    plot(ylims = (0, Inf), xlabel = "Nq", ylabel = "Time (ms)", title = "Face")
plot!(plt, Nqs, 1e3 .* faceTs, label = "ClimaCore")
plot!(plt, Nqs, 1e3 .* faceRs, label = "Reference")
if DA === CuArray
    plot!(plt, Nqs, 1e3 .* faceRGPUs, label = "Reference GPU")
end

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
