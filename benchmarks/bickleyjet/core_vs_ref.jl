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
    y0_ref = init_y0_ref(X, Val(Nq), parameters)
    dydt_ref = similar(y0_ref)
    tendency_states = init_tendency_states(n1, n2, Val(Nq))
    volume_ref!(
        dydt_ref,
        y0_ref,
        (n1, n2, parameters, Val(Nq), tendency_states),
        0.0,
    )

    # check equivalent
    @assert y0_ref ≈ reshape(parent(y0), (Nq, Nq, 4, n1, n2))
    @assert dydt_ref ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    # setup GPU reference
    if DA === CuArray
        y0_gpu = DA(y0_ref)
        dydt_gpu = similar(y0_gpu)
        volume_ref_cuda!(dydt_gpu, y0_gpu, parameters, Nq)
        # check equivalent
        @assert Array(y0_gpu) ≈ y0_ref
        @assert Array(dydt_gpu) ≈ dydt_ref
    end
    # run benchmarks
    @info("Benchmark volume!", Nq)
    push!(volTs, @belapsed volume!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark volume_ref!", Nq)
    push!(
        volRs,
        @belapsed volume_ref!(
            $dydt_ref,
            $y0_ref,
            ($n1, $n2, $parameters, $(Val(Nq)), $tendency_states),
            0.0,
        )
    )
    if DA === CuArray
        @info("Benchmark volume_ref GPU!", Nq)
        push!(
            volRGPUs,
            @belapsed volume_ref_cuda!($dydt_gpu, $y0_gpu, $parameters, $Nq)
        )
    end

    # faces
    fill!(parent(dydt), 0.0)
    add_face!(dydt, y0, (parameters,), 0.0)
    # TODO: move this to volume!
    dydt_data = Fields.field_values(dydt)
    dydt_data .= RecursiveApply.rdiv.(dydt_data, space.local_geometry.WJ)

    fill!(dydt_ref, 0.0)
    add_face_ref!(dydt_ref, y0_ref, (n1, n2, parameters, Val(Nq)), 0.0)

    @assert dydt_ref ≈ reshape(parent(dydt), (Nq, Nq, 4, n1, n2))

    # setup GPU face reference
    if DA === CuArray
        fill!(dydt_gpu, 0.0)
        add_face_ref_cuda!(dydt_gpu, y0_gpu, parameters, Nq)
        @assert Array(y0_gpu) ≈ y0_ref
        @assert Array(dydt_gpu) ≈ dydt_ref
    end
    @info("Benchmark face!", Nq)
    push!(faceTs, @belapsed add_face!($dydt, $y0, ($parameters,), 0.0))
    @info("Benchmark face_ref!", Nq)
    push!(
        faceRs,
        @belapsed add_face_ref!(
            $dydt_ref,
            $y0_ref,
            ($n1, $n2, $parameters, $(Val(Nq))),
            0.0,
        )
    )
    if DA === CuArray
        @info("Benchmark face_ref GPU!", Nq)
        push!(
            faceRGPUs,
            @belapsed add_face_ref_cuda!($dydt_gpu, $y0_gpu, $parameters, $Nq)
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
