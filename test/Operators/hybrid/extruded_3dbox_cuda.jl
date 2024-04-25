#=
julia --project
using Revise; include(joinpath("test", "Spaces", "extruded_3dbox_cuda.jl"))
=#
using LinearAlgebra, IntervalSets
using CUDA
using ClimaComms
if pkgversion(ClimaComms) >= v"0.6"
    ClimaComms.@import_required_backends
end, ClimaCore
import ClimaCore:
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Geometry,
    column,
    Fields,
    Operators,
    Quadratures
using Test

function get_space(context)
    FT = Float64

    # Define vert domain and mesh
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)

    # Define vert topology and space
    verttopology = Topologies.IntervalTopology(context, vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

    # Define horz domain and mesh
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic = true,
    )
    horzmesh = Meshes.RectilinearMesh(horzdomain, 17, 16)
    quad = Quadratures.GLL{3 + 1}()

    # Define horz topology and space
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    # Define hv spaces
    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    return hv_center_space
end

@testset "GPU extruded 3d hybrid box strong grad, div and curl" begin
    cpu_context =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    gpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    println("running test on $device device")

    # Define hv CPU space
    hv_center_space_cpu = get_space(cpu_context)

    # Define hv GPU space
    hv_center_space_gpu = get_space(gpu_context)

    # Define CPU & GPU coordinate fields
    coords_cpu = Fields.coordinate_field(hv_center_space_cpu)
    coords_gpu = Fields.coordinate_field(hv_center_space_gpu)

    # Define CPU & GPU fields
    x_cpu =
        Geometry.UVWVector.(
            sin.(coords_cpu.x .+ 2 .* coords_cpu.y .+ 3 .* coords_cpu.z),
            0.0,
            0.0,
        )
    f_cpu = sin.(coords_cpu.x .+ 2 .* coords_cpu.y .+ 3 .* coords_cpu.z)
    g_cpu =
        Geometry.UVWVector.(
            sin.(coords_cpu.x),
            2 .* cos.(coords_cpu.y .+ coords_cpu.x .+ coords_cpu.z),
            cos.(coords_cpu.z),
        )
    x_gpu =
        Geometry.UVWVector.(
            sin.(coords_gpu.x .+ 2 .* coords_gpu.y .+ 3 .* coords_gpu.z),
            0.0,
            0.0,
        )
    f_gpu = sin.(coords_gpu.x .+ 2 .* coords_gpu.y .+ 3 .* coords_gpu.z)
    g_gpu =
        Geometry.UVWVector.(
            sin.(coords_gpu.x),
            2 .* cos.(coords_gpu.y .+ coords_gpu.x .+ coords_gpu.z),
            cos.(coords_gpu.z),
        )

    # Test grad operator
    grad = Operators.Gradient()
    @test parent(grad.(f_cpu)) ≈ Array(parent(grad.(f_gpu)))

    # Test div operator
    div = Operators.Divergence()
    @test parent(div.(x_cpu)) ≈ Array(parent(div.(x_gpu)))

    # Test curl operator
    curl = Operators.Curl()
    @test parent(curl.(Geometry.Covariant12Vector.(g_cpu))) ≈
          Array(parent(curl.(Geometry.Covariant12Vector.(g_gpu))))
    @test parent(curl.(Geometry.Covariant3Vector.(f_cpu))) ≈
          Array(parent(curl.(Geometry.Covariant3Vector.(f_gpu))))
end

@testset "GPU extruded 3d hybrid box weak grad, div, curl and wdiv(grad)" begin
    cpu_context =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    gpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    println("running test on $device device")

    # Define hv CPU spaces
    hv_center_space_cpu = get_space(cpu_context)

    # Define hv GPU spaces
    hv_center_space_gpu = get_space(gpu_context)

    # Define CPU & GPU coordinate fields
    coords_cpu = Fields.coordinate_field(hv_center_space_cpu)
    coords_gpu = Fields.coordinate_field(hv_center_space_gpu)

    # Define CPU & GPU fields
    x_cpu =
        Geometry.UVWVector.(
            sin.(coords_cpu.x .+ 2 .* coords_cpu.y .+ 3 .* coords_cpu.z),
            0.0,
            0.0,
        )
    f_cpu = sin.(coords_cpu.x .+ 2 .* coords_cpu.y .+ 3 .* coords_cpu.z)
    g_cpu =
        Geometry.UVWVector.(
            sin.(coords_cpu.x),
            2 .* cos.(coords_cpu.y .+ coords_cpu.x .+ coords_cpu.z),
            cos.(coords_cpu.z),
        )
    x_gpu =
        Geometry.UVWVector.(
            sin.(coords_gpu.x .+ 2 .* coords_gpu.y .+ 3 .* coords_gpu.z),
            0.0,
            0.0,
        )
    f_gpu = sin.(coords_gpu.x .+ 2 .* coords_gpu.y .+ 3 .* coords_gpu.z)
    g_gpu =
        Geometry.UVWVector.(
            sin.(coords_gpu.x),
            2 .* cos.(coords_gpu.y .+ coords_gpu.x .+ coords_gpu.z),
            cos.(coords_gpu.z),
        )

    CUDA.allowscalar(false)

    # Test weak grad operator
    wgrad = Operators.WeakGradient()
    @test parent(wgrad.(f_cpu)) ≈ Array(parent(wgrad.(f_gpu)))

    # Test weak div operator
    wdiv = Operators.WeakDivergence()
    @test parent(wdiv.(x_cpu)) ≈ Array(parent(wdiv.(x_gpu)))

    # Test weak curl operator
    wcurl = Operators.WeakCurl()
    @test parent(wcurl.(Geometry.Covariant12Vector.(g_cpu))) ≈
          Array(parent(wcurl.(Geometry.Covariant12Vector.(g_gpu))))
    @test parent(wcurl.(Geometry.Covariant3Vector.(f_cpu))) ≈
          Array(parent(wcurl.(Geometry.Covariant3Vector.(f_gpu))))

    # Test wdiv(grad()) composed Laplace operator
    grad = Operators.Gradient()
    @test parent(wdiv.(grad.(f_cpu))) ≈ Array(parent(wdiv.(grad.(f_gpu))))
    @test_broken parent(wdiv.(grad.(x_cpu))) ≈
                 Array(parent(wdiv.(grad.(x_gpu))))

    # Test DSS
    @test parent(Spaces.weighted_dss!(wdiv.(grad.(f_cpu)))) ≈
          Array(parent(Spaces.weighted_dss!(wdiv.(grad.(f_gpu)))))
end
