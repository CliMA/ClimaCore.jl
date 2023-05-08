#=
julia --project
using Revise; include(joinpath("test", "Spaces", "extruded_sphere_cuda.jl"))
=#
using LinearAlgebra, IntervalSets, UnPack
using CUDA
using ClimaComms, ClimaCore
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, column, Fields, Operators
using Test

function get_space(context)
    FT = Float64

    # Define vert domain and mesh
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)

    # Define vert topology and space
    verttopology = Topologies.IntervalTopology(context, vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

    # Define horz domain and mesh
    horzdomain = Domains.SphereDomain(FT(30.0))
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 4)

    quad = Spaces.Quadratures.GLL{3 + 1}()

    # Define horz topology and space
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    # Define hv spaces
    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    return hv_center_space
end

@testset "GPU extruded sphere strong grad, div and curl" begin
    cpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice())
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
    x_cpu = Geometry.UVWVector.(cosd.(coords_cpu.lat), 0.0, 0.0)
    f_cpu = sin.(coords_cpu.lat .+ 2 .* coords_cpu.long)
    g_cpu =
        Geometry.UVVector.(
            sin.(coords_cpu.lat),
            2 .* cos.(coords_cpu.long .+ coords_cpu.lat),
        )
    x_gpu = Geometry.UVWVector.(cosd.(coords_gpu.lat), 0.0, 0.0)
    CUDA.allowscalar(false)
    f_gpu = sin.(coords_gpu.lat .+ 2 .* coords_gpu.long)
    g_gpu =
        Geometry.UVVector.(
            sin.(coords_gpu.lat),
            2 .* cos.(coords_gpu.long .+ coords_gpu.lat),
        )

    # Test grad operator
    grad = Operators.Gradient()
    @test parent(grad.(f_cpu)) ≈ Array(parent(grad.(f_gpu)))

    # Test div operator
    div = Operators.Divergence()
    @test parent(div.(x_cpu)) ≈ Array(parent(div.(x_cpu)))

    # Test curl operator
    curl = Operators.Curl()
    @test parent(curl.(Geometry.Covariant12Vector.(g_cpu))) ≈
          Array(parent(curl.(Geometry.Covariant12Vector.(g_gpu))))
    @test parent(curl.(Geometry.Covariant3Vector.(f_cpu))) ≈
          Array(parent(curl.(Geometry.Covariant3Vector.(f_gpu))))
end

@testset "GPU extruded sphere weak grad, div, curl, and wdiv(grad)" begin
    cpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice())
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
    x_cpu = Geometry.UVWVector.(cosd.(coords_cpu.lat), 0.0, 0.0)
    f_cpu = sin.(coords_cpu.lat .+ 2 .* coords_cpu.long)
    g_cpu =
        Geometry.UVVector.(
            sin.(coords_cpu.lat),
            2 .* cos.(coords_cpu.long .+ coords_cpu.lat),
        )
    x_gpu = Geometry.UVWVector.(cosd.(coords_gpu.lat), 0.0, 0.0)
    CUDA.allowscalar(false)
    f_gpu = sin.(coords_gpu.lat .+ 2 .* coords_gpu.long)
    g_gpu =
        Geometry.UVVector.(
            sin.(coords_gpu.lat),
            2 .* cos.(coords_gpu.long .+ coords_gpu.lat),
        )

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
end
