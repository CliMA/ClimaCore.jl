using Test
using StaticArrays
using ClimaComms, ClimaCore
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets
using CUDA
using OrdinaryDiffEq

function hvspace_3D_box(
    context,
    xlim = (-π, π),
    ylim = (-π, π),
    zlim = (0, 4π),
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 7,
)
    FT = Float64

    # Define vert domain and mesh
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)

    # Define vert topology and space
    verttopology = Topologies.IntervalTopology(context, vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(ylim[1]) .. Geometry.YPoint{FT}(ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)

    quad = Spaces.Quadratures.GLL{npoly + 1}()

    # Define horz topology and space
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

function hvspace_3D_sphere(context)
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
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return hv_center_space, hv_face_space
end

@testset "Finite difference GradientF2C CUDA" begin
    gpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    println("running test on $device device")

    # Define hv GPU space
    hv_center_space_gpu, hv_face_space_gpu = hvspace_3D_sphere(gpu_context)

    z = Fields.coordinate_field(hv_face_space_gpu).z

    gradc = Operators.GradientF2C()

    @test parent(Geometry.WVector.(gradc.(z))) ≈
          parent(Geometry.WVector.(ones(hv_center_space_gpu)))
end

@testset "2D SE, 1D FD Extruded Domain ∇ ODE Solve horizontal CUDA" begin

    # Advection Equation
    # ∂_t f + c ∂_x f  = 0
    # the solution translates to the right at speed c,
    # so if you you have a periodic domain of size [-π, π]
    # at time t, the solution is f(x - c * t, y)
    # here c == 1, integrate t == 2π or one full period

    function rhs!(dudt, u, _, t)
        # horizontal divergence operator applied to all levels
        hdiv = Operators.Divergence()
        @. dudt = -hdiv(u * Geometry.UVVector(1.0, 1.0))
        Spaces.weighted_dss!(dudt)
        return dudt
    end

    gpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    println("running test on $device device")

    hv_center_space_gpu, _ = hvspace_3D_box(gpu_context)
    U = sin.(Fields.coordinate_field(hv_center_space_gpu).x)
    dudt = zeros(eltype(U), hv_center_space_gpu)
    rhs!(dudt, U, nothing, 0.0)

    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(prob, SSPRK33(), dt = Δt)

    @test Array(parent(U)) ≈ Array(parent(sol.u[end])) rtol = 1e-6
end
