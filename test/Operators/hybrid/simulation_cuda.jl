include("utils_cuda.jl")
using OrdinaryDiffEq

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
