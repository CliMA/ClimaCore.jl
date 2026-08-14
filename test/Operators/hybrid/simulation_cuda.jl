include("utils_cuda.jl")
import ClimaTimeSteppers as CTS

@testset "2D SE, 1D FD Extruded Domain ∇ ODE Solve horizontal CUDA" begin

    # Advection Equation
    # ∂_t f + c ∂_x f  = 0
    # the solution translates to the right at speed c,
    # so if you you have a periodic domain of size [-π, π]
    # at time t, the solution is f(x - c * t, y)
    # here c == 1, integrate t == 2π or one full period

    function rhs!(dudt, u, _, t)
        # Horizontal divergence operator applied to all levels
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
    Δt = 0.01
    # The integrator advances the state it is handed in place, so it gets a
    # copy: `U` has to stay the initial condition for the comparison below to
    # mean anything.
    prob = CTS.ODEProblem(
        CTS.ClimaODEFunction(; T_exp! = rhs!),
        copy(U),
        (0.0, 2π),
        nothing,
    )
    sol = CTS.solve(prob, CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()), dt = Δt)

    @test Array(parent(U)) ≈ Array(parent(sol.u[end])) rtol = 1e-6
end
