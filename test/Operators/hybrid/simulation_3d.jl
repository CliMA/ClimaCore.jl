include("utils_3d.jl")

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
device = ClimaComms.device()

@testset "2D SE, 1D FV Extruded Domain ∇ ODE Solve vertical" begin

    hv_center_space, hv_face_space =
        hvspace_3D(; device = ClimaComms.CPUSingleThreaded())
    V =
        Geometry.UVWVector.(
            zeros(Float64, hv_face_space),
            zeros(Float64, hv_face_space),
            ones(Float64, hv_face_space),
        )

    function rhs!(dudt, u, _, t)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(sin(-t)),
            top = Operators.Extrapolate(),
        )
        return @. dudt = -A(V, u)
    end

    U = sin.(Fields.coordinate_field(hv_center_space).z)
    dudt = zeros(eltype(U), hv_center_space)
    rhs!(dudt, U, nothing, 0.0)

    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(prob, SSPRK33(), dt = Δt)

    htopo = Spaces.topology(hv_center_space)
    for h in 1:Topologies.nlocalelems(htopo)
        sol_column_field = ClimaCore.column(sol.u[end], 1, 1, h)
        ref_column_field = ClimaCore.column(U, 1, 1, h)
        @test sol_column_field ≈ ref_column_field rtol = 0.6
    end
end

@testset "2D SE, 1D FD Extruded Domain ∇ ODE Solve horizontal" begin

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

    hv_center_space, _ = hvspace_3D()
    U = sin.(Fields.coordinate_field(hv_center_space).x)
    dudt = zeros(eltype(U), hv_center_space)
    rhs!(dudt, U, nothing, 0.0)

    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(prob, SSPRK33(), dt = Δt)

    @test U ≈ sol.u[end] rtol = 1e-6
end

@testset "2D SE, 1D FV Extruded Domain ∇ ODE Solve diagonally" begin

    # Advection equation in Cartesian domain with
    # uₕ = (cₕ, 0), uᵥ = (0, cᵥ)
    # ∂ₜf + ∇ₕ⋅(uₕ * f) + ∇ᵥ⋅(uᵥ * f)  = 0
    # the solution translates diagonally to the top right and
    # at time t, the solution is f(x - cₕ * t, y - cᵥ * t)
    # here cₕ == cᵥ == 1, integrate t == 2π or one full period.
    # This is only correct if the solution is localized in the vertical
    # as we don't use periodic boundary conditions in the vertical.
    #
    # NOTE: the equation setup is only correct for Cartesian domains!

    hv_center_space, hv_face_space = hvspace_3D(
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0);
        device = ClimaComms.CPUSingleThreaded(),
    )

    abstract type BCtag end
    struct ZeroFlux <: BCtag end

    bc_divF2C_bottom!(::ZeroFlux, dY, Y, p, t) =
        Operators.SetValue(Geometry.UVWVector(0.0, 0.0, 0.0))

    struct ZeroFieldFlux <: BCtag end
    function bc_divF2C_bottom!(::ZeroFieldFlux, dY, Y, p, t)
        space = Spaces.horizontal_space(axes(Y.h))
        FT = Spaces.undertype(space)
        zeroflux = Fields.zeros(FT, space)
        return Operators.SetValue(
            Geometry.UVWVector.(zeroflux, zeroflux, zeroflux),
        )
    end

    function rhs!(dudt, u, p, t)
        h = u.h
        dh = dudt.h

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c = Operators.DivergenceF2C(
            bottom = bc_divF2C_bottom!(p.bc, dudt, u, p, t),
        )
        # only upward advection
        @. dh = -divf2c(Ic2f(h) * Geometry.UVWVector(0.0, 0.0, 1.0))

        # only horizontal advection
        hdiv = Operators.Divergence()
        @. dh += -hdiv(h * Geometry.UVVector(1.0, 1.0))
        Spaces.weighted_dss!(dh)

        return dudt
    end

    # initial conditions
    function h_init(x_init, y_init, z_init, A = 1.0, σ = 0.2)
        coords = Fields.coordinate_field(hv_center_space)
        h = map(coords) do coord
            A * exp(
                -(
                    (coord.x + x_init)^2 +
                    (coord.y + y_init)^2 +
                    (coord.z + z_init)^2
                ) / (2 * σ^2),
            )
        end
        return h
    end

    U = Fields.FieldVector(h = h_init(0.5, 0.5, 0.5))
    U_fieldbc = copy(U)
    Δt = 0.01
    t_end = 1.0
    p = (bc = ZeroFlux(),)
    p_fieldbc = (bc = ZeroFieldFlux(),)
    prob = ODEProblem(rhs!, U, (0.0, t_end), p)
    prob_fieldbc = ODEProblem(rhs!, U, (0.0, t_end), p_fieldbc)
    sol = solve(prob, SSPRK33(), dt = Δt)
    sol_fieldbc = solve(prob_fieldbc, SSPRK33(), dt = Δt)

    h_end = h_init(0.5 - t_end, 0.5 - t_end, 0.5 - t_end)
    @test h_end ≈ sol.u[end].h rtol = 0.5

    @test sol.u == sol_fieldbc.u
end
