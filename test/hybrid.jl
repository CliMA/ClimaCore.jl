using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
import ClimaCore.Domains.Geometry: Cartesian2DPoint

function hvspace_2D()
    FT = Float64
    vertdomain =
        Domains.IntervalDomain(FT(-π), FT(π); x3boundary = (:bottom, :top))
    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        Meshes.ExponentialStretching(π / 2),
        nelems = 20,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    horzdomain = Domains.RectangleDomain(
        -π..π,
        -0..0,
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, 20, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{4}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hvspace = Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    return hvspace
end

@testset "1D SE, 1D FV Extruded Domain ∇ ODE Solve horzizontal" begin


    # Advection Equation
    # ∂_t f + c ∂_x f  = 0
    # the solution translates to the right at speed c,
    # so if you you have a periodic domain of size [-π, π]
    # at time t, the solution is f(x - c * t, y)
    # here c == 1, integrate t == 2π or one full period

    function rhs!(dudt, u, _, t)
        # horizontal divergence operator applied to all levels
        hdiv = Operators.Divergence()
        @. dudt = hdiv(u * Geometry.Cartesian1Vector(1.0))
        ClimaCore.Spaces.weighted_dss!(dudt)
        return dudt
    end

    hvspace = hvspace_2D()
    U = sin.(Fields.coordinate_field(hvspace).x)
    dudt = zeros(eltype(U), hvspace)
    rhs!(dudt, U, nothing, 0.0)

    using OrdinaryDiffEq
    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(prob, SSPRK33(), dt = Δt)

    @test norm(U .- sol.u[end]) ≤ 5e-5
end
