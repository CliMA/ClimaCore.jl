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

#@testset "1D SE, 1D FV Extruded Domain ∇ ODE Solve" begin
FT = Float64
vertdomain = Domains.IntervalDomain(FT(0), FT(2π); x3boundary = (:bottom, :top))
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

#hvspace = Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

function slab_gradient!(∇data, data, space)
    # all derivatives calculated in the reference local geometry FT precision
    FT = ClimaCore.Spaces.undertype(space)
    D = ClimaCore.Spaces.Quadratures.differentiation_matrix(
        FT,
        Spaces.quadrature_style(space),
    )
    Nq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(space),
    )
    # for each element in the element stack
    ∂f∂ξ₁ = zeros(StaticArrays.MVector{Nq, FT})
    Nv = Spaces.nlevels(space)
    for h in 1:Topologies.nlocalelems(space.horizontal_space)
        for v in 1:Nv
            data_slab = ClimaCore.slab(data, v, h)
            ∇data_slab = ClimaCore.slab(∇data, v, h)
            ∂f∂ξ₁ .= zero(FT)
            local_geometry_slab = slab(Spaces.local_geometry_data(space), v, h)
            @inbounds for i in 1:Nq
                # compute covariant derivatives
                ∂ξ∂x = local_geometry_slab[i].∂ξ∂x
                ∂f∂ξ₁ .+= ∂ξ∂x[1, 1] * D[:, i] * data_slab[i]
            end
            # convert to desired basis
            @inbounds for i in 1:Nq
                ∇data_slab[i] = ∂f∂ξ₁[i]
            end
        end
    end
    return ∇data
end

# Advection Equation
# ∂_t f + c ∂_x f  = 0
# the solution translates to the right at speed c,
# so if you you have a periodic domain of size [-π, π]
# at time t, the solution is f(x - c * t, y)
# here c == 1, integrate t == 2π or one full period

function rhs!(dudt, u, _, t)
    grad = Operators.Gradient()
    div = Operators.Divergence()
    @. dudt = div(u * Geometry.Cartesian1Vector(1.0))
    #space = axes(u)
    #slab_gradient!(Fields.field_values(dudt), Fields.field_values(u), space)
    ClimaCore.Spaces.weighted_dss!(dudt)
end


U = sin.(Fields.coordinate_field(horzspace).x)
dudt = zeros(eltype(U), horzspace)
rhs!(dudt, U, nothing, 0.0)
#=
    using OrdinaryDiffEq
    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(
        prob,
        SSPRK33(),
        dt = Δt,
        #progress = true,
        #saveat = 100 * Δt,
    )

    @test norm(U .- sol.u[end]) ≤ 5e-5
end
=#
