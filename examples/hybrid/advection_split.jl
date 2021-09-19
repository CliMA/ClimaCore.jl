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

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem::I = 10,
    velem::I = 64,
    npoly::I = 7,
) where {I <: Int}
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        FT(zlim[1]),
        FT(zlim[2]);
        x3boundary = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vert_center_space)

    horzdomain = Domains.RectangleDomain(
        xlim[1]..xlim[2],
        -0..0,
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, helem, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end


@testset "1D SE, 1D FV Extruded Domain ∇ ODE Solve diagonally" begin

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

    hv_center_space, hv_face_space = hvspace_2D((-1, 1), (-1, 1))

    function rhs!(dudt, u, _, t)
        h = u.h
        dh = dudt.h

        # vertical advection no inflow at bottom 
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0, 0.0)),
        )
        # only upward advection
        @. dh = -divf2c(Ic2f(h) * Geometry.Cartesian13Vector(0.0, 1.0))

        # only horizontal advection
        hdiv = Operators.Divergence()
        @. dh += -hdiv(h * Geometry.Cartesian1Vector(1.0))
        Spaces.weighted_dss!(dh)

        return dudt
    end

    function rhs_vertical!(dudt, u, _, t)
        h = u.h
        dh = dudt.h

        # vertical advection no inflow at bottom 
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0, 0.0)),
        )
        # only upward advection
        @. dh = -divf2c(Ic2f(h) * Geometry.Cartesian13Vector(0.0, 1.0))

        @info "vertical norm dh = ", norm(dh), "norm h = ", norm(h)
        return dudt
    end

    function rhs_horizontal!(dudt, u, _, t)
        h = u.h
        dh = dudt.h

        # only horizontal advection
        hdiv = Operators.Divergence()
        @. dh = -hdiv(h * Geometry.Cartesian1Vector(1.0))
        Spaces.weighted_dss!(dh)

        @info "horizontal norm dh = ", norm(dh), "norm h = ", norm(h)
        return dudt
    end

    # initial conditions
    h_init(x_init, z_init) = begin
        coords = Fields.coordinate_field(hv_center_space)
        h = map(coords) do coord
            exp(-((coord.x + x_init)^2 + (coord.z + z_init)^2) / (2 * 0.2^2))
        end

        return h
    end

    using OrdinaryDiffEq
    U = Fields.FieldVector(h = h_init(0.5, 0.5))
    Δt = 0.01
    t_end = 1.0

    # prob = ODEProblem(rhs!, U, (0.0, t_end))
    # sol = solve(prob, SSPRK33(), dt = Δt)

    prob = SplitODEProblem(rhs_vertical!, rhs_horizontal!, U, (0.0,t_end))
    sol = solve(prob, KenCarp3(autodiff=false,diff_type=Val{:central}), dt=Δt)
    # sol = solve(prob, SSPRK33(), dt = Δt)

    h_end = h_init(0.5 - t_end, 0.5 - t_end)
    @test norm(h_end .- sol.u[end].h) ≤ 5e-2
end
