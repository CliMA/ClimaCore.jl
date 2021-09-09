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

function hvspace_2D(;
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

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "1D SE, 1D FV Extruded Domain ∇ ODE Solve vertical \\w convergence test" begin
    metric(x, y) = norm(x .- y)
    
    n_elems_seq = 2 .^ [4, 5, 6, 7, 8]
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    # convergence test loop
    for (k, nv) in enumerate(n_elems_seq)
        hv_center_space, hv_face_space = hvspace_2D(velem = nv)
        V =
            Geometry.Cartesian13Vector.(
                zeros(Float64, hv_face_space),
                ones(Float64, hv_face_space),
            )

        # rhs! for test
        function rhs!(dudt, u, _, t)
            A = Operators.AdvectionC2C(
                bottom = Operators.SetValue(sin(-t)),
                top = Operators.Extrapolate(),
            )
            return @. dudt = -A(V, u)
        end

        # run setup
        using OrdinaryDiffEq
        U = sin.(Fields.coordinate_field(hv_center_space).z)
        Δt = 0.01
        prob = ODEProblem(rhs!, U, (0.0, 2π))
        sol = solve(prob, SSPRK33(), dt = Δt)

        # calculate error metrics
        Δh[k] = ClimaCore.column(hv_center_space.face_local_geometry.J, 1, 1, 1)[1]
        err[k] = metric(U, sol.u[end])    
    end
    
    conv = convergence_rate(err, Δh)
    # conv should be approximately 2 for second order-accurate stencil.
    @test conv[1] ≈ 2 atol = 0.1
    @test conv[2] ≈ 2 atol = 0.1
    @test conv[3] ≈ 2 atol = 0.1
    @test conv[4] ≈ 2 atol = 0.1
    @test conv[5] ≈ 2 atol = 0.1

    # post-processing
    summary = Plots.plot(
        log10.(Δh), 
        log10.(err), 
        xlabel = "log10 of vertical grid spacing",
        ylabel = "log10 of absolute error norm",
        marker = true,
        label = "data",
        title = "Error convergence for vertical FD grid refinement"
    )
    plot!(summary, log10.(Δh), log10.(err[3].*(Δh./Δh[3]).^2), label = "expected-2nd-order")
    Plots.png(summary, "err_vadvect.png")        
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
        @. dudt = -hdiv(u * Geometry.Cartesian1Vector(1.0))
        Spaces.weighted_dss!(dudt)
        return dudt
    end

    hv_center_space, _ = hvspace_2D()
    U = sin.(Fields.coordinate_field(hv_center_space).x)
    dudt = zeros(eltype(U), hv_center_space)
    rhs!(dudt, U, nothing, 0.0)

    using OrdinaryDiffEq
    Δt = 0.01
    prob = ODEProblem(rhs!, U, (0.0, 2π))
    sol = solve(prob, SSPRK33(), dt = Δt)

    @test norm(U .- sol.u[end]) ≤ 5e-5
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
    prob = ODEProblem(rhs!, U, (0.0, t_end))
    sol = solve(prob, SSPRK33(), dt = Δt)

    h_end = h_init(0.5 - t_end, 0.5 - t_end)
    @test norm(h_end .- sol.u[end].h) ≤ 5e-2
end
