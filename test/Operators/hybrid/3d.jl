using Test
using ClimaComms
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
import ClimaCore.Geometry: WVector

import ClimaCore.Utilities: half
import ClimaCore.DataLayouts: level

@testset "sphere divergence" begin
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(30.0)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 4)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice()),
        horzmesh,
    )
    quad = Spaces.Quadratures.GLL{3 + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    coords = Fields.coordinate_field(hv_center_space)
    x = Geometry.UVWVector.(cosd.(coords.lat), 0.0, 0.0)
    div = Operators.Divergence()
    @test norm(div.(x)) < 2e-2
end

function hvspace_3D(
    xlim = (-π, π),
    ylim = (-π, π),
    zlim = (0, 4π),
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 7,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(ylim[1]) .. Geometry.YPoint{FT}(ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice()),
        horzmesh,
    )

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

@testset "2D SE, 1D FD Extruded Domain level extraction" begin
    hv_center_space, hv_face_space = hvspace_3D()

    coord = Fields.coordinate_field(hv_face_space)

    @test parent(Fields.field_values(level(coord.x, half))) == parent(
        Fields.field_values(
            Fields.coordinate_field(hv_face_space.horizontal_space).x,
        ),
    )
    @test parent(Fields.field_values(level(coord.z, half))) ==
          parent(
        Fields.field_values(
            Fields.coordinate_field(hv_face_space.horizontal_space).x,
        ),
    ) .* 0
end

@testset "2D SE, 1D FV Extruded Domain ∇ ODE Solve vertical" begin

    hv_center_space, hv_face_space = hvspace_3D()
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

    using OrdinaryDiffEq
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

@testset "2D SE, 1D FV Extruded Domain ∇ ODE Solve horizontal" begin

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

    using OrdinaryDiffEq
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

    hv_center_space, hv_face_space =
        hvspace_3D((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    abstract type BCtag end
    struct ZeroFlux <: BCtag end

    bc_divF2C_bottom!(::ZeroFlux, dY, Y, p, t) =
        Operators.SetValue(Geometry.UVWVector(0.0, 0.0, 0.0))

    struct ZeroFieldFlux <: BCtag end
    function bc_divF2C_bottom!(::ZeroFieldFlux, dY, Y, p, t)
        space = axes(Y.h).horizontal_space
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

    using OrdinaryDiffEq
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

@testset "Spatially varying boundary conditions" begin
    FT = Float64
    n_elems_seq = (5, 6, 7, 8)

    err = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        cs, fs = hvspace_3D((-pi, pi), (-pi, pi), (-pi, pi), 4, 4, 2^n)
        coords = Fields.coordinate_field(cs)

        c = sin.(coords.x .+ coords.z)

        bottom_face = level(fs, half)
        top_face = level(fs, 2^n + half)
        bottom_coords = Fields.coordinate_field(bottom_face)
        top_coords = Fields.coordinate_field(top_face)
        flux_bottom =
            @. Geometry.WVector(sin(bottom_coords.x + bottom_coords.z))
        flux_top = @. Geometry.WVector(sin(top_coords.x + top_coords.z))
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(flux_bottom),
            top = Operators.SetValue(flux_top),
        )
        Ic2f = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )

        div = divf2c.(Ic2f.(c) .* Geometry.WVector.(Fields.ones(fs)))

        err[k] = norm(div .- cos.(coords.x .+ coords.z))
    end
    @show err
    @test err[4] ≤ err[3] ≤ err[2] ≤ err[1]
end


@testset "bycolumn fuse" begin
    hv_center_space, hv_face_space =
        hvspace_3D((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    fz = Fields.coordinate_field(hv_face_space).z
    ∇ = Operators.GradientF2C()
    ∇z = map(coord -> WVector(0.0), Fields.coordinate_field(hv_center_space))
    Fields.bycolumn(hv_center_space) do colidx
        @. ∇z[colidx] = WVector(∇(fz[colidx]))
    end
    @test ∇z == WVector.(∇.(fz))
end
