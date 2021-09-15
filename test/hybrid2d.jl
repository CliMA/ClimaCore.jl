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
import ClimaCore.Domains.Geometry: Cartesian2DPoint, ⊗

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 64,
    npoly = 7,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        FT(zlim[1]),
        FT(zlim[2]);
        x3boundary = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

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

# V_face = Covariant3Vector
# V_center = Covariant12Vector
# V = V_center + V_face

# divergence(V) = horz_div(V) + vert_div(V)
# divergence(V) = horz_div(V_center) + horz_div(V_face) + vert_div(V_center) + vert_div(V_face)


# 1) horz_div(V_center): project to Contravariant1 + Contravariant2, take spectral derivative
# 2) horz_div(V_face): project to Contravariant1 + Contravariant2, interpolate to center, take spectral derivative
#   - will be zero if orthogional geom
# 3) vert_div(V_center): project to Contravariant3, interpolate to face, take FD deriv
#   - will be zero if orthogional geom
# 4) vert_div(V_face): project to Contravariant3, take FD deriv





@testset "1D SE, 1D FD Extruded Domain vertical advection operator" begin

    # Advection Operator
    # c ∂_z f
    # for this test, we use f(z) = sin(z) and c = 1
    # => c ∂_z f = cos(z)
    hv_center_space, hv_face_space = hvspace_2D()

    function rhs!(f)
        adv = zeros(eltype(f), hv_center_space)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(0.0),
            top = Operators.Extrapolate(),
        )
        return @. adv = A(c, f)
    end

    # advective velocity
    c = Geometry.Cartesian3Vector.(ones(Float64, hv_face_space),)
    # scalar-valued field to be advected
    f = sin.(Fields.coordinate_field(hv_center_space).z)
    # evaluate tendency
    adv = rhs!(f)

    @test norm(adv .- cos.(Fields.coordinate_field(hv_center_space).z)) ≤ 5e-2
end

@testset "1D SE, 1D FD Extruded Domain horizontal divergence operator" begin

    # Divergence Operator
    # ∂_x (c f)
    # for this test, we use f(x) = sin(x) and c = 1
    # => ∂_x (c f) = cos(x)

    function rhs!(f)
        divf = zeros(eltype(f), hv_center_space)
        # horizontal divergence operator applied to all levels
        hdiv = Operators.Divergence()
        @. divf = hdiv(f * Geometry.Cartesian1Vector(1.0))
        Spaces.weighted_dss!(divf)
        return divf
    end

    hv_center_space, _ = hvspace_2D()
    f = sin.(Fields.coordinate_field(hv_center_space).x)
    # evaluate tendency
    divf = rhs!(f)

    @test norm(divf .- cos.(Fields.coordinate_field(hv_center_space).x)) ≤ 5e-5
end

@testset "1D SE, 1D FD Extruded Domain ∇ ODE Solve diagonally" begin

    # Advection equation in Cartesian domain with
    # uₕ = (cₕ, 0), uᵥ = (0, cᵥ)
    # ∂ₜf + ∇ₕ⋅(uₕ * f) + ∇ᵥ⋅(uᵥ * f)  = 0
    # the solution translates diagonally to the top right and
    # at time t, the solution is f(x - cₕ * t, z - cᵥ * t)
    # here cₕ == cᵥ == 1, integrate t == 2π or one full period.
    # This is only correct if the solution is localized in the vertical
    # as we don't use periodic boundary conditions in the vertical.
    #
    # NOTE: the equation setup is only correct for Cartesian domains!

    hv_center_space, hv_face_space = hvspace_2D((-1, 1), (-1, 1))

    function rhs!(f)

        divuf = Fields.FieldVector(h = zeros(eltype(f), hv_center_space))
        h = f.h
        dh = divuf.h

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(
            top = Operators.Extrapolate(),
            bottom = Operators.Extrapolate(),
        )
        divf2c = Operators.DivergenceF2C()
        # only upward advection
        @. dh = divf2c(Ic2f(h) * Geometry.Cartesian3Vector(1.0))

        # only horizontal advection
        hdiv = Operators.Divergence()
        @. dh += hdiv(h * Geometry.Cartesian1Vector(1.0))
        Spaces.weighted_dss!(dh)

        return divuf
    end

    # A horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function h_blob(x_0, z_0, a = 1.0, σ = 0.2)
        coords = Fields.coordinate_field(hv_center_space)
        f = map(coords) do coord
            a * exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))
        end

        return f
    end

    # Spatial derivative of a horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function div_h_blob(x_0, z_0, a = 1.0, σ = 0.2)
        coords = Fields.coordinate_field(hv_center_space)
        div_uf = map(coords) do coord
            -a * (exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))) /
            (σ^2) * ((coord.x + x_0) + (coord.z + z_0))
        end

        return div_uf
    end

    f = Fields.FieldVector(h = h_blob(0.5, 0.5))

    divuf = rhs!(f)
    exact_divuf = div_h_blob(0.5, 0.5)

    @test norm(exact_divuf .- divuf.h) ≤ 5e-2
end


@testset "1D SE, 1D FD Extruded Domain vertical vector advection" begin

    # Vector Advection Operator
    # C ∂_z F
    # for this test, we use F(z) = sin(z) and C = ones
    # => C ∂_z F = cos(z)
    hv_center_space, hv_face_space = hvspace_2D()

    function advect!(f)
        advf = zeros(eltype(f), hv_center_space)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
            top = Operators.Extrapolate(),
        )
        @. advf = A(C, f)
    end

    # advective velocity
    C = Geometry.Cartesian3Vector.(ones(Float64, hv_face_space),)
    # vector-valued field to be advected
    f =
        Geometry.Cartesian1Vector.(
            sin.(Fields.coordinate_field(hv_center_space).z),
        )

    advf = advect!(f)

    function div!(F)
        vecdivf = zeros(eltype(F), hv_center_space)
        Ic2f = Operators.InterpolateC2F()
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(
                Geometry.Cartesian3Vector(1.0) ⊗ Geometry.Cartesian1Vector(0.0),
            ),
            top = Operators.Extrapolate(),
        )
        # only upward advection
        @. vecdivf = divf2c(C ⊗ Ic2f(F))

        hdiv = Operators.Divergence()
        @. vecdivf += hdiv(Geometry.Cartesian1Vector(1.0) ⊗ F)
        Spaces.weighted_dss!(vecdivf)
        return vecdivf
    end

    F =
        Geometry.Cartesian1Vector.(
            sin.(Fields.coordinate_field(hv_center_space).z),
        )

    vecdivf = div!(F)

    @test norm(
        vecdivf .-
        Geometry.Cartesian1Vector.(
            cos.(Fields.coordinate_field(hv_center_space).z),
        ),
    ) ≤ 5e-2

end

@testset "1D SE, 1D FD Extruded Domain scalar diffusion" begin

    # Scalar diffusion operator in 2D
    # ∂_xx u + ∂_zz u

    hv_center_space, hv_face_space = hvspace_2D((-5, 5), (-5, 5))

    K = 1.0
    function diff!(u, K)

        diffu = zeros(eltype(u), hv_center_space)
        gradc2f = Operators.GradientC2F(
            top = Operators.SetValue(0.0),
            bottom = Operators.SetValue(0.0),
        )
        divf2c = Operators.DivergenceF2C()
        @. diffu = divf2c(K * gradc2f(u))

        hgrad = Operators.Gradient()
        hdiv = Operators.Divergence()

        @. diffu += hdiv(K * hgrad(u))
        Spaces.weighted_dss!(diffu)
        return diffu
    end

    u = map(Fields.coordinate_field(hv_center_space)) do coord
        exp(-(coord.x^2 + coord.z^2) / 2)
    end

    # Laplacian of u = exp(-(coord.x^2 + coord.z^2) / 2)
    function laplacian_u()
        coords = Fields.coordinate_field(hv_center_space)
        laplacian_u = map(coords) do coord
            ((coord.x^2 - 1) + (coord.z^2 - 1)) *
            (exp(-((coord.x^2) / 2 + (coord.z^2) / 2)))
        end

        return laplacian_u
    end

    diffu = diff!(u, K)

    exact_laplacian = laplacian_u()

    @test norm(exact_laplacian .- diffu) ≤ 1e-2

end

@testset "1D SE, 1D FV Extruded Domain vector diffusion" begin

    hv_center_space, hv_face_space = hvspace_2D((-5, 5), (-5, 5))

    K = 1.0
    function vec_diff!(U, K)

        vec_diff = zeros(eltype(U), hv_center_space)
        gradc2f = Operators.GradientC2F(
            top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
            bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        )
        divf2c = Operators.DivergenceF2C()
        @. vec_diff = divf2c(K * gradc2f(U))

        hgrad = Operators.Gradient()
        hdiv = Operators.Divergence()

        @. vec_diff += hdiv(K * hgrad(U))
        Spaces.weighted_dss!(vec_diff)

        return vec_diff
    end

    U = map(Fields.coordinate_field(hv_center_space)) do coord
        Geometry.Cartesian1Vector(exp(-(coord.x^2 + coord.z^2) / 2))
    end

    # Laplacian of u = exp(-(coord.x^2 + coord.z^2) / 2)
    function hessian_u()
        coords = Fields.coordinate_field(hv_center_space)
        hessian_u = map(coords) do coord
            Geometry.Cartesian1Vector(
                ((coord.x^2 - 1) + (coord.z^2 - 1)) *
                (exp(-((coord.x^2) / 2 + (coord.z^2) / 2))),
            )
        end

        return hessian_u
    end

    vec_diff = vec_diff!(U, K)

    exact_hessian = hessian_u()

    @test norm(exact_hessian .- vec_diff) ≤ 1e-2

end
