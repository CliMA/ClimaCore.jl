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
import ClimaCore.Domains.Geometry: Geometry.Cartesian12Point, ⊗

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 64,
    npoly = 7,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(-0.0)..Geometry.YPoint{FT}(0.0),
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
    # for this test, we use f(z) = sin(z) and c = 1, a Cartesian3Vector
    # => c ∂_z f = cos(z)
    hv_center_space, hv_face_space = hvspace_2D()

    function advection(c, f)
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

    # Call the advection operator
    adv = advection(c, f)

    @test norm(adv .- cos.(Fields.coordinate_field(hv_center_space).z)) ≤ 5e-2
end

@testset "1D SE, 1D FD Extruded Domain horizontal divergence operator" begin

    # Divergence Operator
    # ∂_x (c f)
    # for this test, we use f(x) = sin(x) and c = 2, a Cartesian1Vector
    # => ∂_x (c f) = 2 * cos(x)

    function divergence(c, f)
        divf = zeros(eltype(f), hv_center_space)
        # horizontal divergence operator applied to all levels
        hdiv = Operators.Divergence()
        divf .= hdiv.(f .* Ref(c)) # Ref is needed to treat c as a scalar for the broadcasting operator
        Spaces.weighted_dss!(divf)
        return divf
    end

    hv_center_space, _ = hvspace_2D()
    f = sin.(Fields.coordinate_field(hv_center_space).x)

    # Set up constant velocity field
    c = Geometry.Cartesian1Vector(2.0)

    # Call the divergence operator
    divf = divergence(c, f)

    @test norm(
        divf .- 2.0 .* cos.(Fields.coordinate_field(hv_center_space).x),
    ) ≤ 5e-5
end

@testset "1D SE, 1D FD Extruded Domain horz & vert divergence operator" begin

    # Divergence operator in 2D Cartesian domain with
    # Cₕ = (cₕ, 0), Cᵥ = (0, cᵥ)
    # ∇ₕ⋅(Cₕ * f) + ∇ᵥ⋅(Cᵥ * f)
    # here cₕ == cᵥ == 1

    # NOTE: the equation setup is only correct for Cartesian domains!

    hv_center_space, hv_face_space = hvspace_2D((-1, 1), (-1, 1))

    function divergence(f)

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
        # only upward component of divergence
        @. dh = divf2c(Ic2f(h) * Geometry.Cartesian3Vector(1.0))

        # only horizontal component of divergence
        hdiv = Operators.Divergence()
        @. dh += hdiv(h * Geometry.Cartesian1Vector(1.0)) # add the two components for full divergence +=
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

    divuf = divergence(f)
    exact_divuf = div_h_blob(0.5, 0.5)

    @test norm(exact_divuf .- divuf.h) ≤ 5e-2
end


@testset "1D SE, 1D FD Extruded Domain vertical vector advection" begin

    # Vector Advection Operator
    # C ∂_z F
    # for this test, we use F(z) = sin(z),
    # with F a Cartesian1Vector and C = ones, a Cartesian3Vector constant field
    # => C ∂_z F = cos(z)
    hv_center_space, hv_face_space = hvspace_2D()

    function advect(f)
        advf = zeros(eltype(f), hv_center_space)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)), # value of f at the boundary (Cartesian1Vector field)
            top = Operators.Extrapolate(),
        )
        @. advf = A(vC, f)
    end

    # Vertical advective velocity
    vC = Geometry.Cartesian3Vector.(ones(Float64, hv_face_space),)
    # vector-valued field to be advected (one component only, a Cartesian1Vector)
    f =
        Geometry.Cartesian1Vector.(
            sin.(Fields.coordinate_field(hv_center_space).z),
        )

    advf = advect(f)

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
        @. vecdivf = divf2c(vC ⊗ Ic2f(F))

        hdiv = Operators.Divergence()
        @. vecdivf += hdiv(Geometry.Cartesian1Vector(1.0) ⊗ F)
        Spaces.weighted_dss!(vecdivf)
        return vecdivf
    end

    F =
        Geometry.Cartesian1Vector.(
            sin.(Fields.coordinate_field(hv_center_space).z),
        )

    # Call the divergence operator
    vecdivf = div!(F)

    @test norm(
        vecdivf .-
        Geometry.Cartesian1Vector.(
            cos.(Fields.coordinate_field(hv_center_space).z),
        ),
    ) ≤ 6.5e-2

end

@testset "1D SE, 1D FD Extruded Domain scalar diffusion" begin

    # Scalar diffusion operator in 2D
    # ∂_xx u + ∂_zz u

    hv_center_space, hv_face_space = hvspace_2D((-5, 5), (-5, 5))

    K = 1.0

    function diff(K, u)

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

    # scalar-valued field to be diffused, u = exp(-(coord.x^2 + coord.z^2) / 2)
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

    # Call the diffusion operator
    diffu = diff(K, u)

    exact_laplacian = laplacian_u()

    @test norm(exact_laplacian .- diffu) ≤ 1e-2

end

@testset "1D SE, 1D FD Extruded Domain vector diffusion" begin

    # Vector diffusion operator in 2D, of vector-valued field U, with U == u
    # (one component only)
    # ∂_xx u + ∂_zz u

    hv_center_space, hv_face_space = hvspace_2D((-5, 5), (-5, 5))

    K = 1.0

    function vec_diff(K, U)

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

    # vector-valued field to be diffused (one component only) u = exp(-(coord.x^2 + coord.z^2) / 2)
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

    # Call the vector diffusion operator
    vec_diff = vec_diff(K, U)

    exact_hessian = hessian_u()

    @test norm(exact_hessian .- vec_diff) ≤ 1e-2

end
