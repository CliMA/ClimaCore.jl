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
    Operators,
    level
import ClimaCore.Geometry: WVector
import ClimaCore.Domains.Geometry: ⊗
import ClimaCore.Utilities: half

convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

function hvspace_2D(;
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

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

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


@testset "1D SE, 1D FD Extruded Domain level extraction" begin
    hv_center_space, hv_face_space =
        hvspace_2D(npoly = 4, velem = 10, helem = 10)

    fcoord = Fields.coordinate_field(hv_face_space)
    ccoord = Fields.coordinate_field(hv_center_space)
    @test parent(Fields.field_values(level(fcoord.x, half))) == parent(
        Fields.field_values(
            Fields.coordinate_field(hv_face_space.horizontal_space).x,
        ),
    )
    @test parent(Fields.field_values(level(ccoord.x, 1))) == parent(
        Fields.field_values(
            Fields.coordinate_field(hv_center_space.horizontal_space).x,
        ),
    )
    @test parent(Fields.field_values(level(fcoord.z, half))) ==
          parent(
        Fields.field_values(
            Fields.coordinate_field(hv_face_space.horizontal_space).x,
        ),
    ) .* 0
end


@testset "1D SE, 1D FD Extruded Domain matrix interpolation" begin
    hv_center_space, hv_face_space =
        hvspace_2D(npoly = 4, velem = 10, helem = 10)

    center_field = sin.(Fields.coordinate_field(hv_center_space).z)
    face_field = sin.(Fields.coordinate_field(hv_face_space).z)

    for npoints in (3, 8)
        M_center = Operators.matrix_interpolate(center_field, npoints)
        M_face = Operators.matrix_interpolate(face_field, npoints)
        @test size(M_center) == (10, 10 * npoints)
        @test size(M_face) == (10 + 1, 10 * npoints)
    end
end


@testset "1D SE, 1D FD Extruded Domain vertical advection operator" begin

    function advection(c, f, hv_center_space)
        adv = zeros(eltype(f), hv_center_space)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(0.0),
            top = Operators.Extrapolate(),
        )
        return @. adv = A(c, f)
    end

    n_elems_seq = 2 .^ (5, 6, 7, 8)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        # Advection Operator
        # c ∂_z f
        # for this test, we use f(z) = sin(z) and c = 1, a WVector
        # => c ∂_z f = cos(z)
        hv_center_space, hv_face_space = hvspace_2D(helem = n, velem = n)

        Δh[k] = 1.0 / n

        # advective velocity
        c = Geometry.WVector.(ones(Float64, hv_face_space),)
        # scalar-valued field to be advected
        f = sin.(Fields.coordinate_field(hv_center_space).z)

        # Call the advection operator
        adv = advection(c, f, hv_center_space)

        err[k] = norm(adv .- cos.(Fields.coordinate_field(hv_center_space).z))
    end
    # AdvectionC2C convergence rate
    conv_adv_c2c = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_adv_c2c[1] ≈ 2 atol = 0.1
    @test conv_adv_c2c[2] ≈ 2 atol = 0.1
    @test conv_adv_c2c[3] ≈ 2 atol = 0.1
end

@testset "1D SE, 1D FD Extruded Domain Discrete Product Rule Operations: Kinetic Energy" begin

    gradc2f = Operators.GradientC2F(
        top = Operators.SetValue(0.0),
        bottom = Operators.SetValue(0.0),
    )
    gradf2c = Operators.GradientF2C()

    n_elems_seq = 2 .^ (5, 6, 7, 8)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        # Discrete Prodouct Rule Test
        # ∂(ab)/∂s = a̅∂b/∂s + b̅∂a∂s
        # a, b are interface variables, and  ̅ represents interpolation
        # s is the coordinate along horizontal surfaces (terrain following)
        # For this test, we use a(z) = z and b = sin(z),
        hv_center_space, hv_face_space = hvspace_2D(helem = n, velem = n)
        ᶠz = Fields.coordinate_field(hv_face_space).z
        ᶜz = Fields.coordinate_field(hv_center_space).z
        Δh[k] = 1.0 / n

        # advective velocity
        a = Geometry.WVector.(ones(Float64, hv_face_space) .* ᶠz,)
        # scalar-valued field to be advected
        b = sin.(ᶠz)
        ∂ab_numerical = @. gradf2c(a * b)
        ∂ab_analytical = @. ᶜz * cos(ᶜz) + sin(ᶜz)

        err[k] = ∂ab_numerical .- ∂ab_analytical
    end
    # AdvectionC2C convergence rate
    conv_adv_c2c = convergence_rate(err, Δh)
    @show conv_adv_c2c
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_adv_c2c[1] ≈ 2 atol = 0.1
    @test conv_adv_c2c[2] ≈ 2 atol = 0.1
    @test conv_adv_c2c[3] ≈ 2 atol = 0.1
end

@testset "1D SE, 1D FD Extruded Domain horizontal divergence operator" begin

    # Divergence Operator
    # ∂_x (c f)
    # for this test, we use f(x) = sin(x) and c = 2, a UVector
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
    c = Geometry.UVector(2.0)

    # Call the divergence operator
    divf = divergence(c, f)

    @test norm(
        divf .- 2.0 .* cos.(Fields.coordinate_field(hv_center_space).x),
    ) ≤ 5e-5
end

@testset "1D SE, 1D FD Extruded Domain horz & vert divergence operator, with Extrapolate BCs" begin

    # Divergence operator in 2D Cartesian domain with
    # Cₕ = (cₕ, 0), Cᵥ = (0, cᵥ)
    # ∇ₕ⋅(Cₕ * f) + ∇ᵥ⋅(Cᵥ * f)
    # here cₕ == cᵥ == 1

    # NOTE: the equation setup is only correct for Cartesian domains!

    function divergence(h, bottom_flux)

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c =
            Operators.DivergenceF2C(bottom = Operators.SetValue(bottom_flux))
        # only upward component of divergence
        dh = @. divf2c(Ic2f(h))

        # only horizontal component of divergence
        hdiv = Operators.Divergence()
        @. dh += hdiv(h) # add the two components for full divergence +=
        Spaces.weighted_dss!(dh)

        return dh
    end


    # A horizontal Gaussian blob, centered at (x_0, z_0),
    # with `a` the height of the blob and σ the standard deviation
    function h_blob(x, z, x_0, z_0, a = 1.0, σ = 0.2)
        a * exp(-((x - x_0)^2 + (z - z_0)^2) / (2 * σ^2))
    end
    # Spatial gradient of a horizontal Gaussian blob, centered at (x_0, z_0),
    # with `a` the height of the blob and σ the standard deviation
    ∇h_blob(x, z, x_0, z_0, a = 1.0, σ = 0.2) =
        -a * (exp(-((x - x_0)^2 + (z - z_0)^2) / (2 * σ^2))) / (σ^2) *
        Geometry.UWVector(x - x_0, z - z_0)


    n_elems_seq = 2 .^ (6, 7, 8, 9)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        hv_center_space, hv_face_space =
            hvspace_2D(xlim = (-1, 1), zlim = (-1, 1), helem = n, velem = n)
        ccoords = Fields.coordinate_field(hv_center_space)
        bcoords = Fields.coordinate_field(hv_center_space.horizontal_space)

        Δh[k] = 1.0 / n

        h =
            h_blob.(ccoords.x, ccoords.z, -0.5, -0.5) .*
            Ref(Geometry.UWVector(1.0, 1.0))
        bottom_flux =
            h_blob.(bcoords.x, -1.0, -0.5, -0.5) .*
            Ref(Geometry.UWVector(1.0, 1.0))
        exact_divh =
            Ref(Geometry.UWVector(1.0, 1.0)') .*
            ∇h_blob.(ccoords.x, ccoords.z, -0.5, -0.5)

        divh = divergence(h, bottom_flux)

        err[k] = norm(exact_divh .- divh)
    end
    # Divergence convergence rate
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.01
    conv_comp_div = convergence_rate(err, Δh)
    #= not sure these are right?
    @test conv_comp_div[1] ≈ 0.5 atol = 0.1
    @test conv_comp_div[2] ≈ 0.5 atol = 0.1
    @test conv_comp_div[3] ≈ 0.5 atol = 0.1
    =#
end

@testset "1D SE, 1D FD Extruded Domain horz & vert divergence operator, with Dirichelet BCs" begin

    # Divergence operator in 2D Cartesian domain with
    # Cₕ = (cₕ, 0), Cᵥ = (0, cᵥ)
    # ∇ₕ⋅(Cₕ * f) + ∇ᵥ⋅(Cᵥ * f)
    # here cₕ == cᵥ == 1

    # NOTE: the equation setup is only correct for Cartesian domains!

    function divergence(f, hv_center_space)

        divuf = Fields.FieldVector(h = zeros(eltype(f), hv_center_space))
        h = f.h
        dh = divuf.h

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(
            top = Operators.SetValue(0.0),
            bottom = Operators.SetValue(0.0),
        )
        divf2c = Operators.DivergenceF2C()
        # only upward component of divergence
        @. dh = divf2c(Ic2f(h) * Geometry.WVector(1.0))

        # only horizontal component of divergence
        hdiv = Operators.Divergence()
        @. dh += hdiv(h * Geometry.UVector(1.0)) # add the two components for full divergence +=
        Spaces.weighted_dss!(dh)

        return divuf
    end

    # A horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function h_blob(coords, x_0, z_0, a = 1.0, σ = 0.2)
        f = map(coords) do coord
            a * exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))
        end

        return f
    end

    # Spatial derivative of a horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function div_h_blob(coords, x_0, z_0, a = 1.0, σ = 0.2)
        div_uf = map(coords) do coord
            -a * (exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))) /
            (σ^2) * ((coord.x + x_0) + (coord.z + z_0))
        end

        return div_uf
    end

    n_elems_seq = 2 .^ (6, 7, 8, 9)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        hv_center_space, hv_face_space =
            hvspace_2D(xlim = (-1, 1), zlim = (-1, 1), helem = n, velem = n)

        Δh[k] = 1.0 / n

        coords = Fields.coordinate_field(hv_center_space)
        f = Fields.FieldVector(h = h_blob(coords, 0.0, 0.0))

        divuf = divergence(f, hv_center_space)
        exact_divuf = div_h_blob(coords, 0.0, 0.0)

        err[k] = norm(exact_divuf .- divuf.h)
    end
    # Divergence convergence rate
    conv_comp_div = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_comp_div[1] ≈ 2 atol = 0.1
    @test conv_comp_div[2] ≈ 2 atol = 0.1
    @test conv_comp_div[3] ≈ 2 atol = 0.1
end

@testset "1D SE, 1D FD Extruded Domain vertical vector advection" begin

    # Vector Advection Operator
    # C ∂_z F
    # for this test, we use F(z) = sin(z),
    # with F a UVector and C = ones, a WVector constant field
    # => C ∂_z F = cos(z)
    hv_center_space, hv_face_space = hvspace_2D()

    function advect(f)
        advf = zeros(eltype(f), hv_center_space)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.UVector(0.0)), # value of f at the boundary (UVector field)
            top = Operators.Extrapolate(),
        )
        @. advf = A(vC, f)
    end

    # Vertical advective velocity
    vC = Geometry.WVector.(ones(Float64, hv_face_space),)
    # vector-valued field to be advected (one component only, a UVector)
    f = Geometry.UVector.(sin.(Fields.coordinate_field(hv_center_space).z),)

    advf = advect(f)

    function div!(F)
        vecdivf = zeros(eltype(F), hv_center_space)
        Ic2f = Operators.InterpolateC2F()
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(
                Geometry.WVector(1.0) ⊗ Geometry.UVector(0.0),
            ),
            top = Operators.Extrapolate(),
        )
        # only upward advection
        @. vecdivf = divf2c(vC ⊗ Ic2f(F))

        hdiv = Operators.Divergence()
        @. vecdivf += hdiv(Geometry.UVector(1.0) ⊗ F)
        Spaces.weighted_dss!(vecdivf)
        return vecdivf
    end

    F = Geometry.UVector.(sin.(Fields.coordinate_field(hv_center_space).z),)

    # Call the divergence operator
    vecdivf = div!(F)

    @test norm(
        vecdivf .-
        Geometry.UVector.(cos.(Fields.coordinate_field(hv_center_space).z),),
    ) ≤ 6.5e-2

end

@testset "1D SE, 1D FD Extruded Domain scalar diffusion" begin

    # Scalar diffusion operator in 2D
    # ∂_xx u + ∂_zz u

    hv_center_space, hv_face_space = hvspace_2D(xlim = (-5, 5), zlim = (-5, 5))

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

    hv_center_space, hv_face_space = hvspace_2D(xlim = (-5, 5), zlim = (-5, 5))

    K = 1.0

    function vec_diff(K, U)

        vec_diff = zeros(eltype(U), hv_center_space)
        gradc2f = Operators.GradientC2F(
            top = Operators.SetValue(Geometry.UVector(0.0)),
            bottom = Operators.SetValue(Geometry.UVector(0.0)),
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
        Geometry.UVector(exp(-(coord.x^2 + coord.z^2) / 2))
    end

    # Laplacian of u = exp(-(coord.x^2 + coord.z^2) / 2)
    function hessian_u()
        coords = Fields.coordinate_field(hv_center_space)
        hessian_u = map(coords) do coord
            Geometry.UVector(
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



@testset "curl" begin
    hv_center_space, hv_face_space =
        hvspace_2D(xlim = (-pi, pi), zlim = (-pi, pi))

    ccoords = Fields.coordinate_field(hv_center_space)
    fcoords = Fields.coordinate_field(hv_face_space)

    u =
        Geometry.transform.(
            Ref(Geometry.Covariant1Axis()),
            Geometry.UVector.(sin.(ccoords.x .+ 2 .* ccoords.z)),
        )
    w =
        Geometry.transform.(
            Ref(Geometry.Covariant3Axis()),
            Geometry.WVector.(cos.(3 .* fcoords.x .+ 4 .* fcoords.z)),
        )

    curl = Operators.Curl()
    curlC2F = Operators.CurlC2F(
        bottom = Operators.SetValue(Geometry.Covariant1Vector(0.0)),
        top = Operators.SetValue(Geometry.Covariant1Vector(0.0)),
    )

    curlu = curlC2F.(u)
    curlw = curl.(w)
    curluw = curlu .+ curlw
    Spaces.weighted_dss!(curluw)

    curlw_ref =
        Geometry.Contravariant2Vector.(
            3 .* sin.(3 .* fcoords.x .+ 4 .* fcoords.z),
        )
    curlu_ref =
        Geometry.Contravariant2Vector.(2 .* cos.(fcoords.x .+ 2 .* fcoords.z))

    curluw_ref = curlu_ref .+ curlw_ref

    # TODO: make SetValue do reasonable things on Extruded spaces (#79)
    zeroboundary = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetValue(Geometry.Contravariant2Vector(0.0)),
    )
    curluw_diff = zeroboundary.(curluw .- curluw_ref)

    @test norm(curluw_diff) < norm(curluw_ref) * 1e-2
end

@testset "curl (field b.c.)" begin
    hv_center_space, hv_face_space =
        hvspace_2D(xlim = (-pi, pi), zlim = (-pi, pi))

    ccoords = Fields.coordinate_field(hv_center_space)
    fcoords = Fields.coordinate_field(hv_face_space)
    fcoords_1 = Fields.level(fcoords, ClimaCore.Utilities.half)
    curl_bcfield₁ = Geometry.Covariant1Vector.(0.0 .* fcoords_1.z)
    curl_bcfield² = Geometry.Contravariant2Vector.(0.0 .* fcoords_1.z)

    u =
        Geometry.transform.(
            Ref(Geometry.Covariant1Axis()),
            Geometry.UVector.(sin.(ccoords.x .+ 2 .* ccoords.z)),
        )
    w =
        Geometry.transform.(
            Ref(Geometry.Covariant3Axis()),
            Geometry.WVector.(cos.(3 .* fcoords.x .+ 4 .* fcoords.z)),
        )

    curl = Operators.Curl()
    curlC2F = Operators.CurlC2F(
        bottom = Operators.SetValue(curl_bcfield₁),
        top = Operators.SetValue(Geometry.Covariant1Vector(0.0)),
    )

    curlu = curlC2F.(u)
    curlw = curl.(w)
    curluw = curlu .+ curlw
    Spaces.weighted_dss!(curluw)

    curlw_ref =
        Geometry.Contravariant2Vector.(
            3 .* sin.(3 .* fcoords.x .+ 4 .* fcoords.z),
        )
    curlu_ref =
        Geometry.Contravariant2Vector.(2 .* cos.(fcoords.x .+ 2 .* fcoords.z))

    curluw_ref = curlu_ref .+ curlw_ref

    # TODO: make SetValue do reasonable things on Extruded spaces (#79)
    zeroboundary = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(curl_bcfield²),
        top = Operators.SetValue(Geometry.Contravariant2Vector(0.0)),
    )
    curluw_diff = zeroboundary.(curluw .- curluw_ref)

    @test norm(curluw_diff) < norm(curluw_ref) * 1e-2
end


@testset "curl-cross" begin
    hv_center_space, hv_face_space =
        hvspace_2D(xlim = (-pi, pi), zlim = (-pi, pi))

    ccoords = Fields.coordinate_field(hv_center_space)
    fcoords = Fields.coordinate_field(hv_face_space)

    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0

    u =
        Geometry.transform.(
            Ref(Geometry.Covariant1Axis()),
            Geometry.UVector.(a .* ccoords.x .+ b .* ccoords.z),
        )
    w =
        Geometry.transform.(
            Ref(Geometry.Covariant3Axis()),
            Geometry.WVector.(c .* fcoords.x .+ d .* fcoords.z),
        )

    curl = Operators.Curl()
    curlC2F = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(b)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(b)),
    )

    curlu = curlC2F.(u)
    curlw = curl.(w)
    curluw = curlu .+ curlw
    @test norm(curluw .- Ref(Geometry.Contravariant2Vector(b - c))) < 1e-10

    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )


    fu¹² = Geometry.Contravariant1Vector.(Geometry.Covariant13Vector.(Ic2f.(u))) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant13Vector.(w))


    Geometry.WVector.(Geometry.Covariant13Vector.(curluw .× fu¹²))

    curl_cross =
        Geometry.UWVector.(
            Geometry.Covariant13Vector.(curluw .× fu¹²) .+
            Geometry.Covariant13Vector.(curluw .× fu³),
        )

    curl_cross_ref =
        Geometry.UWVector.(
            (b - c) .* (c .* fcoords.x .+ d .* fcoords.z),
            .-(b - c) .* (a .* fcoords.x .+ b .* fcoords.z),
        )

    @test curl_cross ≈ curl_cross_ref rtol = 1e-2


end

@testset "2D hybrid hyperdiffusion" begin
    hv_center_space, hv_face_space =
        hvspace_2D(xlim = (-pi, pi), zlim = (-pi, pi))

    coords = Fields.coordinate_field(hv_center_space)
    k = 2
    y = @. sin(k * coords.x)
    ∇⁴y_ref = @. k^4 * sin(k * coords.x)

    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    χ = Spaces.weighted_dss!(@. wdiv(grad(y)))
    ∇⁴y = Spaces.weighted_dss!(@. wdiv(grad(χ)))

    @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
end


@testset "bycolumn fuse" begin
    hv_center_space, hv_face_space =
        hvspace_2D(xlim = (-pi, pi), zlim = (-pi, pi))

    fz = Fields.coordinate_field(hv_face_space).z
    ∇ = Operators.GradientF2C()
    ∇z = map(coord -> WVector(0.0), Fields.coordinate_field(hv_center_space))
    Fields.bycolumn(hv_center_space) do colidx
        @. ∇z[colidx] = WVector(∇(fz[colidx]))
    end
    @test ∇z == WVector.(∇.(fz))

end
