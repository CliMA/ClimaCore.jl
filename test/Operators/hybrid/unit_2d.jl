#=
julia --project
using Revise; include(joinpath("test", "Operators", "hybrid", "unit_2d.jl"))
=#
include("utils_2d.jl")

@testset "1D SE, 1D FD Extruded Domain level extraction" begin
    hv_center_space, hv_face_space =
        hvspace_2D(npoly = 4, velem = 10, helem = 10)

    fcoord = Fields.coordinate_field(hv_face_space)
    ccoord = Fields.coordinate_field(hv_center_space)
    @test all(parent(Fields.field_values(level(fcoord.x, half))) .== parent(
        Fields.field_values(
            Fields.coordinate_field(Spaces.horizontal_space(hv_face_space)).x,
        ),
    ))
    @test all(parent(Fields.field_values(level(ccoord.x, 1))) .== parent(
        Fields.field_values(
            Fields.coordinate_field(Spaces.horizontal_space(hv_center_space)).x,
        ),
    ))
    @test all(parent(Fields.field_values(level(fcoord.z, half))) .==
          parent(
        Fields.field_values(
            Fields.coordinate_field(Spaces.horizontal_space(hv_face_space)).x,
        ),
    ) .* 0)
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
