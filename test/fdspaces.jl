using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

import ClimaCore.Operators: half, PlusHalf

@testset "PlusHalf" begin
    @test half + 0 == half
    @test half < half + 1
    @test half <= half + 1
    @test !(half > half + 1)
    @test !(half >= half + 1)
    @test half != half + 1
    @test half + half == 1
    @test half - half == 0
    @test half + 3 == 3 + half == PlusHalf(3)
    @test min(half, half + 3) == half
    @test max(half, half + 3) == half + 3

    @test collect(half:(2 + half)) == [half, 1 + half, 2 + half]

    @test_throws InexactError convert(Int, half)
    @test_throws InexactError convert(PlusHalf, 1)
    @test_throws InexactError convert(PlusHalf{Int}, 1)
end


@testset "Scalar Field FiniteDifferenceSpaces" begin
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_tags = (:left, :right),
        )
        @test eltype(domain) === Geometry.ZPoint{FT}

        mesh = Meshes.IntervalMesh(domain; nelems = 16)

        center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        @test sum(ones(FT, center_space)) ≈ pi
        @test sum(ones(FT, face_space)) ≈ pi

        centers = getproperty(Fields.coordinate_field(center_space), :z)
        @test sum(sin.(centers)) ≈ FT(2.0) atol = 1e-2

        faces = getproperty(Fields.coordinate_field(face_space), :z)
        @test sum(sin.(faces)) ≈ FT(2.0) atol = 1e-2

        ∇ᶜ = Operators.GradientF2C()
        ∂sin = Geometry.CartesianVector.(∇ᶜ.(sin.(faces)))
        @test ∂sin ≈ Geometry.Cartesian3Vector.(cos.(centers)) atol = 1e-2

        divᶜ = Operators.DivergenceF2C()
        ∂sin = divᶜ.(Geometry.Cartesian3Vector.(sin.(faces)))
        @test ∂sin ≈ cos.(centers) atol = 1e-2

        # Center -> Face operator
        # first order convergence at boundaries
        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(pi)),
        )
        ∂z = Geometry.CartesianVector.(∇ᶠ.(centers))
        @test ∂z ≈ Geometry.Cartesian3Vector.(ones(FT, face_space)) rtol =
            10 * eps(FT)

        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetValue(FT(1)),
            right = Operators.SetValue(FT(-1)),
        )
        ∂cos = Geometry.CartesianVector.(∇ᶠ.(cos.(centers)))
        @test ∂cos ≈ Geometry.Cartesian3Vector.(.-sin.(faces)) atol = 1e-1

        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetGradient(Geometry.Cartesian3Vector(FT(0))),
            right = Operators.SetGradient(Geometry.Cartesian3Vector(FT(0))),
        )
        ∂cos = Geometry.CartesianVector.(∇ᶠ.(cos.(centers)))
        @test ∂cos ≈ Geometry.Cartesian3Vector.(.-sin.(faces)) atol = 1e-2

        # test that broadcasting into incorrect field space throws an error
        empty_centers = zeros(FT, center_space)
        @test_throws Exception empty_centers .= ∇ᶠ.(cos.(centers))
    end
end

@testset "Test composed stencils" begin
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_tags = (:left, :right),
        )
        @test eltype(domain) === Geometry.ZPoint{FT}

        mesh = Meshes.IntervalMesh(domain; nelems = 16)

        center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        centers = getproperty(Fields.coordinate_field(center_space), :z)
        w = ones(FT, face_space)
        θ = sin.(centers)

        # 1) we set boundaries on the 2nd operator
        I = Operators.InterpolateC2F()
        ∂ = Operators.GradientF2C(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )

        ∂sin = Geometry.CartesianVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.Cartesian3Vector.(cos.(centers)) atol = 1e-2

        # can't define Neumann conditions on GradientF2C
        ∂ = Operators.GradientF2C(
            left = Operators.SetGradient(FT(1)),
            right = Operators.SetGradient(FT(-1)),
        )

        @test_throws Exception ∂.(w .* I.(θ))

        # 2) we set boundaries on the 1st operator
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )
        ∂ = Operators.GradientF2C()

        ∂sin = Geometry.CartesianVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.Cartesian3Vector.(cos.(centers)) atol = 1e-2

        I = Operators.InterpolateC2F(
            left = Operators.SetGradient(Geometry.Cartesian3Vector(FT(1))),
            right = Operators.SetGradient(Geometry.Cartesian3Vector(FT(-1))),
        )
        ∂ = Operators.GradientF2C()

        ∂sin = Geometry.CartesianVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.Cartesian3Vector.(cos.(centers)) atol = 1e-2

        # 3) we set boundaries on both: 2nd should take precedence
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(NaN)),
            right = Operators.SetValue(FT(NaN)),
        )
        ∂ = Operators.GradientF2C(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )

        ∂sin = Geometry.CartesianVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.Cartesian3Vector.(cos.(centers)) atol = 1e-2

        # test that broadcasting into incorrect field space throws an error
        empty_faces = zeros(FT, face_space)
        @test_throws Exception empty_faces .= ∂.(w .* I.(θ))

        # 5) we set boundaries on neither
        I = Operators.InterpolateC2F()
        ∂ = Operators.GradientF2C()

        # TODO: should we throw something else?
        @test_throws BoundsError ∂.(w .* I.(θ))
    end
end

@testset "Composite Field FiniteDifferenceSpaces" begin
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_tags = (:left, :right),
        )

        @test eltype(domain) === Geometry.ZPoint{FT}
        mesh = Meshes.IntervalMesh(domain; nelems = 16)

        center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        FieldType = NamedTuple{(:a, :b), Tuple{FT, FT}}

        center_sum = sum(ones(FieldType, center_space))
        @test center_sum isa FieldType
        @test center_sum.a ≈ FT(pi)
        @test center_sum.b ≈ FT(pi)

        face_sum = sum(ones(FieldType, face_space))
        @test face_sum isa FieldType
        @test face_sum.a ≈ FT(pi)
        @test face_sum.b ≈ FT(pi)
    end
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


@testset "Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))

        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a)..Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_tags = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)

            cent_field_exact = zeros(FT, cs)
            cent_field = zeros(FT, cs)
            face_field = zeros(FT, fs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            face_field .= sin.(3π .* faces.z)
            cent_field_exact .= sin.(3π .* centers.z)
            operator = Operators.InterpolateF2C()
            cent_field .= operator.(face_field)

            Δh[k] = cs.face_local_geometry.J[1]
            err[k] =
                norm(parent(cent_field) .- parent(cent_field_exact)) /
                length(parent(cent_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.5 ≤ conv[1] ≤ 3
        @test 1.5 ≤ conv[3] ≤ 3
        if i == 1
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 1e-2
    end
end


@testset "Center -> Face interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))

        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a)..Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_tags = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)

            face_field_exact = zeros(FT, fs)
            cent_field = zeros(FT, cs)
            face_field = zeros(FT, fs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            cent_field .= sin.(3π .* centers.z)
            face_field_exact .= sin.(3π .* faces.z)

            operator = Operators.InterpolateC2F(
                left = Operators.SetValue(0.0),
                right = Operators.SetValue(0.0),
            )
            face_field .= operator.(cent_field)

            Δh[k] = cs.face_local_geometry.J[1]
            err[k] =
                norm(parent(face_field) .- parent(face_field_exact)) /
                length(parent(face_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.5 ≤ conv[1] ≤ 3
        @test 1.5 ≤ conv[3] ≤ 3
        if i == 1
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 1e-2
    end
end

@testset "∂ Center -> Face interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a)..Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_tags = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)

            face_field_exact = Geometry.Covariant3Vector.(zeros(FT, fs))
            cent_field = zeros(FT, cs)
            face_field = Geometry.Covariant3Vector.(zeros(FT, fs))

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            cent_field .= sin.(3π .* centers.z)
            face_field_exact .=
                Geometry.CovariantVector.(
                    Geometry.Cartesian3Vector.(3π .* cos.(3π .* faces.z)),
                )

            operator = Operators.GradientC2F(
                left = Operators.SetGradient(Geometry.Cartesian3Vector(3π)),
                right = Operators.SetGradient(Geometry.Cartesian3Vector(-3π)),
            )

            face_field .= operator.(cent_field)

            Δh[k] = cs.face_local_geometry.J[1]
            err[k] = norm(face_field .- face_field_exact)
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
        @test conv[1] ≈ 2 atol = 0.1
        @test conv[2] ≈ 2 atol = 0.1
        @test conv[3] ≈ 2 atol = 0.1
        @test conv[1] ≤ conv[2] ≤ conv[3]
    end
end

@testset "∂ Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a)..Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_tags = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)

            cent_field_exact = Geometry.Covariant3Vector.(zeros(FT, cs))
            cent_field = Geometry.Covariant3Vector.(zeros(FT, cs))
            face_field = zeros(FT, fs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            face_field .= sin.(3π .* faces.z)
            cent_field_exact .=
                Geometry.CovariantVector.(
                    Geometry.Cartesian3Vector.(3π .* cos.(3π .* centers.z)),
                )

            operator = Operators.GradientF2C()

            cent_field .= operator.(face_field)

            Δh[k] = cs.face_local_geometry.J[1]
            err[k] = norm(cent_field .- cent_field_exact)
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
        @test conv[1] ≈ 2 atol = 0.1
        @test conv[2] ≈ 2 atol = 0.1
        @test conv[3] ≈ 2 atol = 0.1
        @test conv[1] ≤ conv[2] ≤ conv[3]
    end
end

@testset "∂ Center -> Face and ∂ Face-> Center (uniform)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_grad_sin_c = zeros(FT, length(n_elems_seq))
    err_div_sin_c = zeros(FT, length(n_elems_seq))
    err_grad_z_f = zeros(FT, length(n_elems_seq))
    err_grad_cos_f1 = zeros(FT, length(n_elems_seq))
    err_grad_cos_f2 = zeros(FT, length(n_elems_seq))
    err_div_sin_f = zeros(FT, length(n_elems_seq))
    err_div_cos_f = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_tags = (:left, :right),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        faces = getproperty(Fields.coordinate_field(fs), :z)

        # Face -> Center operators:
        # GradientF2C
        # f(z) = sin(z)
        ∇ᶜ = Operators.GradientF2C()
        gradsinᶜ = Geometry.CartesianVector.(∇ᶜ.(sin.(faces)))

        # DivergenceF2C
        # f(z) = sin(z)
        divᶜ = Operators.DivergenceF2C()
        divsinᶜ = divᶜ.(Geometry.Cartesian3Vector.(sin.(faces)))

        # Center -> Face operators:
        # GradientC2F, SetValue
        # f(z) = z
        ∇ᶠ⁰ = Operators.GradientC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(pi)),
        )
        ∂zᶠ = Geometry.CartesianVector.(∇ᶠ⁰.(centers))

        # GradientC2F, SetValue
        # f(z) = cos(z)
        ∇ᶠ¹ = Operators.GradientC2F(
            left = Operators.SetValue(FT(1)),
            right = Operators.SetValue(FT(-1)),
        )
        gradcosᶠ¹ = Geometry.CartesianVector.(∇ᶠ¹.(cos.(centers)))

        # GradientC2F, SetGradient
        # f(z) = cos(z)
        ∇ᶠ² = Operators.GradientC2F(
            left = Operators.SetGradient(Geometry.Cartesian3Vector(FT(0))),
            right = Operators.SetGradient(Geometry.Cartesian3Vector(FT(0))),
        )
        gradcosᶠ² = Geometry.CartesianVector.(∇ᶠ².(cos.(centers)))

        # DivergenceC2F, SetValue
        # f(z) = sin(z)
        divᶠ⁰ = Operators.DivergenceC2F(
            left = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
            right = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        )
        divsinᶠ = divᶠ⁰.(Geometry.Cartesian3Vector.(sin.(centers)))

        # DivergenceC2F, SetDivergence
        # f(z) = cos(z)
        divᶠ¹ = Operators.DivergenceC2F(
            left = Operators.SetDivergence(FT(0)),
            right = Operators.SetDivergence(FT(0)),
        )
        divcosᶠ = divᶠ¹.(Geometry.Cartesian3Vector.(cos.(centers)))

        Δh[k] = cs.face_local_geometry.J[1]
        # Errors
        err_grad_sin_c[k] =
            norm(gradsinᶜ .- Geometry.Cartesian3Vector.(cos.(centers)))
        err_div_sin_c[k] = norm(divsinᶜ .- cos.(centers))
        err_grad_z_f[k] = norm(∂zᶠ .- Geometry.Cartesian3Vector.(ones(FT, fs)))
        err_grad_cos_f1[k] =
            norm(gradcosᶠ¹ .- Geometry.Cartesian3Vector.(.-sin.(faces)))
        err_grad_cos_f2[k] =
            norm(gradcosᶠ² .- Geometry.Cartesian3Vector.(.-sin.(faces)))
        err_div_sin_f[k] = norm(
            divsinᶠ .-
            (Geometry.Cartesian3Vector.(cos.(faces))).components.data.:1,
        )
        err_div_cos_f[k] = norm(
            divcosᶠ .-
            (Geometry.Cartesian3Vector.(.-sin.(faces))).components.data.:1,
        )
    end

    # GradientF2C conv, with f(z) = sin(z)
    conv_grad_sin_c = convergence_rate(err_grad_sin_c, Δh)
    # DivergenceF2C conv, with f(z) = sin(z)
    conv_div_sin_c = convergence_rate(err_div_sin_c, Δh)
    # GradientC2F conv, with f(z) = z, SetValue
    conv_grad_z = convergence_rate(err_grad_z_f, Δh)
    # GradientC2F conv, with f(z) = cos(z), SetValue
    conv_grad_cos_f1 = convergence_rate(err_grad_cos_f1, Δh)
    # GradientC2F conv, with f(z) = cos(z), SetGradient
    conv_grad_cos_f2 = convergence_rate(err_grad_cos_f2, Δh)
    # DivergenceC2F conv, with f(z) = sin(z), SetValue
    conv_div_sin_f = convergence_rate(err_div_sin_f, Δh)
    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    conv_div_cos_f = convergence_rate(err_div_cos_f, Δh)

    # GradientF2C conv, with f(z) = sin(z)
    @test err_grad_sin_c[3] ≤ err_grad_sin_c[2] ≤ err_grad_sin_c[1] ≤ 0.1
    @test conv_grad_sin_c[1] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[2] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[3] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[1] ≤ conv_grad_sin_c[2] ≤ conv_grad_sin_c[3]


    # DivergenceF2C conv, with f(z) = sin(z)
    @test err_div_sin_c[3] ≤ err_div_sin_c[2] ≤ err_div_sin_c[1] ≤ 0.1
    @test conv_div_sin_c[1] ≈ 2 atol = 0.1
    @test conv_div_sin_c[2] ≈ 2 atol = 0.1
    @test conv_div_sin_c[3] ≈ 2 atol = 0.1
    @test conv_div_sin_c[1] ≤ conv_div_sin_c[2] ≤ conv_div_sin_c[3]

    # GradientC2F conv, with f(z) = z, SetValue
    @test norm(err_grad_z_f) ≤ 200 * eps(FT)
    # Convergence rate for this case is noisy because error very small

    # GradientC2F conv, with f(z) = cos(z), SetValue
    @test err_grad_cos_f1[3] ≤ err_grad_cos_f1[2] ≤ err_grad_cos_f1[1] ≤ 0.1
    @test conv_grad_cos_f1[1] ≈ 1.5 atol = 0.1
    @test conv_grad_cos_f1[2] ≈ 1.5 atol = 0.1
    @test conv_grad_cos_f1[3] ≈ 1.5 atol = 0.1
    # @test conv_grad_cos_f1[1] ≤ conv_grad_cos_f1[2] ≤ conv_grad_cos_f1[3]

    # GradientC2F conv, with f(z) = cos(z), SetGradient
    @test err_grad_cos_f2[3] ≤ err_grad_cos_f2[2] ≤ err_grad_cos_f2[1] ≤ 0.1
    @test conv_grad_cos_f2[1] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[2] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[3] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[1] ≤ conv_grad_cos_f2[2] ≤ conv_grad_cos_f2[3]

    # DivergenceC2F conv, with f(z) = sin(z), SetValue
    @test err_div_sin_f[3] ≤ err_div_sin_f[2] ≤ err_div_sin_f[1] ≤ 0.1
    @test conv_div_sin_f[1] ≈ 2 atol = 0.1
    @test conv_div_sin_f[2] ≈ 2 atol = 0.1
    @test conv_div_sin_f[3] ≈ 2 atol = 0.1
    @test conv_div_sin_f[1] ≤ conv_div_sin_f[2] ≤ conv_div_sin_f[3]

    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    @test err_div_cos_f[3] ≤ err_div_cos_f[2] ≤ err_div_cos_f[1] ≤ 0.1
    @test conv_div_cos_f[1] ≈ 2 atol = 0.1
    @test conv_div_cos_f[2] ≈ 2 atol = 0.1
    @test conv_div_cos_f[3] ≈ 2 atol = 0.1
    @test conv_div_cos_f[1] ≤ conv_div_cos_f[2] ≤ conv_div_cos_f[3]

end
