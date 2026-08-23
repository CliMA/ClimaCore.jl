using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Geometry,
    Operators,
    Utilities

@testset "Finite Difference Upwind & Biased Schemes" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        L = FT(2)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(-L / 2),
            Geometry.ZPoint(L / 2);
            boundary_names = (:bottom, :top),
        )
        nelems = 32
        mesh = Meshes.IntervalMesh(domain; nelems)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        zc = Fields.coordinate_field(center_space).z
        zf = Fields.coordinate_field(face_space).z

        # 1. Step function field bounded in [0, 1]
        step_c = @. ifelse(zc < 0, FT(1), FT(0))
        step_f = @. ifelse(zf < 0, FT(1), FT(0))

        # Operators: 1st order biased C2F and F2C
        left_c2f = Operators.LeftBiasedC2F(bottom = Operators.SetValue(FT(1)))
        right_c2f = Operators.RightBiasedC2F(top = Operators.SetValue(FT(0)))
        left_f2c = Operators.LeftBiasedF2C(bottom = Operators.SetValue(FT(1)))
        right_f2c = Operators.RightBiasedF2C(top = Operators.SetValue(FT(0)))

        # A. Monotonicity / Boundedness on step function: 0 <= result <= 1
        res_left_c2f = left_c2f.(step_c)
        res_right_c2f = right_c2f.(step_c)
        res_left_f2c = left_f2c.(step_f)
        res_right_f2c = right_f2c.(step_f)

        @test all(0 .<= parent(res_left_c2f) .<= 1)
        @test all(0 .<= parent(res_right_c2f) .<= 1)
        @test all(0 .<= parent(res_left_f2c) .<= 1)
        @test all(0 .<= parent(res_right_f2c) .<= 1)

        # B. Upwind directionality with physical velocity w
        w_pos = Geometry.WVector.(ones(face_space))
        w_neg = Geometry.WVector.(.-ones(face_space))

        # `UpwindBiasedProductC2F` takes no boundary condition that sets the
        # boundary faces, so its stencil is evaluated there here, with θ_bot
        # and θ_top standing in for the points outside the domain, and imposed
        # with a `SetBoundaryOperator`.
        lg_face = Fields.local_geometry_field(face_space)
        bottom_face = Utilities.PlusHalf(0)
        top_face = Fields.nlevels(zf) - Utilities.PlusHalf(0)
        n_center_levels = Fields.nlevels(step_c)
        function upwind_c2f_with_boundaries(w, θ, θ_bot, θ_top)
            v³ = Geometry.contravariant3.(w, lg_face)
            v³_bot = Fields.level(v³, bottom_face)
            v³_top = Fields.level(v³, top_face)
            θ_first = Fields.Field(
                Fields.field_values(Fields.level(θ, 1)),
                axes(v³_bot),
            )
            θ_last = Fields.Field(
                Fields.field_values(Fields.level(θ, n_center_levels)),
                axes(v³_top),
            )
            set_bcs = Operators.SetBoundaryOperator(
                bottom = Operators.SetValue(
                    Geometry.Contravariant3Vector.(
                        Operators.upwind_biased_product.(
                            v³_bot,
                            θ_bot,
                            θ_first,
                        ),
                    ),
                ),
                top = Operators.SetValue(
                    Geometry.Contravariant3Vector.(
                        Operators.upwind_biased_product.(v³_top, θ_last, θ_top),
                    ),
                ),
            )
            upwind = Operators.UpwindBiasedProductC2F()
            return @. set_bcs(upwind(w, θ))
        end

        flux_pos = Geometry.WVector.(
            upwind_c2f_with_boundaries(w_pos, step_c, FT(1), FT(0)),
        )
        flux_neg = Geometry.WVector.(
            upwind_c2f_with_boundaries(w_neg, step_c, FT(1), FT(0)),
        )

        # For w > 0, flux = w * left_biased
        @test parent(flux_pos) ≈ parent(res_left_c2f)
        # For w < 0, flux = w * right_biased
        @test parent(flux_neg) ≈ -parent(res_right_c2f)

        # C. Constant preservation: biased interpolation of 1 is 1
        left_c2f_const = Operators.LeftBiasedC2F(bottom = Operators.SetValue(FT(1)))
        right_c2f_const = Operators.RightBiasedC2F(top = Operators.SetValue(FT(1)))
        left_f2c_const = Operators.LeftBiasedF2C(bottom = Operators.SetValue(FT(1)))
        right_f2c_const = Operators.RightBiasedF2C(top = Operators.SetValue(FT(1)))

        ones_c = ones(center_space)
        ones_f = ones(face_space)
        @test maximum(abs, parent(left_c2f_const.(ones_c)) .- FT(1)) < 10 * eps(FT)
        @test maximum(abs, parent(right_c2f_const.(ones_c)) .- FT(1)) < 10 * eps(FT)
        @test maximum(abs, parent(left_f2c_const.(ones_f)) .- FT(1)) < 10 * eps(FT)
        @test maximum(abs, parent(right_f2c_const.(ones_f)) .- FT(1)) < 10 * eps(FT)
    end
end

# The nonlinear advection operators compute every face with the interior
# stencil. By default (with Extrapolate{0}), out-of-range center indices are
# clamped to the domain (ghost points padded with the closest interior value)
# and out-of-range face indices of the velocity slot are clamped likewise;
# with a higher-order Extrapolate{N}, the ghost points are instead padded with
# the condition's extrapolation from the in-range interior points of the
# stencil (the order is reduced at the boundary face itself, where only 2
# interior points are in range). The reference for every face, boundary faces
# included, is therefore the operator's own pointwise function applied to
# hand-clamped (or hand-extrapolated) stencil values.

# Ghost-point reconstructions are no longer user-supplied callables; this is
# used to check that custom callables are rejected.
struct CustomGhost end
(::CustomGhost)(closest, second_closest) = 3 * closest - 2 * second_closest

@testset "Nonlinear advection operators at boundary faces" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        device = ClimaComms.device(context)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(FT(0)),
            Geometry.ZPoint(FT(1));
            boundary_names = (:bottom, :top),
        )
        n = 8
        # Stretched mesh, so that the references would catch a wrong level in
        # a local geometry lookup (every J factor is different).
        mesh = Meshes.IntervalMesh(
            domain,
            Meshes.ExponentialStretching(FT(1) / 2);
            nelems = n,
        )
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
        lg_face = Fields.local_geometry_field(face_space)
        zface = Fields.coordinate_field(face_space).z

        θ = sin.(3 .* Fields.coordinate_field(center_space).z)
        dt = FT(0.1)

        # Velocity profiles: uniformly positive, uniformly negative, and
        # sign-alternating within the column, all varying from face to face so
        # that a wrong face index in a velocity lookup changes the result.
        w_profiles = (z -> 1 + z / 2, z -> -1 - z / 2, z -> cospi(5 * z))

        # host copies for the pointwise references
        cpu(x) = Array(parent(x))[:]
        t = cpu(θ)
        clamp_c(j) = clamp(j, 1, n)

        # face i has centers i - 2, i - 1 below it and i, i + 1 above it
        stencil(i) =
            (t[clamp_c(i - 2)], t[clamp_c(i - 1)], t[clamp_c(i)], t[clamp_c(i + 1)])

        for w_fn in w_profiles
            w = @. Geometry.WVector(FT(w_fn(zface)))
            v³ = cpu(Geometry.contravariant3.(w, lg_face))

            # LinVanLeerC2F: materialized directly, no enclosing operator
            for constraint in (
                Operators.AlgebraicMean(),
                Operators.PositiveDefinite(),
                Operators.MonotoneHarmonic(),
                Operators.MonotoneLocalExtrema(),
            )
                op = Operators.LinVanLeerC2F(; constraint)
                flux = cpu(op.(w, θ, dt))
                ref = [op(v³[i], stencil(i)..., dt) for i in 1:(n + 1)]
                @test flux ≈ ref
            end

            # FCTBorisBook: the ghost padding makes the one-sided difference
            # on the boundary side vanish at the boundary faces, and that
            # difference bounds the corrected flux, so the flux there is zero.
            A = @. Geometry.Contravariant3Vector(FT(w_fn(zface)))
            A³ = cpu(A)
            op_bb = Operators.FCTBorisBook()
            flux_bb = cpu(op_bb.(A, θ))
            ref_bb = [op_bb(A³[i], stencil(i)...) for i in 1:(n + 1)]
            @test flux_bb ≈ ref_bb
            @test flux_bb[1] == 0
            @test flux_bb[n + 1] == 0

            # FCTZalesak: the antidiffusive flux is read at the neighboring
            # faces as well, which clamp at the boundary.
            θᵗᵈ = θ .- dt .* θ # any center field with the same structure
            tᵗᵈ = cpu(θᵗᵈ)
            tup(j) = (t[clamp_c(j)] / dt, tᵗᵈ[clamp_c(j)] / dt)
            op_z = Operators.FCTZalesak()
            flux_z = cpu(op_z.(A, tuple.(θ ./ dt, θᵗᵈ ./ dt)))
            ref_z = [
                op_z(
                    A³[max(i - 1, 1)],
                    A³[i],
                    A³[min(i + 1, n + 1)],
                    tup(i - 2),
                    tup(i - 1),
                    tup(i),
                    tup(i + 1),
                ) for i in 1:(n + 1)
            ]
            @test flux_z ≈ ref_z

            # TVDLimitedFluxC2F, with the velocity supplied as contravariant
            # data
            u³ = Geometry.Contravariant3Vector.(
                Geometry.contravariant3.(w, lg_face),
            )
            op_tvd = Operators.TVDLimitedFluxC2F(;
                method = Operators.MinModLimiter(),
            )
            flux_tvd = cpu(op_tvd.(A, θ, u³))
            ref_tvd = [op_tvd(A³[i], stencil(i)..., v³[i]) for i in 1:(n + 1)]
            @test flux_tvd ≈ ref_tvd
        end

        # A field that is flat up to roundoff: the upwind slope is exactly
        # zero at the face where the centered difference is exactly -eps, so
        # the added eps in the slope ratio's denominator cancels and the ratio
        # is 0 / 0, and the limited flux NaN, unless the zero upwind slope
        # short-circuits it.
        z_mid = FT(0.5)
        one_lo = FT(1) - eps(FT)
        zc = Fields.coordinate_field(center_space).z
        θ_flat = @. ifelse(zc < z_mid, FT(1), one_lo)
        w = Geometry.WVector.(ones(FT, face_space))
        A = Geometry.Contravariant3Vector.(ones(FT, face_space))
        u³ = Geometry.Contravariant3Vector.(
            Geometry.contravariant3.(w, lg_face),
        )
        op_tvd =
            Operators.TVDLimitedFluxC2F(; method = Operators.MinModLimiter())
        flux_flat = cpu(op_tvd.(A, θ_flat, u³))
        @test !any(isnan, flux_flat)
        t_flat = cpu(θ_flat)
        drop_face = findfirst(i -> t_flat[i + 1] < t_flat[i], 1:(n - 1)) + 1
        @test t_flat[drop_face - 2] == t_flat[drop_face - 1] # upwind slope 0
        @test flux_flat[drop_face] == 0

        # Ghost-point extrapolation: with Extrapolate{N}, every out-of-range
        # center of a stencil is padded with the condition's extrapolation
        # from the in-range interior points: at the boundary face both ghost
        # points share the extrapolation from the 2 in-range points (so the
        # order is reduced to at most 1 there), and at the face one in from
        # the boundary the single ghost point is extrapolated from the 3
        # in-range points; faces 1, 2, n and n + 1 therefore differ from the
        # clamped `stencil`.
        g2(N, x₁, x₂) = N == 0 ? x₁ : 2x₁ - x₂
        g3(N, x₁, x₂, x₃) = N == 0 ? x₁ : N == 1 ? 2x₁ - x₂ : 3x₁ - 3x₂ + x₃
        function ghost_stencil(i, N)
            i == 1 &&
                return (g2(N, t[1], t[2]), g2(N, t[1], t[2]), t[1], t[2])
            i == 2 && return (g3(N, t[1], t[2], t[3]), t[1], t[2], t[3])
            i == n &&
                return (t[n - 2], t[n - 1], t[n], g3(N, t[n], t[n - 1], t[n - 2]))
            i == n + 1 && return (
                t[n - 1],
                t[n],
                g2(N, t[n], t[n - 1]),
                g2(N, t[n], t[n - 1]),
            )
            return stencil(i)
        end
        extrapolate_bcs(N) =
            (; bottom = Operators.Extrapolate(N), top = Operators.Extrapolate(N))
        for w_fn in w_profiles
            w = @. Geometry.WVector(FT(w_fn(zface)))
            v³ = cpu(Geometry.contravariant3.(w, lg_face))
            A = @. Geometry.Contravariant3Vector(FT(w_fn(zface)))
            A³ = cpu(A)

            # N = 0 also checks that the default Extrapolate{0} matches the
            # hand-written index clamping (`ghost_stencil(i, 0)` reduces to
            # `stencil(i)`).
            for N in 0:2
                bcs = extrapolate_bcs(N)

                op_bb = Operators.FCTBorisBook(; bcs...)
                flux_bb = cpu(op_bb.(A, θ))
                ref_bb = [
                    op_bb(A³[i], ghost_stencil(i, N)...) for i in 1:(n + 1)
                ]
                @test flux_bb ≈ ref_bb

                op_lvl = Operators.LinVanLeerC2F(;
                    constraint = Operators.AlgebraicMean(),
                    bcs...,
                )
                flux_lvl = cpu(op_lvl.(w, θ, dt))
                ref_lvl = [
                    op_lvl(v³[i], ghost_stencil(i, N)..., dt)
                    for i in 1:(n + 1)
                ]
                @test flux_lvl ≈ ref_lvl

                u³ = Geometry.Contravariant3Vector.(
                    Geometry.contravariant3.(w, lg_face),
                )
                op_tvd = Operators.TVDLimitedFluxC2F(;
                    method = Operators.MinModLimiter(),
                    bcs...,
                )
                flux_tvd = cpu(op_tvd.(A, θ, u³))
                ref_tvd = [
                    op_tvd(A³[i], ghost_stencil(i, N)..., v³[i])
                    for i in 1:(n + 1)
                ]
                @test flux_tvd ≈ ref_tvd
            end

            # FCTZalesak: the extrapolation applies componentwise to its
            # tuple-valued stencil entries. As for the scalar-valued operators
            # above, the boundary face's ghost tuples share the 2-point
            # extrapolation (order at most 1), while the one-in face's single
            # ghost tuple uses the full order over the 3 in-range entries.
            θᵗᵈ = θ .- dt .* θ
            tᵗᵈ = cpu(θᵗᵈ)
            tup(j) = (t[clamp_c(j)] / dt, tᵗᵈ[clamp_c(j)] / dt)
            gtup2(N, j₁, j₂) = N == 0 ? tup(j₁) : 2 .* tup(j₁) .- tup(j₂)
            gtup3(N, j₁, j₂, j₃) =
                N == 0 ? tup(j₁) :
                N == 1 ? 2 .* tup(j₁) .- tup(j₂) :
                3 .* tup(j₁) .- 3 .* tup(j₂) .+ tup(j₃)
            for N in 0:2
                ghost_tup(j, i) =
                    (i == 1 && j <= 0) ? gtup2(N, 1, 2) :
                    (i == 2 && j == 0) ? gtup3(N, 1, 2, 3) :
                    (i == n + 1 && j >= n + 1) ? gtup2(N, n, n - 1) :
                    (i == n && j == n + 1) ? gtup3(N, n, n - 1, n - 2) :
                    tup(j)
                op_z = Operators.FCTZalesak(; extrapolate_bcs(N)...)
                flux_z = cpu(op_z.(A, tuple.(θ ./ dt, θᵗᵈ ./ dt)))
                ref_z = [
                    op_z(
                        A³[max(i - 1, 1)],
                        A³[i],
                        A³[min(i + 1, n + 1)],
                        ghost_tup(i - 2, i),
                        ghost_tup(i - 1, i),
                        ghost_tup(i, i),
                        ghost_tup(i + 1, i),
                    ) for i in 1:(n + 1)
                ]
                @test flux_z ≈ ref_z
            end
        end

        # The advection operators only accept Extrapolate boundary
        # conditions. Everything else is an error, including the custom
        # callable ghost-point reconstructions they used to accept.
        @test_throws AssertionError Operators.LinVanLeerC2F(;
            bottom = Operators.SetValue(FT(0)),
            constraint = Operators.MonotoneHarmonic(),
        )
        @test_throws AssertionError Operators.FCTBorisBook(;
            bottom = CustomGhost(),
        )
        @test_throws AssertionError Operators.FCTZalesak(;
            top = CustomGhost(),
        )
        @test_throws AssertionError Operators.TVDLimitedFluxC2F(;
            bottom = CustomGhost(),
            method = Operators.MinModLimiter(),
        )
        @test_throws AssertionError Operators.FCTBorisBook(;
            bottom = FT(1),
        )
        @test_throws AssertionError Operators.UpwindBiasedProductC2F(;
            bottom = Operators.SetValue(FT(0)),
        )
        @test_throws AssertionError Operators.Upwind3rdOrderBiasedProductC2F(;
            bottom = CustomGhost(),
        )
        @test Operators.FCTBorisBook(;
            bottom = Operators.Extrapolate(0),
            top = Operators.Extrapolate(1),
        ).bcs.top === Operators.Extrapolate(1)
        @test Operators.UpwindBiasedProductC2F(;
            bottom = Operators.Extrapolate(2),
            top = Operators.Extrapolate(2),
        ).bcs.bottom === Operators.Extrapolate(2)
        @test Operators.Upwind3rdOrderBiasedProductC2F(;
            bottom = Operators.Extrapolate(0),
            top = Operators.Extrapolate(2),
        ).bcs.top === Operators.Extrapolate(2)

        # The deprecated one-sided conditions are aliases for Extrapolate
        @test Operators.FirstOrderOneSided() === Operators.Extrapolate(0)
        @test Operators.ThirdOrderOneSided() === Operators.Extrapolate(1)

        # Extrapolate{0} is added to `bcs` by default when no boundary
        # conditions are given
        default_bcs = (;
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        @test Operators.UpwindBiasedProductC2F().bcs === default_bcs
        @test Operators.Upwind3rdOrderBiasedProductC2F().bcs === default_bcs
        @test Operators.FCTBorisBook().bcs === default_bcs
        @test Operators.FCTZalesak().bcs === default_bcs
        @test Operators.TVDLimitedFluxC2F(;
            method = Operators.MinModLimiter(),
        ).bcs === default_bcs
        @test Operators.LinVanLeerC2F(;
            constraint = Operators.AlgebraicMean(),
        ).bcs === default_bcs

        # only advection operators whose interior stencil is linear are
        # rewritten as operator-matrix multiplies
        @test Operators.has_linear_stencil(Operators.UpwindBiasedProductC2F())
        @test Operators.has_linear_stencil(
            Operators.Upwind3rdOrderBiasedProductC2F(;
                bottom = Operators.Extrapolate(2),
                top = Operators.Extrapolate(2),
            ),
        )
        @test !Operators.has_linear_stencil(Operators.FCTBorisBook())

        # UpwindBiasedProductC2F's stencil only reaches a ghost point at the
        # boundary face itself, where a single interior point is in range, so
        # every extrapolation order reduces to the value of the closest
        # interior point, and the flux at a boundary face is
        # `v³ θ[closest center]` regardless of the upwind direction.
        θ2 = sin.(3 .* Fields.coordinate_field(center_space).z)
        t2 = cpu(θ2)
        for w_sign in (FT(1), FT(-1))
            w = Geometry.WVector.(w_sign .* ones(FT, face_space))
            v³ = cpu(Geometry.contravariant3.(w, lg_face))
            for upwind in (
                Operators.UpwindBiasedProductC2F(),
                Operators.UpwindBiasedProductC2F(;
                    bottom = Operators.Extrapolate(2),
                    top = Operators.Extrapolate(2),
                ),
            )
                flux = cpu(upwind.(w, θ2))
                @test flux[1] == v³[1] * t2[1]
                @test flux[n + 1] == v³[n + 1] * t2[n]
            end
        end

        # Extrapolate on Upwind3rdOrderBiasedProductC2F evaluates the interior
        # stencil with extrapolated ghost points: at the boundary face itself
        # both ghost points share the extrapolation from the 2 in-range
        # interior points (so the order is reduced to at most 1 there), and at
        # the face one in from the boundary the single ghost point is
        # extrapolated from the 3 in-range points.
        upwind3rd(v, a⁻⁻, a⁻, a⁺, a⁺⁺) =
            v ≥ 0 ? v * (-2a⁻⁻ + 10a⁻ + 4a⁺) / 12 :
            v * (4a⁻ + 10a⁺ - 2a⁺⁺) / 12
        for w_sign in (FT(1), FT(-1))
            w = Geometry.WVector.(w_sign .* ones(FT, face_space))
            v³ = cpu(Geometry.contravariant3.(w, lg_face))
            for N in 0:2
                gb2 = g2(N, t2[1], t2[2])           # bottom face ghosts
                gb3 = g3(N, t2[1], t2[2], t2[3])    # bottom one-in ghost
                gt2 = g2(N, t2[n], t2[n - 1])       # top face ghosts
                gt3 = g3(N, t2[n], t2[n - 1], t2[n - 2]) # top one-in ghost
                upwind = Operators.Upwind3rdOrderBiasedProductC2F(;
                    bottom = Operators.Extrapolate(N),
                    top = Operators.Extrapolate(N),
                )
                flux = cpu(upwind.(w, θ2))
                @test flux[1] ≈ upwind3rd(v³[1], gb2, gb2, t2[1], t2[2])
                @test flux[2] ≈ upwind3rd(v³[2], gb3, t2[1], t2[2], t2[3])
                @test flux[n] ≈
                      upwind3rd(v³[n], t2[n - 2], t2[n - 1], t2[n], gt3)
                @test flux[n + 1] ≈
                      upwind3rd(v³[n + 1], t2[n - 1], t2[n], gt2, gt2)
            end
        end
    end
end

@testset "Ghost-point extrapolation on a 2-center column" begin
    # On a column with only 2 centers, the middle face is one in from both
    # boundaries: each of its two out-of-range centers is padded with the
    # extrapolation of its own boundary's condition, from the only 2 in-range
    # points (so the order is reduced to at most 1, as at a boundary face).
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(FT(0)),
            Geometry.ZPoint(FT(1));
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = 2)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
        lg_face = Fields.local_geometry_field(face_space)

        cpu(x) = Array(parent(x))[:]
        θ = sin.(3 .* Fields.coordinate_field(center_space).z)
        t = cpu(θ)
        dt = FT(0.1)
        g2(N, x₁, x₂) = N == 0 ? x₁ : 2x₁ - x₂
        upwind3rd(v, a⁻⁻, a⁻, a⁺, a⁺⁺) =
            v ≥ 0 ? v * (-2a⁻⁻ + 10a⁻ + 4a⁺) / 12 :
            v * (4a⁻ + 10a⁺ - 2a⁺⁺) / 12

        # Face-varying velocity, so that a wrong face index in a velocity
        # lookup changes the result even on this short column.
        zface = Fields.coordinate_field(face_space).z
        for w_sign in (FT(1), FT(-1)), N in 0:2
            w = @. Geometry.WVector(w_sign * (1 + zface))
            v³ = cpu(Geometry.contravariant3.(w, lg_face))
            A = @. Geometry.Contravariant3Vector(w_sign * (1 + zface))
            A³ = cpu(A)
            bcs =
                (; bottom = Operators.Extrapolate(N), top = Operators.Extrapolate(N))

            g_bot = g2(N, t[1], t[2]) # every bottom ghost
            g_top = g2(N, t[2], t[1]) # every top ghost
            stencils = (
                (g_bot, g_bot, t[1], t[2]), # bottom face
                (g_bot, t[1], t[2], g_top), # middle face: ghosts on both sides
                (t[1], t[2], g_top, g_top), # top face
            )

            op_lvl = Operators.LinVanLeerC2F(;
                constraint = Operators.AlgebraicMean(),
                bcs...,
            )
            flux_lvl = cpu(op_lvl.(w, θ, dt))
            @test flux_lvl ≈ [op_lvl(v³[i], stencils[i]..., dt) for i in 1:3]

            op_bb = Operators.FCTBorisBook(; bcs...)
            flux_bb = cpu(op_bb.(A, θ))
            @test flux_bb ≈ [op_bb(A³[i], stencils[i]...) for i in 1:3]

            # FCTZalesak: neighboring velocities clamp to the column's 3 faces
            θᵗᵈ = θ .- dt .* θ
            tᵗᵈ = cpu(θᵗᵈ)
            tup2c(j) = (t[j] / dt, tᵗᵈ[j] / dt)
            gtup_bot = N == 0 ? tup2c(1) : 2 .* tup2c(1) .- tup2c(2)
            gtup_top = N == 0 ? tup2c(2) : 2 .* tup2c(2) .- tup2c(1)
            tup_stencils = (
                (gtup_bot, gtup_bot, tup2c(1), tup2c(2)),
                (gtup_bot, tup2c(1), tup2c(2), gtup_top),
                (tup2c(1), tup2c(2), gtup_top, gtup_top),
            )
            op_z = Operators.FCTZalesak(; bcs...)
            flux_z = cpu(op_z.(A, tuple.(θ ./ dt, θᵗᵈ ./ dt)))
            @test flux_z ≈ [
                op_z(
                    A³[max(i - 1, 1)],
                    A³[i],
                    A³[min(i + 1, 3)],
                    tup_stencils[i]...,
                ) for i in 1:3
            ]

            # TVDLimitedFluxC2F, with the velocity supplied as contravariant
            # data
            u³ = Geometry.Contravariant3Vector.(
                Geometry.contravariant3.(w, lg_face),
            )
            op_tvd = Operators.TVDLimitedFluxC2F(;
                method = Operators.MinModLimiter(),
                bcs...,
            )
            flux_tvd = cpu(op_tvd.(A, θ, u³))
            @test flux_tvd ≈ [op_tvd(A³[i], stencils[i]..., v³[i]) for i in 1:3]

            # Upwind3rdOrderBiasedProductC2F is rewritten as an operator-matrix
            # multiply; at the middle face its boundary row must fold the ghost
            # extrapolations of both boundaries
            op_u3 = Operators.Upwind3rdOrderBiasedProductC2F(; bcs...)
            flux_u3 = cpu(op_u3.(w, θ))
            @test flux_u3 ≈ [upwind3rd(v³[i], stencils[i]...) for i in 1:3]

            # UpwindBiasedProductC2F's boundary faces reduce to v³ times the
            # closest center for every extrapolation order
            upwind1(v, a⁻, a⁺) = ((v + abs(v)) * a⁻ + (v - abs(v)) * a⁺) / 2
            op_u1 = Operators.UpwindBiasedProductC2F(; bcs...)
            flux_u1 = cpu(op_u1.(w, θ))
            @test flux_u1 ≈
                  [v³[1] * t[1], upwind1(v³[2], t[1], t[2]), v³[3] * t[2]]
        end

        # On a 1-center column, every stencil point collapses to the single
        # interior value, so the flux is v³ θ[1] at both faces for any order.
        mesh1 = Meshes.IntervalMesh(domain; nelems = 1)
        topology1 = Topologies.IntervalTopology(context, mesh1)
        center_space1 = Spaces.CenterFiniteDifferenceSpace(topology1)
        face_space1 = Spaces.FaceFiniteDifferenceSpace(center_space1)
        lg_face1 = Fields.local_geometry_field(face_space1)
        θ1 = FT(0.3) .* ones(center_space1)
        zface1 = Fields.coordinate_field(face_space1).z
        for w_sign in (FT(1), FT(-1)), N in 0:2
            # Face-varying velocity, so the two faces cannot be confused
            w1 = @. Geometry.WVector(w_sign * (1 + zface1))
            v³1 = cpu(Geometry.contravariant3.(w1, lg_face1))
            A1 = @. Geometry.Contravariant3Vector(w_sign * (1 + zface1))
            u³1 = Geometry.Contravariant3Vector.(
                Geometry.contravariant3.(w1, lg_face1),
            )
            bcs =
                (; bottom = Operators.Extrapolate(N), top = Operators.Extrapolate(N))
            for flux in (
                cpu(Operators.UpwindBiasedProductC2F(; bcs...).(w1, θ1)),
                cpu(Operators.Upwind3rdOrderBiasedProductC2F(; bcs...).(w1, θ1)),
                cpu(
                    Operators.LinVanLeerC2F(;
                        constraint = Operators.AlgebraicMean(),
                        bcs...,
                    ).(
                        w1,
                        θ1,
                        dt,
                    ),
                ),
            )
                @test flux ≈ FT(0.3) .* v³1
            end
            # Every stencil difference vanishes on a 1-center column, so the
            # corrected/limited fluxes are exactly zero
            @test cpu(Operators.FCTBorisBook(; bcs...).(A1, θ1)) == [0, 0]
            @test cpu(
                Operators.FCTZalesak(; bcs...).(
                    A1,
                    tuple.(θ1 ./ dt, θ1 ./ dt),
                ),
            ) == [0, 0]
            @test cpu(
                Operators.TVDLimitedFluxC2F(;
                    method = Operators.MinModLimiter(),
                    bcs...,
                ).(
                    A1,
                    θ1,
                    u³1,
                ),
            ) == [0, 0]
        end
    end
end

@testset "Nonlinear advection operators on a periodic column" begin
    # On a periodic column there are no boundaries: stencil indices of the
    # advected field and the neighboring-face velocity indices wrap around
    # instead of clamping, and boundary conditions are ignored.
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(FT(0)),
            Geometry.ZPoint(FT(1));
            periodic = true,
        )
        n = 8
        mesh = Meshes.IntervalMesh(domain; nelems = n)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
        lg_face = Fields.local_geometry_field(face_space)
        zface = Fields.coordinate_field(face_space).z

        cpu(x) = Array(parent(x))[:]
        θ = sin.(3 .* Fields.coordinate_field(center_space).z)
        t = cpu(θ)
        dt = FT(0.1)

        # face i sits between centers i - 1 and i, all indices modulo n
        wrap(j) = mod1(j, n)
        stencil(i) = (t[wrap(i - 2)], t[wrap(i - 1)], t[wrap(i)], t[wrap(i + 1)])

        # sign-alternating, face-varying velocity
        w = @. Geometry.WVector(FT(cospi(5 * zface)))
        v³ = cpu(Geometry.contravariant3.(w, lg_face))
        A = @. Geometry.Contravariant3Vector(FT(cospi(5 * zface)))
        A³ = cpu(A)
        @test length(v³) == n # periodic columns store n faces

        op_lvl = Operators.LinVanLeerC2F(;
            constraint = Operators.MonotoneLocalExtrema(),
        )
        flux_lvl = cpu(op_lvl.(w, θ, dt))
        @test flux_lvl ≈ [op_lvl(v³[i], stencil(i)..., dt) for i in 1:n]

        op_bb = Operators.FCTBorisBook()
        flux_bb = cpu(op_bb.(A, θ))
        @test flux_bb ≈ [op_bb(A³[i], stencil(i)...) for i in 1:n]

        θᵗᵈ = θ .- dt .* θ
        tᵗᵈ = cpu(θᵗᵈ)
        tup(j) = (t[wrap(j)] / dt, tᵗᵈ[wrap(j)] / dt)
        op_z = Operators.FCTZalesak()
        flux_z = cpu(op_z.(A, tuple.(θ ./ dt, θᵗᵈ ./ dt)))
        @test flux_z ≈ [
            op_z(
                A³[wrap(i - 1)],
                A³[i],
                A³[wrap(i + 1)],
                tup(i - 2),
                tup(i - 1),
                tup(i),
                tup(i + 1),
            ) for i in 1:n
        ]

        u³ = Geometry.Contravariant3Vector.(
            Geometry.contravariant3.(w, lg_face),
        )
        op_tvd = Operators.TVDLimitedFluxC2F(;
            method = Operators.MinModLimiter(),
        )
        flux_tvd = cpu(op_tvd.(A, θ, u³))
        @test flux_tvd ≈ [op_tvd(A³[i], stencil(i)..., v³[i]) for i in 1:n]
    end
end

@testset "Advection boundary condition names must match the space's" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        make_column(boundary_names) = begin
            domain = Domains.IntervalDomain(
                Geometry.ZPoint(FT(0)),
                Geometry.ZPoint(FT(1));
                boundary_names,
            )
            mesh = Meshes.IntervalMesh(domain; nelems = 8)
            topology = Topologies.IntervalTopology(context, mesh)
            center_space = Spaces.CenterFiniteDifferenceSpace(topology)
            (center_space, Spaces.FaceFiniteDifferenceSpace(center_space))
        end
        (center_space, face_space) = make_column((:bottom, :top))
        θ = ones(center_space)
        w = Geometry.WVector.(ones(FT, face_space))
        A = Geometry.Contravariant3Vector.(ones(FT, face_space))

        # A boundary condition whose keyword name matches neither of the
        # space's boundary names is an error at broadcast time, instead of
        # being silently ignored in favor of the default reconstruction.
        op_wrong_names = Operators.FCTBorisBook(;
            left = Operators.Extrapolate(1),
            right = Operators.Extrapolate(1),
        )
        @test_throws "must be named after a boundary" op_wrong_names.(A, θ)
        op_typo = Operators.Upwind3rdOrderBiasedProductC2F(;
            botom = Operators.Extrapolate(1), # sic
            top = Operators.Extrapolate(1),
        )
        @test_throws "must be named after a boundary" op_typo.(w, θ)

        # The default (bottom, top) pair of Extrapolate{0}s applies at any
        # boundary names: constructing the operator with no boundary
        # conditions works on a space whose boundaries are named differently,
        # and matches the explicitly-named equivalent.
        (center_lr, face_lr) = make_column((:left, :right))
        θ_lr = sin.(3 .* Fields.coordinate_field(center_lr).z)
        w_lr = Geometry.WVector.(ones(FT, face_lr))
        default_flux = parent(Operators.Upwind3rdOrderBiasedProductC2F().(w_lr, θ_lr))
        named_flux = parent(
            Operators.Upwind3rdOrderBiasedProductC2F(;
                left = Operators.Extrapolate(0),
                right = Operators.Extrapolate(0),
            ).(
                w_lr,
                θ_lr,
            ),
        )
        @test default_flux == named_flux
        # ... while non-default conditions under mismatched names still error
        op_bt_on_lr = Operators.FCTBorisBook(;
            bottom = Operators.Extrapolate(1),
            top = Operators.Extrapolate(1),
        )
        A_lr = Geometry.Contravariant3Vector.(ones(FT, face_lr))
        @test_throws "must be named after a boundary" op_bt_on_lr.(A_lr, θ_lr)
    end
end
