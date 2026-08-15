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

# The nonlinear advection operators take no boundary conditions: every face is
# computed with the interior stencil, with out-of-range center indices clamped
# to the domain (ghost cells padded with the closest interior value) and
# out-of-range face indices of the velocity slot clamped likewise. The
# reference for every face, boundary faces included, is therefore the
# operator's own pointwise function applied to hand-clamped stencil values.
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
        mesh = Meshes.IntervalMesh(domain; nelems = n)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
        lg_face = Fields.local_geometry_field(face_space)

        θ = sin.(3 .* Fields.coordinate_field(center_space).z)
        dt = FT(0.1)

        # host copies for the pointwise references
        cpu(x) = Array(parent(x))[:]
        t = cpu(θ)
        clamp_c(j) = clamp(j, 1, n)

        # face i has centers i - 2, i - 1 below it and i, i + 1 above it
        stencil(i) =
            (t[clamp_c(i - 2)], t[clamp_c(i - 1)], t[clamp_c(i)], t[clamp_c(i + 1)])

        for w_sign in (FT(1), FT(-1))
            w = Geometry.WVector.(w_sign .* ones(FT, face_space))
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

            # FCTBorisBook: the ghost padding makes the one-sided differences
            # that bound the corrected flux vanish on the boundary faces, so
            # the corrected flux there is zero.
            A = Geometry.Contravariant3Vector.(w_sign .* ones(FT, face_space))
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
        # is 0 / 0 unless the zero upwind slope short-circuits it (it used to
        # produce NaN limited fluxes).
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

        # The nonlinear operators take no boundary conditions at all.
        @test_throws ErrorException Operators.LinVanLeerC2F(;
            bottom = Operators.Extrapolate(),
            constraint = Operators.MonotoneHarmonic(),
        )
        @test_throws ErrorException Operators.FCTBorisBook(;
            bottom = Operators.Extrapolate(),
        )
        @test_throws ErrorException Operators.FCTZalesak(;
            top = Operators.Extrapolate(),
        )
        @test_throws ErrorException Operators.TVDLimitedFluxC2F(;
            bottom = Operators.Extrapolate(),
            method = Operators.MinModLimiter(),
        )

        # `Extrapolate` on `UpwindBiasedProductC2F` copies the adjacent
        # interior face onto each boundary face.
        θ2 = sin.(3 .* Fields.coordinate_field(center_space).z)
        for w_sign in (FT(1), FT(-1))
            w = Geometry.WVector.(w_sign .* ones(FT, face_space))
            upwind = Operators.UpwindBiasedProductC2F(;
                bottom = Operators.Extrapolate(),
                top = Operators.Extrapolate(),
            )
            flux = cpu(upwind.(w, θ2))
            @test flux[1] == flux[2]
            @test flux[n + 1] == flux[n]
        end
    end
end
