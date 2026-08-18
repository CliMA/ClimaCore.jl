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
    Operators

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

        upwind_c2f = Operators.UpwindBiasedProductC2F(
            bottom = Operators.SetValue(FT(1)),
            top = Operators.SetValue(FT(0)),
        )

        flux_pos = Geometry.WVector.(upwind_c2f.(w_pos, step_c))
        flux_neg = Geometry.WVector.(upwind_c2f.(w_neg, step_c))

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
