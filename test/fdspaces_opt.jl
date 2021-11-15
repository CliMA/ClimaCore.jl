using Test
using JET

using IntervalSets

import ClimaCore: Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

# We need to pull these broadcasted expressions out as
# toplevel functions due to how broadcast expressions are
# lowered so JETTest can have a single callsite to analyze.

function opt_InterpolateF2C(face_field)
    I = Operators.InterpolateF2C()
    return I.(identity.(face_field))
end

function opt_WeightedInterpolateF2C(weights, face_field)
    WI = Operators.WeightedInterpolateF2C()
    return identity.(WI.(weights, face_field))
end

function opt_LeftBiasedF2C(face_field)
    LB = Operators.LeftBiasedF2C(left = Operators.SetValue(0.0))
    return LB.(identity.(face_field))
end

function opt_RightBiasedF2C(face_field)
    RB = Operators.RightBiasedF2C(right = Operators.SetValue(0.0))
    return RB.(identity.(face_field))
end

function opt_GradientF2C(face_field)
    ∇ᶜ = Operators.GradientF2C()
    # TODO: Geometry.WVector.(∇ᶜ.(sin.(faces)))
    return ∇ᶜ.(sin.(face_field))
end

function opt_InterpolateC2F_SetValue(center_field)
    I = Operators.InterpolateC2F(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    return I.(identity.(center_field))
end

function opt_InterpolateC2F_SetGradient(center_field)
    I = Operators.InterpolateC2F(
        left = Operators.SetGradient(0.0),
        right = Operators.SetGradient(0.0),
    )
    return I.(identity.(center_field))
end

function opt_InterpolateC2F_Extrapolate(center_field)
    I = Operators.InterpolateC2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return I.(identity.(center_field))
end

function opt_WeightedInterpolateC2F_SetValue(weights, center_field)
    WI = Operators.WeightedInterpolateC2F(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    return identity.(WI.(weights, center_field))
end

function opt_WeightedInterpolateC2F_SetGradient(weights, center_field)
    WI = Operators.WeightedInterpolateC2F(
        left = Operators.SetGradient(0.0),
        right = Operators.SetGradient(0.0),
    )
    return identity.(WI.(weights, center_field))
end

function opt_WeightedInterpolateC2F_Extrapolate(weights, center_field)
    WI = Operators.WeightedInterpolateC2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return identity.(WI.(weights, center_field))
end

function opt_LeftBiasedC2F(center_field)
    LB = Operators.LeftBiasedC2F(left = Operators.SetValue(0.0))
    return LB.(identity.(center_field))
end

function opt_RightBiasedC2F(center_field)
    RB = Operators.RightBiasedC2F(right = Operators.SetValue(0.0))
    return RB.(identity.(center_field))
end

function opt_GradientC2F_SetValue(center_field)
    ∇ᶠ = Operators.GradientC2F(
        left = Operators.SetValue(1.0),
        right = Operators.SetValue(-1.0),
    )
    #TODO: Geometry.WVector.(∇ᶠ.(cos.(centers)))
    return ∇ᶠ.(cos.(center_field))
end

function opt_DivergenceC2F_SetValue(center_field)
    divᶠ = Operators.DivergenceC2F(
        left = Operators.SetValue(Geometry.WVector(0.0)),
        right = Operators.SetValue(Geometry.WVector(0.0)),
    )
    return divᶠ.(Geometry.WVector.(sin.(center_field)))
end

function opt_DivergenceC2F_SetDivergence(center_field)
    # DivergenceC2F, SetDivergence
    divᶠ = Operators.DivergenceC2F(
        left = Operators.SetDivergence(0.0),
        right = Operators.SetDivergence(0.0),
    )
    return divᶠ.(Geometry.WVector.(cos.(center_field)))
end

# Test that Julia ia able to optimize Stencil operations v1.7+
@static if @isdefined(var"@test_opt")
    @testset "Scalar Field FiniteDifferenceSpaces optimizations" begin
        for FT in (Float64,)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(0.0),
                Geometry.ZPoint{FT}(pi);
                boundary_tags = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain; nelems = 16)

            center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
            face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

            # face space operators
            @test_opt sum(ones(FT, face_space))

            faces = getproperty(Fields.coordinate_field(face_space), :z)
            @test_opt sum(sin.(faces))

            @test_opt opt_InterpolateF2C(faces)

            weights = ones(FT, face_space)
            @test_opt opt_WeightedInterpolateF2C(weights, faces)

            @test_opt opt_LeftBiasedF2C(faces)
            @test_opt opt_RightBiasedF2C(faces)
            @test_opt opt_GradientF2C(faces)
            # @test_opt opt_DivergenceF2C(faces)

            # center space operators
            @test_opt sum(ones(FT, center_space))
            centers = getproperty(Fields.coordinate_field(center_space), :z)
            @test_opt sum(sin.(centers))

            @test_opt opt_InterpolateC2F_SetValue(centers)
            # @test_opt opt_InterpolateC2F_SetGradient(centers)
            @test_opt opt_InterpolateC2F_Extrapolate(centers)

            weights = ones(FT, center_space)
            @test_opt opt_WeightedInterpolateC2F_SetValue(weights, centers)
            # @test_opt opt_WeightedInterpolateC2F_SetGradient(weights, centers)
            @show opt_WeightedInterpolateC2F_Extrapolate(weights, centers)

            @test_opt opt_LeftBiasedC2F(centers)
            @test_opt opt_RightBiasedC2F(centers)

            @test_opt opt_GradientC2F_SetValue(centers)
            # @test_opt opt_GradientC2F_SetGradient(centers)

            # @test_opt opt_DivergenceC2F_SetValue(centers)
            # @test_opt opt_DivergenceC2F_SetDivergence(centers)
        end
    end
end
