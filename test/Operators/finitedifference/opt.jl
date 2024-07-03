#=
julia --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "opt.jl"))
=#
using Test
using JET

using IntervalSets

import ClimaCore
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Meshes, Spaces, Fields, Operators
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

function opt_AdvectionF2F(face_vel, face_field)
    A = Operators.AdvectionF2F()
    return A.(face_vel, identity.(face_field))
end

function opt_FluxCorrectionF2F_Extrapolate(face_vel, face_field)
    FC = Operators.FluxCorrectionF2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return FC.(face_vel, identity.(face_field))
end

function opt_GradientF2C(face_field)
    ∇ᶜ = Operators.GradientF2C()
    return Geometry.WVector.(∇ᶜ.(sin.(face_field)))
end

function opt_DivergenceF2C(face_field)
    divᶜ = Operators.DivergenceF2C()
    return divᶜ.(Geometry.WVector.(sin.(face_field)))
end

function opt_SetBoundary_SetValue(face_field)
    B = Operators.SetBoundaryOperator(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    return B.(sin.(face_field))
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
        left = Operators.SetGradient(Geometry.WVector(0.0)),
        right = Operators.SetGradient(Geometry.WVector(0.0)),
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
        left = Operators.SetGradient(Geometry.WVector(0.0)),
        right = Operators.SetGradient(Geometry.WVector(0.0)),
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

function opt_UpwindBiasedProductC2F_SetValue(face_vel, center_field)
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    return UB.(face_vel, identity.(center_field))
end

function opt_UpwindBiasedProductC2F_Extrapolate(face_vel, center_field)
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return UB.(face_vel, identity.(center_field))
end

function opt_AdvectionC2C_SetValue(face_vel, center_field)
    A = Operators.AdvectionC2C(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    return A.(face_vel, identity.(center_field))
end

function opt_AdvectionC2C_Extrapolate(face_vel, center_field)
    A = Operators.AdvectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return A.(face_vel, identity.(center_field))
end

function opt_FluxCorrectionC2C_Extrapolate(face_vel, center_field)
    FC = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return FC.(face_vel, identity.(center_field))
end

function opt_GradientC2F_SetValue(center_field)
    ∇ᶠ = Operators.GradientC2F(
        left = Operators.SetValue(1.0),
        right = Operators.SetValue(-1.0),
    )
    return Geometry.WVector.(∇ᶠ.(cos.(center_field)))
end

function opt_GradientC2F_SetGradient(center_field)
    ∇ᶠ = Operators.GradientC2F(
        left = Operators.SetGradient(Geometry.WVector(0.0)),
        right = Operators.SetGradient(Geometry.WVector(0.0)),
    )
    return Geometry.WVector.(∇ᶠ.(cos.(center_field)))
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


function opt_CurlC2F_SetValue(center_field)
    # DivergenceC2F, SetDivergence
    curlᶠ = Operators.CurlC2F(
        left = Operators.SetValue(Geometry.Covariant1Vector(0.0)),
        right = Operators.SetValue(Geometry.Covariant1Vector(0.0)),
    )
    return curlᶠ.(Geometry.Covariant1Vector.(cos.(center_field)))
end

# Test that Julia ia able to optimize Stencil operations v1.7+
@static if @isdefined(var"@test_opt")
    @testset "Scalar Field FiniteDifferenceSpaces optimizations" begin
        device = ClimaComms.device()
        is_gpu = device isa ClimaComms.CUDADevice
        for FT in (Float64,)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(0.0),
                Geometry.ZPoint{FT}(pi);
                boundary_names = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain; nelems = 16)

            center_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

            faces = getproperty(Fields.coordinate_field(face_space), :z)
            face_values = ones(FT, face_space)
            face_velocities = Geometry.WVector.(face_values)

            centers = getproperty(Fields.coordinate_field(center_space), :z)
            center_values = ones(FT, center_space)
            center_velocities = Geometry.WVector.(center_values)

            filter(@nospecialize f) = f !== Base.mapreduce_empty

            # face space operators
            @test_opt broken = is_gpu function_filter = filter sum(
                ones(FT, face_space),
            )
            @test_opt broken = is_gpu function_filter = filter sum(sin.(faces))

            @test_opt broken = is_gpu opt_InterpolateF2C(faces)
            @test_opt broken = is_gpu opt_WeightedInterpolateF2C(
                face_values,
                faces,
            )

            @test_opt broken = is_gpu opt_LeftBiasedF2C(faces)
            @test_opt broken = is_gpu opt_RightBiasedF2C(faces)

            @test_opt broken = is_gpu opt_AdvectionF2F(face_velocities, faces)

            @test_opt broken = is_gpu opt_FluxCorrectionF2F_Extrapolate(
                center_velocities,
                faces,
            )

            @test_opt broken = is_gpu opt_GradientF2C(faces)
            @test_opt broken = is_gpu opt_DivergenceF2C(faces)

            @test_opt broken = is_gpu opt_SetBoundary_SetValue(faces)

            # center space operators
            @test_opt broken = is_gpu function_filter = filter sum(
                ones(FT, center_space),
            )
            @test_opt broken = is_gpu function_filter = filter sum(
                sin.(centers),
            )

            @test_opt broken = is_gpu opt_InterpolateC2F_SetValue(centers)
            @test_opt broken = is_gpu opt_InterpolateC2F_SetGradient(centers)
            @test_opt broken = is_gpu opt_InterpolateC2F_Extrapolate(centers)

            @test_opt broken = is_gpu opt_WeightedInterpolateC2F_SetValue(
                center_values,
                centers,
            )
            @test_opt broken = is_gpu opt_WeightedInterpolateC2F_SetGradient(
                center_values,
                centers,
            )
            @test_opt broken = is_gpu opt_WeightedInterpolateC2F_Extrapolate(
                center_values,
                centers,
            )

            @test_opt broken = is_gpu opt_LeftBiasedC2F(centers)
            @test_opt broken = is_gpu opt_RightBiasedC2F(centers)

            @test_opt broken = is_gpu opt_UpwindBiasedProductC2F_SetValue(
                face_velocities,
                centers,
            )
            @test_opt broken = is_gpu opt_UpwindBiasedProductC2F_Extrapolate(
                face_velocities,
                centers,
            )

            @test_opt broken = is_gpu opt_AdvectionC2C_SetValue(
                face_velocities,
                centers,
            )
            @test_opt broken = is_gpu opt_AdvectionC2C_Extrapolate(
                face_velocities,
                centers,
            )

            @test_opt broken = is_gpu opt_FluxCorrectionC2C_Extrapolate(
                face_velocities,
                centers,
            )

            @test_opt broken = is_gpu opt_GradientC2F_SetValue(centers)
            @test_opt broken = is_gpu opt_GradientC2F_SetGradient(centers)

            @test_opt broken = is_gpu opt_DivergenceC2F_SetValue(centers)
            @test_opt broken = is_gpu opt_DivergenceC2F_SetDivergence(centers)
            @test_opt broken = is_gpu opt_CurlC2F_SetValue(centers)
        end
    end
end
