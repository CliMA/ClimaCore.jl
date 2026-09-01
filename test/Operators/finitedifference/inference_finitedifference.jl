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

function opt_BottomBiasedF2C(face_field)
    LB = Operators.BottomBiasedF2C(left = Operators.SetValue(0.0))
    return LB.(identity.(face_field))
end

function opt_TopBiasedF2C(face_field)
    RB = Operators.TopBiasedF2C(right = Operators.SetValue(0.0))
    return RB.(identity.(face_field))
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

function opt_WeightedInterpolateC2F_Extrapolate(weights, center_field)
    WI = Operators.WeightedInterpolateC2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    return identity.(WI.(weights, center_field))
end

function opt_BottomBiasedC2F(center_field)
    LB = Operators.BottomBiasedC2F(left = Operators.SetValue(0.0))
    return LB.(identity.(center_field))
end

function opt_TopBiasedC2F(center_field)
    RB = Operators.TopBiasedC2F(right = Operators.SetValue(0.0))
    return RB.(identity.(center_field))
end

# The boundary conditions here must be named after the space's boundaries
# (left/right below), so that the named-boundary-condition path is what gets
# analyzed; a (bottom, top) pair of Extrapolate{0}s is the default, which
# short-circuits the name lookup.
function opt_UpwindBiasedProductC2F_Extrapolate(face_vel, center_field)
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.Extrapolate(0),
        right = Operators.Extrapolate(0),
    )
    return UB.(face_vel, identity.(center_field))
end

function opt_Upwind3rdOrderBiasedProductC2F_mixed(face_vel, center_field)
    UB = Operators.Upwind3rdOrderBiasedProductC2F(
        left = Operators.Extrapolate(0),
        right = Operators.Extrapolate(2),
    )
    return UB.(face_vel, identity.(center_field))
end

function opt_FCTBorisBook(face_flux, center_field)
    op = Operators.FCTBorisBook(
        left = Operators.Extrapolate(1),
        right = Operators.Extrapolate(1),
    )
    return op.(face_flux, identity.(center_field))
end

function opt_FCTZalesak(face_flux, center_field, center_field_td)
    op = Operators.FCTZalesak(
        left = Operators.Extrapolate(1),
        right = Operators.Extrapolate(1),
    )
    return op.(face_flux, tuple.(center_field, center_field_td))
end

function opt_TVDLimitedFluxC2F(face_flux, center_field, face_ct3)
    op = Operators.TVDLimitedFluxC2F(
        left = Operators.Extrapolate(1),
        right = Operators.Extrapolate(1),
        method = Operators.MinModLimiter(),
    )
    return op.(face_flux, identity.(center_field), face_ct3)
end

function opt_LinVanLeerC2F(face_vel, center_field, dt)
    op = Operators.LinVanLeerC2F(
        left = Operators.Extrapolate(1),
        right = Operators.Extrapolate(1),
        constraint = Operators.MonotoneLocalExtrema(),
    )
    return op.(face_vel, identity.(center_field), dt)
end


function opt_GradientC2F_SetGradient(center_field)
    ∇ᶠ = Operators.GradientC2F(
        left = Operators.SetGradient(Geometry.WVector(0.0)),
        right = Operators.SetGradient(Geometry.WVector(0.0)),
    )
    return Geometry.WVector.(∇ᶠ.(cos.(center_field)))
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
        device = ClimaComms.device()
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

            # Also ignore the runtime dispatch in Threads.threading_run as of Julia
            # 1.10 (UnionAll construction in typejoin, reached through the error
            # message machinery of sprint), used by parallelize_over
            filter(@nospecialize f) =
                f !== Base.mapreduce_empty && f !== Core.UnionAll

            # Face space operators
            @test_opt function_filter = filter sum(ones(FT, face_space))
            @test_opt function_filter = filter sum(sin.(faces))

            @test_opt opt_InterpolateF2C(faces)
            @test_opt opt_WeightedInterpolateF2C(face_values, faces)

            @test_opt opt_BottomBiasedF2C(faces)
            @test_opt opt_TopBiasedF2C(faces)

            @test_opt opt_GradientF2C(faces)
            @test_opt opt_DivergenceF2C(faces)

            @test_opt opt_SetBoundary_SetValue(faces)

            # Center space operators
            @test_opt function_filter = filter sum(ones(FT, center_space))
            @test_opt function_filter = filter sum(sin.(centers))

            @test_opt opt_InterpolateC2F_SetValue(centers)
            @test_opt opt_InterpolateC2F_Extrapolate(centers)

            @test_opt opt_WeightedInterpolateC2F_SetValue(
                center_values,
                centers,
            )
            @test_opt opt_WeightedInterpolateC2F_Extrapolate(
                center_values,
                centers,
            )

            @test_opt opt_BottomBiasedC2F(centers)
            @test_opt opt_TopBiasedC2F(centers)

            @test_opt opt_UpwindBiasedProductC2F_Extrapolate(
                face_velocities,
                centers,
            )

            # The reworked advection operators: matrix-rewritten (upwind) and
            # pointwise (FCT/TVD/van Leer), including the ghost-point
            # extrapolations and, for FCTZalesak, the tuple-valued advected
            # field and the neighboring-face velocities
            face_ct3 = Geometry.Contravariant3Vector.(face_values)
            @test_opt opt_Upwind3rdOrderBiasedProductC2F_mixed(
                face_velocities,
                centers,
            )
            @test_opt opt_FCTBorisBook(face_ct3, centers)
            @test_opt opt_FCTZalesak(face_ct3, centers, centers)
            @test_opt opt_TVDLimitedFluxC2F(face_ct3, centers, face_ct3)
            @test_opt opt_LinVanLeerC2F(face_velocities, centers, FT(0.1))

            @test_opt opt_GradientC2F_SetGradient(centers)

            @test_opt opt_DivergenceC2F_SetDivergence(centers)
        end
    end
end
