using Test
using StaticArrays, IntervalSets, LinearAlgebra
using Random: seed!

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

# If op is a linear Operator, then, for any Field a, there is some matrix of
# coefficients C such that op.(a)[i] = ∑_j C[i, j] * a[j]. Operator2Stencil(op)
# is an operator for which Operator2Stencil(op).(a) is a Field of StencilCoefs;
# when it is interpreted as a matrix, this Field has the property that
# Operator2Stencil(op).(a)[i, j] = C[i, j] * a[j]. More specifically,
# Operator2Stencil(op).(a)[i] is a StencilCoefs object that stores the tuple
# (C[i, i+lbw] a[i+lbw], C[i, i+lbw+1] a[i+lbw+1], ..., C[i, i+ubw] a[i+ubw]),
# where (lbw, ubw) are the bandwidths of op (that is, the bandwidths of C).

# This property can be used to find Jacobian matrices. If we let b = op.(f.(a)),
# where op is a linear Operator and f is a Function (or an object that acts like
# a Function), then b[i] = ∑_j C[i, j] * f(a[j]). If f has a derivative f′, then
# the Jacobian matrix of b with respect to a is given by
# (∂b/∂a)[i, j] =
#   ∂(b[i])/∂(a[j]) =
#   C[i, j] * f′(a[j]) =
#   Operator2Stencil(op).(f′.(a))[i, j].
# This means that ∂b/∂a = Operator2Stencil(op).(f′.(a)).

# More generally, we can have b = op2.(f2.(op1.(f1.(a)))), where op1 is either a
# single Operator or a composition of multiple Operators and Functions. If
# op1.(a)[i] = ∑_j C1[i, j] * a[j] and op2.(a)[i] = ∑_k C2[i, k] * a[k], then
# b[i] =
#   ∑_k C2[i, k] * f2(op1.(f1.(a))[k]) =
#   ∑_k C2[i, k] * f2(∑_j C1[k, j] * f1(a[j])).
# Let stencil_op1 = Operator2Stencil(op1), stencil_op2 = Operator2Stencil(op2).
# We then find that the Jacobian matrix of b with respect to a is given by
# (∂b/∂a)[i, j] =
#   ∂(b[i])/∂(a[j]) =
#   ∑_k C2[i, k] * f2′(op1.(f1.(a))[k]) * C1[k, j] * f1′(a[j]) =
#   ∑_k stencil_op2.(f2′.(op1.(f1.(a))))[i, k] * stencil_op1.(f1′.(a))[k, j] =
#   ComposeStencils().(stencil_op2.(f2′.(op1.(f1.(a)))), stencil_op1.(f1′.(a)))[i, j].
# This means that
# ∂b/∂a =
#   ComposeStencils().(stencil_op2.(f2′.(op1.(f1.(a)))), stencil_op1.(f1′.(a))).

# op.(a)[i] = ∑_j C[i, j] * a[j]                                             ==>
# op.(b .* a)[i] =
#   ∑_j C[i, j] * b[j] * a[j] =
#   ∑_j Operator2Stencil(op).(b)[i, j] * a[j] =
#   ApplyStencil().(Operator2Stencil(op).(b), a)[i]                          ==>
# op.(b .* a) = ApplyStencil().(Operator2Stencil(op).(b), a)

# Let stencil_op1 = Operator2Stencil(op1), stencil_op2 = Operator2Stencil(op2).
# op1.(a)[i] = ∑_j C1[i, j] * a[j] and op2.(a)[i] = ∑_k C2[i, k] * a[k]      ==>
# op2.(c .* op1.(b .* a))[i] =
#   ∑_k C2[i, k] * c[k] * op1.(b .* a)[k] =
#   ∑_k C2[i, k] * c[k] * (∑_j C1[k, j] * b[j] * a[j]) =
#   ∑_j (∑_k C2[i, k] * c[k] * C1[k, j] * b[j]) * a[j] =
#   ∑_j (∑_k stencil_op2.(c)[i, k] * stencil_op1.(b)[k, j]) * a[j] =
#   ∑_j ComposeStencils().(stencil_op2.(c), stencil_op1.(b))[i, j] * a[j] =
#   ApplyStencil().(ComposeStencils().(stencil_op2.(c), stencil_op1.(b)), a)[i]
#                                                                            ==>
# op2.(c .* op1.(b .* a)) =
#   ApplyStencil().(ComposeStencils().(stencil_op2.(c), stencil_op1.(b)), a)

@testset "Pointwise Stencil Construction/Composition/Application" begin
    seed!(1) # Ensure reproducibility.

    for FT in (Float32, Float64)
        radius = FT(1e7)
        zmax = FT(1e4)
        velem = helem = npoly = 4

        hdomain = Domains.SphereDomain(radius)
        hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
        htopology = Topologies.Topology2D(hmesh)
        quad = Spaces.Quadratures.GLL{npoly + 1}()
        hspace = Spaces.SpectralElementSpace2D(htopology, quad)

        vdomain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(zero(FT)),
            Geometry.ZPoint{FT}(zmax);
            boundary_tags = (:bottom, :top),
        )
        vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
        vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

        # TODO: Replace this with a space that includes topography.
        center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

        rand_scalar = (_...) -> rand(FT)
        rand_vector = (_...) -> Geometry.UVWVector(rand(FT), rand(FT), rand(FT))
        scalar_c = map(rand_scalar, Fields.coordinate_field(center_space))
        scalar_f = map(rand_scalar, Fields.coordinate_field(face_space))
        vector_c = map(rand_vector, Fields.coordinate_field(center_space))
        vector_f = map(rand_vector, Fields.coordinate_field(face_space))

        # single-argument operations
        input_agnostic_ops = (
            Operators.InterpolateF2C(),
            Operators.InterpolateC2F(
                bottom = Operators.Extrapolate(),
                top = Operators.Extrapolate(),
            ),
            Operators.LeftBiasedF2C(),
            Operators.RightBiasedF2C(),
        )
        scalar_only_ops = (
            Operators.InterpolateC2F(
                bottom = Operators.SetValue(rand_scalar()),
                top = Operators.SetValue(rand_scalar()),
            ),
            Operators.InterpolateC2F(
                bottom = Operators.SetGradient(rand_vector()),
                top = Operators.SetGradient(rand_vector()),
            ),
            Operators.LeftBiasedF2C(
                bottom = Operators.SetValue(rand_scalar()),
            ),
            Operators.LeftBiasedC2F(
                bottom = Operators.SetValue(rand_scalar()),
            ),
            Operators.RightBiasedF2C(
                top = Operators.SetValue(rand_scalar()),
            ),
            Operators.RightBiasedC2F(
                top = Operators.SetValue(rand_scalar()),
            ),
        )
        vector_only_ops = (
            Operators.InterpolateC2F(
                bottom = Operators.SetValue(rand_vector()),
                top = Operators.SetValue(rand_vector()),
            ),
            Operators.LeftBiasedF2C(
                bottom = Operators.SetValue(rand_vector()),
            ),
            Operators.LeftBiasedC2F(
                bottom = Operators.SetValue(rand_vector()),
            ),
            Operators.RightBiasedF2C(
                top = Operators.SetValue(rand_vector()),
            ),
            Operators.RightBiasedC2F(
                top = Operators.SetValue(rand_vector()),
            ),
        )
    end
end
