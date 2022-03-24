# If op is a linear Operator, then there is some matrix of coefficients C such
# that, for any Field a, op.(a)[i] = ∑_j C[i, j] * a[j]. Operator2Stencil(op)
# is an operator such that Operator2Stencil(op).(a) is a Field of StencilCoefs;
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
#   ComposeStencils().(
#     stencil_op2.(f2′.(op1.(f1.(a)))),
#     stencil_op1.(f1′.(a))
#   )[i, j].
# This means that
# ∂b/∂a =
#   ComposeStencils().(stencil_op2.(f2′.(op1.(f1.(a)))), stencil_op1.(f1′.(a))).

# The stencils constructed by Operator2Stencil do not account for nonzero values
# added by boundary conditions. To account for these values, stencils would need
# to encode affine transformations, rather than linear ones. That is, they would
# have to encode Operators of the form op.(a)[i] = ∑_j C[i, j] * a[j] + C′[i].

# Similarly, Operator2Stencil does not support any WeightedInterpolationOperator
# because such an Operator is not linear with respect to its first argument.

# Operator2Stencil currently supports all Operators that are F2C or C2F with
# respect to their first arguments. Other operators can be added if needed.

struct Operator2Stencil{O <: FiniteDifferenceOperator} <:
       FiniteDifferenceOperator
    op::O
end

extrapolation_increases_bandwidth_error(op_type::Type) = throw(
    ArgumentError(
        "Operator2Stencil currently only supports operators whose stencils \
        have identical bandwidths in the interior and on the boundary of the \
        domain. So, it cannot be applied to the `$op_type` operator with an \
        `Extrapolate` boundary condition, since the corresponding stencil's \
        bandwidth on the boundary is larger than its bandwidth in the interior \
        by 1.",
    ),
)

has_boundary(op::Operator2Stencil, bw::LeftBoundaryWindow{name}) where {name} =
    has_boundary(op.op, bw)
has_boundary(op::Operator2Stencil, bw::RightBoundaryWindow{name}) where {name} =
    has_boundary(op.op, bw)

get_boundary(op::Operator2Stencil, bw::LeftBoundaryWindow{name}) where {name} =
    get_boundary(op.op, bw)
get_boundary(op::Operator2Stencil, bw::RightBoundaryWindow{name}) where {name} =
    get_boundary(op.op, bw)

function return_eltype(op::Operator2Stencil, args...)
    lbw, ubw = stencil_interior_width(op.op, args...)[1]
    N = ubw - lbw + 1
    return StencilCoefs{lbw, ubw, NTuple{N, return_eltype(op.op, args...)}}
end

return_space(op::Operator2Stencil, spaces...) = return_space(op.op, spaces...)

stencil_interior_width(op::Operator2Stencil, args...) =
    stencil_interior_width(op.op, args...)

boundary_width(op::Operator2Stencil, bc::BoundaryCondition, args...) =
    boundary_width(op.op, bc, args...)


function stencil_interior(
    ::Operator2Stencil{<:Union{InterpolateF2C, InterpolateC2F}},
    loc,
    idx,
    arg,
)
    val⁻ = RecursiveApply.rdiv(getidx(arg, loc, idx - half), 2)
    val⁺ = RecursiveApply.rdiv(getidx(arg, loc, idx + half), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    arg,
)
    val⁺ = getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    arg,
)
    val⁻ = getidx(arg, loc, idx - half)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


function stencil_interior(
    ::Operator2Stencil{<:Union{LeftBiasedF2C, LeftBiasedC2F}},
    loc,
    idx,
    arg,
)
    val⁻ = getidx(arg, loc, idx - half)
    return StencilCoefs{-half, -half}((val⁻,))
end
stencil_left_boundary(
    ::Operator2Stencil{<:Union{LeftBiasedF2C, LeftBiasedC2F}},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{-half, -half}((zero(eltype(arg)),))


function stencil_interior(
    ::Operator2Stencil{<:Union{RightBiasedF2C, RightBiasedC2F}},
    loc,
    idx,
    arg,
)
    val⁺ = getidx(arg, loc, idx + half)
    return StencilCoefs{half, half}((val⁺,))
end
stencil_right_boundary(
    ::Operator2Stencil{<:Union{RightBiasedF2C, RightBiasedC2F}},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{half, half}((zero(eltype(arg)),))


function stencil_interior(
    ::Operator2Stencil{<:AdvectionC2C},
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ⁻ = getidx(arg, loc, idx - 1)
    θ = getidx(arg, loc, idx)
    θ⁺ = getidx(arg, loc, idx + 1)
    val⁻ = RecursiveApply.rdiv(w³⁻ ⊠ (θ ⊟ θ⁻), 2)
    val⁺ = RecursiveApply.rdiv(w³⁺ ⊠ (θ⁺ ⊟ θ), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::SetValue,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ⁻ = bc.val
    θ = getidx(arg, loc, idx)
    θ⁺ = getidx(arg, loc, idx + 1)
    val⁻ = w³⁻ ⊠ (θ ⊟ θ⁻)
    val⁺ = RecursiveApply.rdiv(w³⁺ ⊠ (θ⁺ ⊟ θ), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::SetValue,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ⁻ = getidx(arg, loc, idx - 1)
    θ = getidx(arg, loc, idx)
    θ⁺ = bc.val
    val⁻ = RecursiveApply.rdiv(w³⁻ ⊠ (θ ⊟ θ⁻), 2)
    val⁺ = w³⁺ ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ = getidx(arg, loc, idx)
    θ⁺ = getidx(arg, loc, idx + 1)
    val⁺ = w³⁺ ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    θ = getidx(arg, loc, idx)
    θ⁻ = getidx(arg, loc, idx - 1)
    val⁻ = w³⁻ ⊠ (θ ⊟ θ⁻)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


function stencil_interior(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ⁻ = getidx(arg, loc, idx - 1)
    θ = getidx(arg, loc, idx)
    θ⁺ = getidx(arg, loc, idx + 1)
    val⁻ = ⊟(abs(w³⁻) ⊠ (θ ⊟ θ⁻))
    val⁺ = abs(w³⁺) ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    θ = getidx(arg, loc, idx)
    θ⁺ = getidx(arg, loc, idx + 1)
    val⁺ = abs(w³⁺) ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
)

    space = axes(arg)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    θ⁻ = getidx(arg, loc, idx - 1)
    θ = getidx(arg, loc, idx)
    val⁻ = ⊟(abs(w³⁻) ⊠ (θ ⊟ θ⁻))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


function stencil_interior(
    ::Operator2Stencil{<:Union{GradientF2C, GradientC2F}},
    loc,
    idx,
    arg,
)
    val⁻ = Geometry.Covariant3Vector(-1) ⊗ getidx(arg, loc, idx - half)
    val⁺ = Geometry.Covariant3Vector(1) ⊗ getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    val⁺ = Geometry.Covariant3Vector(1) ⊗ getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    val⁻ = Geometry.Covariant3Vector(-1) ⊗ getidx(arg, loc, idx - half)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
stencil_left_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = extrapolation_increases_bandwidth_error(GradientF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = extrapolation_increases_bandwidth_error(GradientF2C)
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    val⁺ = Geometry.Covariant3Vector(2) ⊗ getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    val⁻ = Geometry.Covariant3Vector(-2) ⊗ getidx(arg, loc, idx - half)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    T = Geometry.gradient_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    T = Geometry.gradient_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end


function stencil_interior(
    ::Operator2Stencil{<:Union{DivergenceF2C, DivergenceC2F}},
    loc,
    idx,
    arg,
)
    space = axes(arg)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    J = Geometry.LocalGeometry(space, idx).J
    val⁻ = ⊟(RecursiveApply.rdiv(Ju³⁻, J))
    val⁺ = RecursiveApply.rdiv(Ju³⁺, J)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    J = Geometry.LocalGeometry(space, idx).J
    val⁺ = RecursiveApply.rdiv(Ju³⁺, J)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    J = Geometry.LocalGeometry(space, idx).J
    val⁻ = ⊟(RecursiveApply.rdiv(Ju³⁻, J))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = extrapolation_increases_bandwidth_error(DivergenceF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = extrapolation_increases_bandwidth_error(DivergenceF2C)
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    J = Geometry.LocalGeometry(space, idx).J
    val⁺ = RecursiveApply.rdiv(Ju³⁺, J / 2)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    J = Geometry.LocalGeometry(space, idx).J
    val⁻ = ⊟(RecursiveApply.rdiv(Ju³⁻, J / 2))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    idx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    idx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end

# Evaluate fd3_curl(u, zero(u), J).
fd3_curl⁺(u::Geometry.Covariant1Vector, J) =
    Geometry.Contravariant2Vector(u.u₁ / J)
fd3_curl⁺(u::Geometry.Covariant2Vector, J) =
    Geometry.Contravariant1Vector(-u.u₂ / J)
fd3_curl⁺(::Geometry.Covariant3Vector, J) =
    Geometry.Contravariant3Vector(zero(eltype(J)))
fd3_curl⁺(u::Geometry.Covariant12Vector, J) =
    Geometry.Contravariant12Vector(-u.u₂ / J, u.u₁ / J)

function stencil_interior(::Operator2Stencil{<:CurlC2F}, loc, idx, arg)
    space = axes(arg)
    u₋ = getidx(arg, loc, idx - half)
    u₊ = getidx(arg, loc, idx + half)
    J = Geometry.LocalGeometry(space, idx).J
    val⁻ = ⊟(fd3_curl⁺(u₋, J))
    val⁺ = fd3_curl⁺(u₊, J)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    u₊ = getidx(arg, loc, idx + half)
    J = Geometry.LocalGeometry(space, idx).J
    val⁺ = fd3_curl⁺(u₊, J / 2)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    u₋ = getidx(arg, loc, idx - half)
    J = Geometry.LocalGeometry(space, idx).J
    val⁻ = ⊟(fd3_curl⁺(u₋, J))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    idx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    idx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
