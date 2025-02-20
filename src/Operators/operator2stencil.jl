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

strip_space(op::Operator2Stencil, parent_space) =
    Operator2Stencil(strip_space(op.op, parent_space))

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

function return_eltype(op::Operator2Stencil, args::Vararg{Any, M}) where {M}
    lbw, ubw = stencil_interior_width(op.op, args...)[1]
    N = ubw - lbw + 1
    return StencilCoefs{lbw, ubw, NTuple{N, return_eltype(op.op, args...)}}
end

return_space(op::Operator2Stencil, spaces::Vararg{Any, N}) where {N} =
    return_space(op.op, spaces...)

stencil_interior_width(op::Operator2Stencil, args::Vararg{Any, N}) where {N} =
    stencil_interior_width(op.op, args...)

left_interior_idx(
    space::AbstractSpace,
    op::Operator2Stencil,
    bc::AbstractBoundaryCondition,
    args::Vararg{Any, N},
) where {N} = left_interior_idx(space, op.op, bc, args...)
right_interior_idx(
    space::AbstractSpace,
    op::Operator2Stencil,
    bc::AbstractBoundaryCondition,
    args::Vararg{Any, N},
) where {N} = right_interior_idx(space, op.op, bc, args...)

# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{InterpolateF2C, InterpolateC2F}},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ = RecursiveApply.rdiv(getidx(space, arg, loc, idx - half, hidx), 2)
    val⁺ = RecursiveApply.rdiv(getidx(space, arg, loc, idx + half, hidx), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁺ = getidx(space, arg, loc, idx + half, hidx)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ = getidx(space, arg, loc, idx - half, hidx)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{LeftBiasedF2C, LeftBiasedC2F}},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ = getidx(space, arg, loc, idx - half, hidx)
    return StencilCoefs{-half, -half}((val⁻,))
end
stencil_left_boundary(
    ::Operator2Stencil{<:Union{LeftBiasedF2C, LeftBiasedC2F}},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
) = StencilCoefs{-half, -half}((zero(eltype(arg)),))


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{RightBiasedF2C, RightBiasedC2F}},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁺ = getidx(space, arg, loc, idx + half, hidx)
    return StencilCoefs{half, half}((val⁺,))
end
stencil_right_boundary(
    ::Operator2Stencil{<:Union{RightBiasedF2C, RightBiasedC2F}},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
) = StencilCoefs{half, half}((zero(eltype(arg)),))


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:AdvectionC2C},
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    val⁻ = RecursiveApply.rdiv(w³⁻ ⊠ (θ ⊟ θ⁻), 2)
    val⁺ = RecursiveApply.rdiv(w³⁺ ⊠ (θ⁺ ⊟ θ), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ⁻ = getidx(space, bc.val, loc, nothing, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    val⁻ = w³⁻ ⊠ (θ ⊟ θ⁻)
    val⁺ = RecursiveApply.rdiv(w³⁺ ⊠ (θ⁺ ⊟ θ), 2)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, bc.val, loc, nothing, hidx)
    val⁻ = RecursiveApply.rdiv(w³⁻ ⊠ (θ ⊟ θ⁻), 2)
    val⁺ = w³⁺ ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    val⁺ = w³⁺ ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:AdvectionC2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    val⁻ = w³⁻ ⊠ (θ ⊟ θ⁻)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    val⁻ = ⊟(abs(w³⁻) ⊠ (θ ⊟ θ⁻))
    val⁺ = abs(w³⁺) ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    val⁺ = abs(w³⁺) ⊠ (θ⁺ ⊟ θ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:Union{FluxCorrectionC2C, FluxCorrectionF2F}},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    val⁻ = ⊟(abs(w³⁻) ⊠ (θ ⊟ θ⁻))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{GradientF2C, GradientC2F}},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ =
        Geometry.Covariant3Vector(-1) ⊗
        getidx(space, arg, loc, idx - half, hidx)
    val⁺ =
        Geometry.Covariant3Vector(1) ⊗ getidx(space, arg, loc, idx + half, hidx)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁺ =
        Geometry.Covariant3Vector(1) ⊗ getidx(space, arg, loc, idx + half, hidx)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ =
        Geometry.Covariant3Vector(-1) ⊗
        getidx(space, arg, loc, idx - half, hidx)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
stencil_left_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
) = extrapolation_increases_bandwidth_error(GradientF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
) = extrapolation_increases_bandwidth_error(GradientF2C)
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁺ =
        Geometry.Covariant3Vector(2) ⊗ getidx(space, arg, loc, idx + half, hidx)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    val⁻ =
        Geometry.Covariant3Vector(-2) ⊗
        getidx(space, arg, loc, idx - half, hidx)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.gradient_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.gradient_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end


# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:Union{DivergenceF2C, DivergenceC2F}},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁻ = Ju³⁻ ⊠ (-invJ)
    val⁺ = Ju³⁺ ⊠ invJ
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁺ = Ju³⁺ ⊠ invJ
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁻ = Ju³⁻ ⊠ (-invJ)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
) = extrapolation_increases_bandwidth_error(DivergenceF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
) = extrapolation_increases_bandwidth_error(DivergenceF2C)
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Ju³⁺ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁺ = Ju³⁺ ⊠ (2 * invJ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Ju³⁻ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁻ = Ju³⁻ ⊠ (-2 * invJ)
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end

# Evaluate fd3_curl(u, zero(u), J).
fd3_curl⁺(u::Geometry.Covariant1Vector, invJ) =
    Geometry.Contravariant2Vector(u.u₁ * invJ)
fd3_curl⁺(u::Geometry.Covariant2Vector, invJ) =
    Geometry.Contravariant1Vector(-u.u₂ * invJ)
fd3_curl⁺(::Geometry.Covariant3Vector, invJ) =
    Geometry.Contravariant3Vector(zero(eltype(invJ)))
fd3_curl⁺(u::Geometry.Covariant12Vector, invJ) =
    Geometry.Contravariant12Vector(-u.u₂ * invJ, u.u₁ * invJ)

# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_interior(
    ::Operator2Stencil{<:CurlC2F},
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u₋ = getidx(space, arg, loc, idx - half, hidx)
    u₊ = getidx(space, arg, loc, idx + half, hidx)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁻ = ⊟(fd3_curl⁺(u₋, invJ))
    val⁺ = fd3_curl⁺(u₊, invJ)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_left_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u₊ = getidx(space, arg, loc, idx + half, hidx)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁺ = fd3_curl⁺(u₊, 2 * invJ)
    return StencilCoefs{-half, half}((zero(val⁺), val⁺))
end
# TODO: find out why using Base.@propagate_inbounds blows up compilation time
function stencil_right_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u₋ = getidx(space, arg, loc, idx - half, hidx)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    val⁻ = ⊟(fd3_curl⁺(u₋, invJ))
    return StencilCoefs{-half, half}((val⁻, zero(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), zero(T)))
end
