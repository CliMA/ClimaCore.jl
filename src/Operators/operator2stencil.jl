# Note: The stencils constructed by Operator2Stencil are linear transformations,
# rather than affine transformations. Therefore, they do not account for constant
# values added by boundary conditions.

# Note: This file only implements Operator2Stencil for F2C and C2F operators.
# Other operators can be added if needed.

struct Operator2Stencil{O <: FiniteDifferenceOperator} <:
       FiniteDifferenceOperator
    op::O
end

bidiagonal_extrapolate_error(op_type::Type) = throw(
    ArgumentError(
        "Operator2Stencil currently only supports operators whose stencils \
        have identical bandwidths in the interior and on the boundary of the \
        domain. So, it cannot be used with the `$(op_type.name.wrapper)` \
        operator with an `Extrapolate` boundary condition, since the \
        corresponding stencil has bandwidths of (-half, half) in the interior \
        and either (-half, 1+half), (-(1+half), half), or (-(1+half), 1+half) \
        on the boundary.",
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
    lbw, ubw = stencil_interior_width(op.op, args...)[end]
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
    return StencilCoefs{-half, half}((nan(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), nan(T)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    arg,
)
    val⁺ = getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:InterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    arg,
)
    val⁻ = getidx(arg, loc, idx - half)
    return StencilCoefs{-half, half}((val⁻, nan(val⁻)))
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
    ::Operator2Stencil{<:LeftBiasedF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{-half, -half}((zero(eltype(arg)),))
stencil_left_boundary(
    ::Operator2Stencil{<:LeftBiasedC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{-half, -half}((nan(eltype(arg)),))


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
    ::Operator2Stencil{<:RightBiasedF2C},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{half, half}((zero(eltype(arg)),))
stencil_right_boundary(
    ::Operator2Stencil{<:RightBiasedC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
) = StencilCoefs{half, half}((nan(eltype(arg)),))


function stencil_interior(
    ::Operator2Stencil{<:Union{WeightedInterpolateF2C, WeightedInterpolateC2F}},
    loc,
    idx,
    weight,
    arg,
)
    w⁻ = getidx(weight, loc, idx - half)
    w⁺ = getidx(weight, loc, idx + half)
    a⁻ = getidx(arg, loc, idx - half)
    a⁺ = getidx(arg, loc, idx + half)
    denominator = 2 ⊠ (w⁻ ⊞ w⁺)
    val⁻ = RecursiveApply.rdiv(w⁻ ⊠ a⁻, denominator)
    val⁺ = RecursiveApply.rdiv(w⁺ ⊠ a⁺, denominator)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:WeightedInterpolateC2F},
    bc::SetValue,
    loc,
    idx,
    weight,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((nan(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:WeightedInterpolateC2F},
    bc::SetValue,
    loc,
    idx,
    weight,
    arg,
)
    T = eltype(arg)
    return StencilCoefs{-half, half}((zero(T), nan(T)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:WeightedInterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    weight,
    arg,
)
    val⁺ = getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:WeightedInterpolateC2F},
    bc::Union{SetGradient, Extrapolate},
    loc,
    idx,
    weight,
    arg,
)
    val⁻ = getidx(arg, loc, idx - half)
    return StencilCoefs{-half, half}((val⁻, nan(val⁻)))
end


function stencil_interior(
    ::Operator2Stencil{<:UpwindBiasedProductC2F},
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁻ = getidx(arg, loc, idx - half)
    a⁺ = getidx(arg, loc, idx + half)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    product⁻ = RecursiveApply.rdiv((vᶠ ⊞ RecursiveApply.rmap(abs, vᶠ)) ⊠ a⁻, 2)
    product⁺ = RecursiveApply.rdiv((vᶠ ⊟ RecursiveApply.rmap(abs, vᶠ)) ⊠ a⁺, 2)
    val⁻ = Geometry.Contravariant3Vector(product⁻)
    val⁺ = Geometry.Contravariant3Vector(product⁺)
    return StencilCoefs{-half, half}((val⁻, val⁺))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:UpwindBiasedProductC2F},
    bc::SetValue,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁺ = getidx(arg, loc, idx + half)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    product⁺ = RecursiveApply.rdiv((vᶠ ⊟ RecursiveApply.rmap(abs, vᶠ)) ⊠ a⁺, 2)
    val⁺ = Geometry.Contravariant3Vector(product⁺)
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:UpwindBiasedProductC2F},
    bc::SetValue,
    loc,
    idx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁻ = getidx(arg, loc, idx - half)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    product⁻ = RecursiveApply.rdiv((vᶠ ⊞ RecursiveApply.rmap(abs, vᶠ)) ⊠ a⁻, 2)
    val⁻ = Geometry.Contravariant3Vector(product⁻)
    return StencilCoefs{-half, half}((val⁻, nan(val⁻)))
end
stencil_left_boundary(
    ::Operator2Stencil{<:UpwindBiasedProductC2F},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
) = bidiagonal_extrapolate_error(UpwindBiasedProductC2F)
stencil_right_boundary(
    ::Operator2Stencil{<:UpwindBiasedProductC2F},
    bc::Extrapolate,
    loc,
    idx,
    velocity,
    arg,
) = bidiagonal_extrapolate_error(UpwindBiasedProductC2F)


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
) = bidiagonal_extrapolate_error(GradientF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:GradientF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = bidiagonal_extrapolate_error(GradientF2C)
function stencil_left_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetValue,
    loc,
    idx,
    arg,
)
    val⁺ = Geometry.Covariant3Vector(2) ⊗ getidx(arg, loc, idx + half)
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
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
    return StencilCoefs{-half, half}((nan(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:GradientC2F},
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    T = Geometry.gradient_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), nan(T)))
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
) = bidiagonal_extrapolate_error(DivergenceF2C)
stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceF2C},
    bc::Extrapolate,
    loc,
    idx,
    arg,
) = bidiagonal_extrapolate_error(DivergenceF2C)
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
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
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
    return StencilCoefs{-half, half}((val⁻, nan(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    idx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((nan(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:DivergenceC2F},
    bc::SetDivergence,
    loc,
    idx,
    arg,
)
    T = Geometry.divergence_result_type(eltype(arg))
    return StencilCoefs{-half, half}((zero(T), nan(T)))
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
    return StencilCoefs{-half, half}((nan(val⁺), val⁺))
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
    return StencilCoefs{-half, half}((val⁻, nan(val⁻)))
end
function stencil_left_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    idx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((nan(T), zero(T)))
end
function stencil_right_boundary(
    ::Operator2Stencil{<:CurlC2F},
    bc::SetCurl,
    loc,
    idx,
    arg,
)
    T = Geometry.curl_result_type(Val((3,)), eltype(arg))
    return StencilCoefs{-half, half}((zero(T), nan(T)))
end
