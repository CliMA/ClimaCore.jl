#=
This was adapted from StaticArrays' `SHermitianCompact`:
https://github.com/JuliaArrays/StaticArrays.jl/blob/609aa343a0504c3f5bae82a6236c563a0fa72681/src/SHermitianCompact.jl
=#

import LinearAlgebra
import StaticArrays: SVector, SMatrix

"""
    SimpleSymmetric{N, T, L} <: StaticMatrix{N, N, T}

A [`StaticArray`](@ref) subtype that represents a symmetric matrix. Unlike
`LinearAlgebra.Symmetric`, `SimpleSymmetric` stores only the lower triangle
of the matrix (as an `SVector`). The lower triangle is stored in column-major order.
For example, for an `SimpleSymmetric{3}`, the indices of the stored elements
can be visualized as follows:

```
┌ 1 ⋅ ⋅ ┐
| 2 4 ⋅ |
└ 3 5 6 ┘
```

Type parameters:
* `N`: matrix dimension;
* `T`: element type for lower triangle;
* `L`: length of the `SVector` storing the lower triangular elements.

Note that `L` is always the `N`th [triangular number](https://en.wikipedia.org/wiki/Triangular_number).

A `SimpleSymmetric` may be constructed either:

* from an `AbstractVector` containing the lower triangular elements; or
* from a `Tuple` containing both upper and lower triangular elements in column major order; or
* from another `StaticMatrix`.

For the latter two cases, only the lower triangular elements are used; the upper triangular
elements are ignored.
"""
struct SimpleSymmetric{N, T, L} <: StaticMatrix{N, N, T}
    lowertriangle::SVector{L, T}

    @inline function SimpleSymmetric{N, T, L}(
        lowertriangle::SVector{L},
    ) where {N, T, L}
        _check_simple_symmetric_parameters(Val(N), Val(L))
        new{N, T, L}(lowertriangle)
    end
end
StaticArrays.check_parameters(
    ::Type{SimpleSymmetric{N, T, L}},
) where {N, T, L} = _check_simple_symmetric_parameters(Val(N), Val(L))

triangular_nonzeros(::SMatrix{N}) where {N} = Int(N * (N + 1) / 2)
triangular_nonzeros(::Type{<:SMatrix{N}}) where {N} = Int(N * (N + 1) / 2)
tail_params(::Type{S}) where {N, T, S <: SMatrix{N, N, T}} =
    (T, S, N, triangular_nonzeros(S))

@generated function SimpleSymmetric(A::S) where {S <: SMatrix}
    N = size(S, 1)
    L = triangular_nonzeros(S)
    _check_simple_symmetric_parameters(Val(N), Val(L))
    expr = Vector{Expr}(undef, L)
    T = eltype(S)
    i = 0
    for col in 1:N, row in 1:N
        if col ≥ row
            expr[i += 1] = :(A[$row, $col])
        end
    end
    quote
        Base.@_inline_meta
        @inbounds return SimpleSymmetric{$N, $T, $L}(
            SVector{$L, $T}(tuple($(expr...))),
        )
    end
end

@inline function _check_simple_symmetric_parameters(
    ::Val{N},
    ::Val{L},
) where {N, L}
    if 2 * L !== N * (N + 1)
        throw(
            ArgumentError(
                "Size mismatch in SimpleSymmetric parameters. Got dimension $N and length $L.",
            ),
        )
    end
end

triangularnumber(N::Int) = div(N * (N + 1), 2)
@generated function triangularroot(::Val{L}) where {L}
    return div(isqrt(8 * L + 1) - 1, 2) # from quadratic formula
end

lowertriangletype(::Type{SimpleSymmetric{N, T, L}}) where {N, T, L} =
    SVector{L, T}
lowertriangletype(::Type{SimpleSymmetric{N, T}}) where {N, T} =
    SVector{triangularnumber(N), T}
lowertriangletype(::Type{SimpleSymmetric{N}}) where {N} =
    SVector{triangularnumber(N)}

@inline SimpleSymmetric{N, T}(lowertriangle::SVector{L}) where {N, T, L} =
    SimpleSymmetric{N, T, L}(lowertriangle)
@inline SimpleSymmetric{N}(lowertriangle::SVector{L, T}) where {N, T, L} =
    SimpleSymmetric{N, T, L}(lowertriangle)

@inline function SimpleSymmetric(lowertriangle::SVector{L, T}) where {T, L}
    N = triangularroot(Val(L))
    SimpleSymmetric{N, T, L}(lowertriangle)
end

@generated function SimpleSymmetric{N, T, L}(a::Tuple) where {N, T, L}
    _check_simple_symmetric_parameters(Val(N), Val(L))
    expr = Vector{Expr}(undef, L)
    i = 0
    for col in 1:N, row in col:N
        index = N * (col - 1) + row
        expr[i += 1] = :(a[$index])
    end
    quote
        Base.@_inline_meta
        @inbounds return SimpleSymmetric{N, T, L}(
            SVector{L, T}(tuple($(expr...))),
        )
    end
end

@inline function SimpleSymmetric{N, T}(a::Tuple) where {N, T}
    L = triangularnumber(N)
    SimpleSymmetric{N, T, L}(a)
end

@inline (::Type{SSC})(a::SimpleSymmetric) where {SSC <: SimpleSymmetric} =
    SSC(a.lowertriangle)

@inline (::Type{SSC})(a::AbstractVector) where {SSC <: SimpleSymmetric} =
    SSC(convert(lowertriangletype(SSC), a))

# disambiguation
@inline (::Type{SSC})(
    a::StaticArray{<:Tuple, <:Any, 1},
) where {SSC <: SimpleSymmetric} = SSC(convert(SVector, a))

@generated function _compact_indices(::Val{N}) where {N}
    # Returns a Tuple{Pair{Int, Bool}} I such that for linear index i,
    # * I[i][1] is the index into the lowertriangle field of a SimpleSymmetric{N};
    # * I[i][2] is true iff i is an index into the lower triangle of an N × N matrix.
    indexmat = Matrix{Int}(undef, N, N)
    i = 0
    for col in 1:N, row in 1:N
        indexmat[row, col] = if row >= col
            i += 1
        else
            indexmat[col, row]
        end
    end
    quote
        Base.@_inline_meta
        return $(tuple(indexmat...))
    end
end

Base.@propagate_inbounds function Base.getindex(
    a::SimpleSymmetric{N},
    i::Int,
) where {N}
    I = _compact_indices(Val(N))
    j = I[i]
    @inbounds value = a.lowertriangle[j]
    return value
end

Base.@propagate_inbounds function Base.setindex(
    a::SimpleSymmetric{N, T, L},
    x,
    i::Int,
) where {N, T, L}
    I = _compact_indices(Val(N))
    j = I[i]
    value = x
    return SimpleSymmetric{N}(setindex(a.lowertriangle, value, j))
end

# needed because it is used in convert.jl and the generic fallback is slow
@generated function Base.Tuple(a::SimpleSymmetric{N}) where {N}
    exprs = [:(a[$i]) for i in 1:(N^2)]
    quote
        Base.@_inline_meta
        tuple($(exprs...))
    end
end

LinearAlgebra.issymmetric(a::SimpleSymmetric) = true

# TODO: factorize?

@inline Base.:(==)(a::SimpleSymmetric, b::SimpleSymmetric) =
    a.lowertriangle == b.lowertriangle

@inline function Base.map(f, a1::SimpleSymmetric, as::AbstractArray...)
    _map(f, a1, as...)
end

@generated function _map(f, a::SimpleSymmetric...)
    S = Size(a[1])
    N = S[1]
    L = triangularnumber(N)
    exprs = Vector{Expr}(undef, L)
    for i in 1:L
        tmp = [:(a[$j].lowertriangle[$i]) for j in 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    return quote
        Base.@_inline_meta
        same_size(a...)
        @inbounds return SimpleSymmetric(SVector(tuple($(exprs...))))
    end
end

@inline Base.:*(a::Number, b::SimpleSymmetric) =
    SimpleSymmetric(a * b.lowertriangle)
@inline Base.:*(a::SimpleSymmetric, b::Number) =
    SimpleSymmetric(a.lowertriangle * b)

@inline Base.:/(a::SimpleSymmetric, b::Number) =
    SimpleSymmetric(a.lowertriangle / b)
@inline Base.:\(a::Number, b::SimpleSymmetric) =
    SimpleSymmetric(a \ b.lowertriangle)

@generated function _plus_uniform(
    ::Size{S},
    a::SimpleSymmetric{N, T, L},
    λ,
) where {S, N, T, L}
    @assert S[1] == N
    @assert S[2] == N
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col in 1:N, row in col:N
        i += 1
        exprs[i] =
            row == col ? :(a.lowertriangle[$i] + λ) : :(a.lowertriangle[$i])
    end
    return quote
        Base.@_inline_meta
        R = promote_type(eltype(a), typeof(λ))
        SimpleSymmetric{N, R, L}(SVector{L, R}(tuple($(exprs...))))
    end
end

LinearAlgebra.transpose(a::SimpleSymmetric) = a

@generated function _one(
    ::Size{S},
    ::Type{SSC},
) where {S, SSC <: SimpleSymmetric}
    N = S[1]
    L = triangularnumber(N)
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col in 1:N, row in col:N
        exprs[i += 1] = row == col ? :(one($T)) : :(zero($T))
    end
    quote
        Base.@_inline_meta
        return SimpleSymmetric(SVector(tuple($(exprs...))))
    end
end

@inline _scalar_matrix(
    s::Size{S},
    t::Type{SSC},
) where {S, SSC <: SimpleSymmetric} = _one(s, t)

# _fill covers fill, zeros, and ones:
@generated function _fill(
    val,
    ::Size{s},
    ::Type{SSC},
) where {s, SSC <: SimpleSymmetric}
    N = s[1]
    L = triangularnumber(N)
    v = [:val for i in 1:L]
    return quote
        Base.@_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

# import Random
# import Random: AbstractRNG
# @generated function _rand(
#     randfun,
#     rng::AbstractRNG,
#     ::Type{SSC},
# ) where {N, SSC <: SimpleSymmetric{N}}
#     T = eltype(SSC)
#     if T == Any
#         T = Float64
#     end
#     L = triangularnumber(N)
#     v = [:(randfun(rng, $T)) for i in 1:L]
#     return quote
#         Base.@_inline_meta
#         $SSC(SVector(tuple($(v...))))
#     end
# end

# @inline Random.rand(
#     rng::AbstractRNG,
#     ::Type{SSC},
# ) where {SSC <: SimpleSymmetric} = _rand(rand, rng, SSC)
# @inline Random.randn(
#     rng::AbstractRNG,
#     ::Type{SSC},
# ) where {SSC <: SimpleSymmetric} = _rand(randn, rng, SSC)
# @inline Random.randexp(
#     rng::AbstractRNG,
#     ::Type{SSC},
# ) where {SSC <: SimpleSymmetric} = _rand(randexp, rng, SSC)
