"""
    RecursiveApply

This module contains operators to recurse over nested `Tuple`s or `NamedTuple`s.

To extend to another type `T`, define `RecursiveApply.rmap(fn, args::T...)`
"""
module RecursiveApply

export ⊞, ⊠, ⊟, tuplemap

"""
    tuplemap(fn::Function, tup)

A simpler `map` impl for mapping function `fn` a tuple argument `tup`
"""
@inline function tuplemap(fn::F, tup::Tuple) where {F}
    N = length(tup)
    ntuple(Val(N)) do I
        Base.@_inline_meta
        @inbounds elem = tup[I]
        fn(elem)
    end
end

"""
    tuplemap(fn::Function, tup1, tup2)

A simpler `map` impl for mapping function `fn` over `tup1`, `tup2` tuple arguments
"""
@inline function tuplemap(fn::F, tup1::Tuple, tup2::Tuple) where {F}
    N1 = length(tup1)
    N2 = length(tup2)
    ntuple(Val(min(N1, N2))) do I
        Base.@_inline_meta
        @inbounds elem1 = tup1[I]
        @inbounds elem2 = tup2[I]
        fn(elem1, elem2)
    end
end

"""
    rmap(fn, X...)

Recursively apply `fn` to each element of `X`
"""
rmap(fn::F, X) where {F} = fn(X)
rmap(fn::F, X, Y) where {F} = fn(X, Y)
rmap(fn::F, X::Tuple) where {F} = tuplemap(x -> rmap(fn, x), X)
rmap(fn::F, X::Tuple, Y::Tuple) where {F} =
    tuplemap((x, y) -> rmap(fn, x, y), X, Y)
rmap(fn::F, X::NamedTuple{names}) where {F, names} =
    NamedTuple{names}(rmap(fn, Tuple(X)))
rmap(fn::F, X::NamedTuple{names}, Y::NamedTuple{names}) where {F, names} =
    NamedTuple{names}(rmap(fn, Tuple(X), Tuple(Y)))

"""
    rmaptype(fn, T)

The return type of `rmap(fn, X::T)`.
"""
rmaptype(fn::F, ::Type{T}) where {F, T} = fn(T)
rmaptype(fn::F, ::Type{T}) where {F, T <: Tuple} =
    Tuple{tuplemap(fn, tuple(T.parameters...))...}
rmaptype(
    fn::F,
    ::Type{T},
) where {F, T <: NamedTuple{names, tup}} where {names, tup} =
    NamedTuple{names, rmaptype(fn, tup)}

"""
    rmul(X, Y)
    X ⊠ Y

Recursively scale each element of `X` by `Y`.
"""
rmul(X, Y) = rmap(*, X, Y)
rmul(w::Number, X) = rmap(x -> w * x, X)
rmul(X, w::Number) = rmap(x -> x * w, X)
rmul(w1::Number, w2::Number) = w1 * w2
const ⊠ = rmul

"""
    radd(X, Y)
    X ⊞ Y

Recursively add elements of `X` and `Y`.
"""
radd(X, Y) = rmap(+, X, Y)
radd(w::Number, X) = rmap(x -> w + x, X)
radd(X, w::Number) = rmap(x -> x + w, X)
radd(w1::Number, w2::Number) = w1 + w2
const ⊞ = radd

# Adapted from Base/operators.jl for general nary operator fallbacks
for op in (:rmul, :radd)
    @eval begin
        ($op)(a, b, c, xs...) = Base.afoldl($op, ($op)(($op)(a, b), c), xs...)
    end
end

"""
    rsub(X, Y)
    X ⊟ Y

Recursively subtract elements of `Y` from `X`.
"""
rsub(X) = rmap(-, X)
rsub(X, Y) = rmap(-, X, Y)
rsub(X, w::Number) = rmap(x -> x - w, X)
rsub(w::Number, X) = rmap(x -> w - x, X)
rsub(w1::Number, w2::Number) = w1 - w2
const ⊟ = rsub

"""
    rdiv(X, Y)

Recursively divide each element of `X` by `Y`
"""
rdiv(X, Y) = rmap(/, X, Y)
rdiv(X, w::Number) = rmap(x -> x / w, X)
rdiv(w::Number, X) = rmap(x -> w / x, X)
rdiv(w1::Number, w2::Number) = w1 / w2

"""
    rmuladd(w, X, Y)

Recursively add elements of `w * X + Y`.
"""
rmuladd(w::Number, X, Y) = rmap((x, y) -> muladd(w, x, y), X, Y)
rmuladd(X, w::Number, Y) = rmap((x, y) -> muladd(x, w, y), X, Y)
rmuladd(w::Number, x::Number, y::Number) = muladd(w, x, y)

"""
    rmatmul1(W, S, i, j)

Recursive matrix product along the 1st dimension of `S`. Equivalent to:

    mapreduce(⊠, ⊞, W[i,:], S[:,j])

"""
function rmatmul1(W, S, i, j)
    Nq = size(W, 2)
    r = W[i, 1] ⊠ S[1, j]
    for ii in 2:Nq
        r = rmuladd(W[i, ii], S[ii, j], r)
    end
    return r
end

"""
    rmatmul2(W, S, i, j)

Recursive matrix product along the 2nd dimension `S`. Equivalent to:

    mapreduce(⊠, ⊞, W[j,:], S[i, :])

"""
function rmatmul2(W, S, i, j)
    Nq = size(W, 2)
    r = W[j, 1] ⊠ S[i, 1]
    for jj in 2:Nq
        r = rmuladd(W[j, jj], S[i, jj], r)
    end
    return r
end

end # module
