"""
    RecursiveApply

This module contains operators to recurse over nested `Tuple`s or `NamedTuple`s.

To extend to another type `T`, define `RecursiveApply.rmap(fn, args::T...)`
"""
module RecursiveApply

export ⊞, ⊠, ⊟

"""
    rmap(fn, X...)

Recursively apply `fn` to each element of `X`
"""
rmap(fn::F, X) where {F} = fn(X)
rmap(fn::F, X, Y) where {F} = fn(X, Y)
rmap(fn::F, X::Tuple) where {F} = map(x -> rmap(fn, x), X)
rmap(fn::F, X::Tuple, Y::Tuple) where {F} = map((x, y) -> rmap(fn, x, y), X, Y)
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
    Tuple{map(fn, tuple(T.parameters...))...}
rmaptype(
    fn::F,
    ::Type{T},
) where {F, T <: NamedTuple{names, tup}} where {names, tup} =
    NamedTuple{names, rmaptype(fn, tup)}

"""
    rmul(w, X)
    w ⊠ X

Recursively scale each element of `X` by `w`.
"""
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
const ⊞ = radd

"""
    rsub(X, Y)
    X ⊟ Y

Recursively subtract elements of `Y` from `X`.
"""
rsub(X) = rmap(-, X)
rsub(X, Y) = rmap(-, X, Y)
const ⊟ = rsub

rdiv(X, w::Number) = rmap(x -> x / w, X)

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
