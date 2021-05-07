"""
    rmap(fn, X)

Recursively apply `fn` to each element of `X`
"""
rmap(fn, X) = fn(X)
rmap(fn, X, Y) = fn(X, Y)
rmap(fn, X::Tuple) = map(x -> rmap(fn, x), X)
rmap(fn, X::Tuple, Y::Tuple) = map((x, y) -> rmap(fn, x, y), X, Y)
rmap(fn, X::NamedTuple) = map(x -> rmap(fn, x), X)
rmap(fn, X::NamedTuple{names}, Y::NamedTuple{names}) where {names} =
    map((x, y) -> rmap(fn, x, y), X, Y)

rmaptype(fn, ::Type{T}) where {T} = fn(T)
rmaptype(fn, ::Type{T}) where {T <: Tuple} =
    Tuple{map(fn, tuple(T.parameters...))...}
rmaptype(fn, ::Type{T}) where {T <: NamedTuple{names, tup}} where {names, tup} =
    NamedTuple{names, rmaptype(fn, tup)}


"""
    rscale(w, X)
    w ⊠ X

Recursively scale each element of `X` by `w`.
"""
rscale(w, X) = rmap(x -> w * x, X)
const ⊠ = rscale

"""
    radd(X, Y)
    X ⊞ Y

Recursively add elements of `X` and `Y`.
"""
radd(X, Y) = rmap(+, X, Y)
const ⊞ = radd



"""
    rmuladd(w, X, Y)

Recursively add elements of `w * X + Y`.
"""
rmuladd(w, X, Y) = rmap((x, y) -> muladd(w, x, y), X, Y)

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
