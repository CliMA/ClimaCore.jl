using StaticArrays, LinearAlgebra

"""
    AbstractAxis

An axis of a [`AxisTensor`](@ref).
"""
abstract type AbstractAxis{I} end

"""
    dual(ax::AbstractAxis)

The dual axis to `ax`.

```
julia> using ClimaCore.Geometry

julia> Geometry.dual(Geometry.Covariant12Axis())
ClimaCore.Geometry.ContravariantAxis{(1, 2)}()

julia> Geometry.dual(Geometry.Cartesian123Axis())
ClimaCore.Geometry.CartesianAxis{(1, 2, 3)}()
```
"""
function dual end

struct CovariantAxis{I} <: AbstractAxis{I} end
symbols(::CovariantAxis) = (:u₁, :u₂, :u₃)

struct ContravariantAxis{I} <: AbstractAxis{I} end
symbols(::ContravariantAxis) = (:u¹, :u², :u³)
dual(::CovariantAxis{I}) where {I} = ContravariantAxis{I}()
dual(::ContravariantAxis{I}) where {I} = CovariantAxis{I}()

struct LocalAxis{I} <: AbstractAxis{I} end
symbols(::LocalAxis) = (:u, :v, :w)
dual(::LocalAxis{I}) where {I} = LocalAxis{I}()

struct CartesianAxis{I} <: AbstractAxis{I} end
symbols(::CartesianAxis) = (:u1, :u2, :u3)
dual(::CartesianAxis{I}) where {I} = CartesianAxis{I}()


@inline function idxin(I::Tuple{Vararg{Int}}, i::Int)
    N = length(I)
    for n in 1:N
        if I[n] == i
            return n
        end
    end
    return nothing
end

@inline function symidx(ax::AbstractAxis{I}, name::Symbol) where {I}
    S = symbols(ax)
    name == S[1] ? idxin(I, 1) :
    name == S[2] ? idxin(I, 2) :
    name == S[3] ? idxin(I, 3) : error("$ax has no symbol $name")
end

# most of these are required for printing
Base.length(ax::AbstractAxis{I}) where {I} = length(I)
Base.unitrange(ax::AbstractAxis) = StaticArrays.SOneTo(length(ax))
Base.LinearIndices(axes::Tuple{Vararg{AbstractAxis}}) =
    1:prod(map(length, axes))
Base.checkindex(::Type{Bool}, ax::AbstractAxis, i) =
    Base.checkindex(Bool, Base.unitrange(ax), i)
Base.lastindex(ax::AbstractAxis) = length(ax)
Base.getindex(m::AbstractAxis, i::Int) = i


"""
    AxisTensor(axes, components)

An `AxisTensor` is a wrapper around a `StaticArray`, where each dimension is
labelled with an [`AbstractAxis`](@ref). These axes must be consistent for
operations such as addition or subtraction, or be dual to each other for
operations such as multiplication.


# See also
[`components`](@ref) to obtain the underlying array.
"""
struct AxisTensor{
    T,
    N,
    A <: NTuple{N, AbstractAxis},
    S <: StaticArray{<:Tuple, T, N},
} <: AbstractArray{T, N}
    axes::A
    components::S
end

AxisTensor(
    axes::A,
    components::S,
) where {
    A <: Tuple{Vararg{AbstractAxis}},
    S <: StaticArray{<:Tuple, T, N},
} where {T, N} = AxisTensor{T, N, A, S}(axes, components)

AxisTensor(axes::Tuple{Vararg{AbstractAxis}}, components) =
    AxisTensor(axes, SArray{Tuple{map(length, axes)...}}(components))

# if the axes are already defined
(AxisTensor{T, N, A, S} where {S})(
    components::AbstractArray{T, N},
) where {T, N, A} = AxisTensor(A.instance, components)
(AxisTensor{T, N, A, S} where {T, S})(
    components::AbstractArray{<:Any, N},
) where {N, A} = AxisTensor(A.instance, components)


Base.axes(a::AxisTensor) = getfield(a, :axes)
Base.axes(::Type{AxisTensor{T, N, A, S}}) where {T, N, A, S} = A.instance
Base.size(a::AxisTensor) = map(length, axes(a))

"""
    components(a::AxisTensor)

Returns a `StaticArray` containing the components of `a` in its stored basis.
"""
components(a::AxisTensor) = getfield(a, :components)

Base.getindex(v::AxisTensor, i...) = getindex(components(v), i...)


function Base.getindex(
    v::AxisTensor{<:Any, 2, Tuple{A1, A2}},
    ::Colon,
    i::Integer,
) where {A1, A2}
    AxisVector(axes(v, 1), getindex(components(v), :, i))
end
function Base.getindex(
    v::AxisTensor{<:Any, 2, Tuple{A1, A2}},
    i::Integer,
    ::Colon,
) where {A1, A2}
    AxisVector(axes(v, 2), getindex(components(v), i, :))
end


Base.map(f::F, a::AxisTensor) where {F} =
    AxisTensor(axes(a), map(f, components(a)))
Base.map(
    f::F,
    a::AxisTensor{Ta, N, A},
    b::AxisTensor{Tb, N, A},
) where {F, Ta, Tb, N, A} =
    AxisTensor(axes(a), map(f, components(a), components(b)))
#Base.map(f, a::AxisTensor{Ta,N}, b::AxisTensor{Tb,N}) where {Ta,Tb,N} =
#    map(f, promote(a,b)...)

Base.zero(a::AxisTensor{T, N, A, S}) where {T, N, A, S} = zero(typeof(a))
Base.zero(::Type{AxisTensor{T, N, A, S}}) where {T, N, A, S} =
    AxisTensor(axes(AxisTensor{T, N, A, S}), zero(S))

import Base: +, -, *, /, \, ==

# Unary ops
@inline +(a::AxisTensor) = map(+, a)
@inline -(a::AxisTensor) = map(-, a)

# Binary ops
# Between arrays
@inline +(a::AxisTensor, b::AxisTensor) = map(+, a, b)
@inline -(a::AxisTensor, b::AxisTensor) = map(-, a, b)
# Scalar-array
@inline *(a::Number, b::AxisTensor) = map(c -> a * c, b)
@inline *(a::AxisTensor, b::Number) = map(c -> c * b, a)
@inline /(a::AxisTensor, b::Number) = map(c -> c / b, a)
@inline \(a, b::AxisTensor) = map(c -> a \ c, b)

@inline (==)(a::AxisTensor, b::AxisTensor) =
    axes(a) == axes(b) && components(a) == components(b)

# vectors
const AxisVector{T, A1, S} = AxisTensor{T, 1, Tuple{A1}, S}

AxisVector(ax::A1, v::SVector{N, T}) where {A1 <: AbstractAxis, N, T} =
    AxisVector{T, A1, SVector{N, T}}((ax,), v)

(AxisVector{T, A, SVector{1, T}} where {T})(arg1) where {A} =
    AxisVector(A.instance, SVector(arg1))
(AxisVector{T, A, SVector{2, T}} where {T})(arg1, arg2) where {A} =
    AxisVector(A.instance, SVector(arg1, arg2))
(AxisVector{T, A, SVector{3, T}} where {T})(arg1, arg2, arg3) where {A} =
    AxisVector(A.instance, SVector(arg1, arg2, arg3))

const CovariantVector{T, A1 <: CovariantAxis, S} = AxisVector{T, A1, S}
const ContravariantVector{T, A1 <: ContravariantAxis, S} = AxisVector{T, A1, S}
const CartesianVector{T, A1 <: CartesianAxis, S} = AxisVector{T, A1, S}

Base.propertynames(x::AxisVector) = symbols(axes(x, 1))
function Base.getproperty(x::AxisVector, name::Symbol)
    n = symidx(axes(x, 1), name)
    isnothing(n) ? zero(eltype(x)) : components(x)[n]
end


const AdjointAxisVector{T, A1, S} = Adjoint{T, AxisVector{T, A1, S}}

components(va::AdjointAxisVector) = components(parent(va))'
Base.getindex(va::AdjointAxisVector, i::Int) = getindex(components(va), i)
Base.getindex(va::AdjointAxisVector, i::Int, j::Int) =
    getindex(components(va), i, j)

# 2-tensors
const Axis2Tensor{T, A, S} = AxisTensor{T, 2, A, S}
Axis2Tensor(
    axes::Tuple{AbstractAxis, AbstractAxis},
    components::AbstractMatrix,
) = AxisTensor(axes, components)

const AdjointAxis2Tensor{T, A, S} = Adjoint{T, Axis2Tensor{T, A, S}}
components(va::AdjointAxis2Tensor) = components(parent(va))'

const Axis2TensorOrAdj{T, A, S} =
    Union{Axis2Tensor{T, A, S}, AdjointAxis2Tensor{T, A, S}}

# based on 1st dimension
const Covariant2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {A <: Tuple{CovariantAxis, AbstractAxis}}
const Contravariant2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {A <: Tuple{ContravariantAxis, AbstractAxis}}
const Cartesian2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {A <: Tuple{CartesianAxis, AbstractAxis}}

const CovariantTensor = Union{CovariantVector, Covariant2Tensor}
const ContravariantTensor = Union{ContravariantVector, Contravariant2Tensor}
const CartesianTensor = Union{CartesianVector, Cartesian2Tensor}

for I in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    strI = join(I)
    N = length(I)
    @eval begin
        const $(Symbol(:Covariant, strI, :Axis)) = CovariantAxis{$I}
        const $(Symbol(:Covariant, strI, :Vector)){T} =
            AxisVector{T, CovariantAxis{$I}, SVector{$N, T}}

        const $(Symbol(:Contravariant, strI, :Axis)) = ContravariantAxis{$I}
        const $(Symbol(:Contravariant, strI, :Vector)){T} =
            AxisVector{T, ContravariantAxis{$I}, SVector{$N, T}}

        const $(Symbol(:Cartesian, strI, :Axis)) = CartesianAxis{$I}
        const $(Symbol(:Cartesian, strI, :Vector)){T} =
            AxisVector{T, CartesianAxis{$I}, SVector{$N, T}}
    end
end

# LinearAlgebra

check_axes(::A, ::A) where {A} = nothing
check_axes(ax1, ax2) where {A} =
    throw(DimensionMismatch("$ax1 and $ax2 do not match"))

check_dual(ax1, ax2) = _check_dual(ax1, ax2, dual(ax2))
_check_dual(::A, _, ::A) where {A} = nothing
_check_dual(ax1, ax2, _) =
    throw(DimensionMismatch("$ax1 is not dual with $ax2"))


function LinearAlgebra.dot(x::AxisVector, y::AxisVector)
    check_dual(axes(x, 1), axes(y, 1))
    return LinearAlgebra.dot(components(x), components(y))
end

function Base.:*(x::AxisVector, y::AdjointAxisVector)
    AxisTensor((axes(x, 1), axes(y, 2)), components(x) * components(y))
end
function Base.:*(A::Axis2TensorOrAdj, x::AxisVector)
    check_dual(axes(A, 2), axes(x, 1))
    return AxisVector(axes(A, 1), components(A) * components(x))
end
function Base.:*(A::Axis2TensorOrAdj, B::Axis2TensorOrAdj)
    check_dual(axes(A, 2), axes(B, 1))
    return AxisTensor((axes(A, 1), axes(B, 2)), components(A) * components(B))
end

function Base.inv(A::Axis2TensorOrAdj)
    return AxisTensor((dual(axes(A, 2)), dual(axes(A, 1))), inv(components(A)))
end
function Base.:\(A::Axis2TensorOrAdj, x::AxisVector)
    check_axes(axes(A, 1), axes(x, 1))
    return AxisVector(dual(axes(A, 2)), components(A) \ components(x))
end


# can only compute norm if self-dual
function LinearAlgebra.norm(x::AxisVector)
    check_dual(axes(x, 1), axes(x, 1))
    LinearAlgebra.norm(components(x))
end
function LinearAlgebra.norm_sqr(x::AxisVector)
    check_dual(axes(x, 1), axes(x, 1))
    LinearAlgebra.norm_sqr(components(x))
end


LinearAlgebra.cross(x::ContravariantVector, y::ContravariantVector) =
    Covariant123Vector(
        x.u² * y.u³ - x.u³ * y.u²,
        x.u³ * y.u¹ - x.u¹ * y.u³,
        x.u¹ * y.u² - x.u² * y.u¹,
    )

LinearAlgebra.cross(x::Contravariant12Vector, y::Contravariant12Vector) =
    Covariant3Vector(x.u¹ * y.u² - x.u² * y.u¹)

LinearAlgebra.cross(x::Contravariant12Vector, y::Contravariant3Vector) =
    Covariant12Vector(x.u² * y.u³, -x.u¹ * y.u³)

LinearAlgebra.cross(x::Cartesian12Vector, y::Cartesian3Vector) =
    Cartesian12Vector(x.u2 * y.u3, -x.u1 * y.u3)
LinearAlgebra.cross(y::Cartesian3Vector, x::Cartesian12Vector) =
    Cartesian12Vector(-x.u2 * y.u3, x.u1 * y.u3)

function Base.:(+)(A::Axis2Tensor, b::LinearAlgebra.UniformScaling)
    check_dual(axes(A)...)
    AxisTensor(axes(A), components(A) + b)
end
function Base.:(-)(A::Axis2Tensor, b::LinearAlgebra.UniformScaling)
    check_dual(axes(A)...)
    AxisTensor(axes(A), components(A) - b)
end


"""
    outer(x, y)
    x ⊗ y

Compute the outer product of `x` and `y`. Typically `x` will be a vector, and
`y` can be either a number, vector or tuple/named tuple.

```julia
# vector ⊗ scalar = vector
julia> [1.0,2.0] ⊗ 2.0
2-element Vector{Float64}:
 2.0
 4.0

# vector ⊗ vector = matrix
julia> [1.0,2.0] ⊗ [1.0,3.0]
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  6.0

# vector ⊗ tuple = recursion
julia> [1.0,2.0] ⊗ (1.0, (a=2.0, b=3.0))
([1.0, 2.0], (a = [2.0, 4.0], b = [3.0, 6.0]))
```
"""
function outer end
const ⊗ = outer

function outer(x::AbstractVector, y::AbstractVector)
    x * y'
end
function outer(x::AbstractVector, y::Number)
    x * y
end
function outer(x::AbstractVector, y)
    RecursiveApply.rmap(y -> x ⊗ y, y)
end

function _transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end
@generated function _transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            errcond = :($errcond || x[$n] != 0)
        end
    end
    vals = []
    for i in Ito
        val = :(zero(T))
        for n in 1:N
            if i == Ifrom[n]
                val = :(x[$n])
                break
            end
        end
        push!(vals, val)
    end
    quote
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        AxisVector(ato, SVector($(vals...)))
    end
end

function _transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@generated function _transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
    N = length(Ifrom)
    M = length(J)
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            for m in 1:M
                errcond = :($errcond || x[$n, $m] != 0)
            end
        end
    end
    vals = []
    for m in 1:M
        for i in Ito
            val = :(zero(T))
            for n in 1:N
                if i == Ifrom[n]
                    val = :(x[$n, $m])
                    break
                end
            end
            push!(vals, val)
        end
    end
    quote
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        Axis2Tensor((ato, axes(x, 2)), SMatrix{$(length(Ito)), $M}($(vals...)))
    end
end

function transform(ato::CovariantAxis, v::CovariantTensor)
    _transform(ato, v)
end
function transform(ato::ContravariantAxis, v::ContravariantTensor)
    _transform(ato, v)
end
function transform(ato::CartesianAxis, v::CartesianTensor)
    _transform(ato, v)
end
