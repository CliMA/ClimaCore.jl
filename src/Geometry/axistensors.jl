using StaticArrays, LinearAlgebra

"""
    AbstractAxis

An axis of a [`AxisTensor`](@ref).
"""
abstract type AbstractAxis{I} end

Base.Broadcast.broadcastable(a::AbstractAxis) = a

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
@inline symbols(::CovariantAxis) = (:u₁, :u₂, :u₃)

struct ContravariantAxis{I} <: AbstractAxis{I} end
@inline symbols(::ContravariantAxis) = (:u¹, :u², :u³)
dual(::CovariantAxis{I}) where {I} = ContravariantAxis{I}()
dual(::ContravariantAxis{I}) where {I} = CovariantAxis{I}()

struct LocalAxis{I} <: AbstractAxis{I} end
@inline symbols(::LocalAxis) = (:u, :v, :w)
dual(::LocalAxis{I}) where {I} = LocalAxis{I}()

struct CartesianAxis{I} <: AbstractAxis{I} end
@inline symbols(::CartesianAxis) = (:u1, :u2, :u3)
dual(::CartesianAxis{I}) where {I} = CartesianAxis{I}()

coordinate_axis(::Type{<:XPoint}) = (1,)
coordinate_axis(::Type{<:YPoint}) = (2,)
coordinate_axis(::Type{<:ZPoint}) = (3,)
coordinate_axis(::Type{<:XYPoint}) = (1, 2)
coordinate_axis(::Type{<:XZPoint}) = (1, 3)
coordinate_axis(::Type{<:XYZPoint}) = (1, 2, 3)

coordinate_axis(::Type{<:Cartesian1Point}) = (1,)
coordinate_axis(::Type{<:Cartesian2Point}) = (2,)
coordinate_axis(::Type{<:Cartesian3Point}) = (3,)

coordinate_axis(::Type{<:Cartesian123Point}) = (1, 2, 3)

coordinate_axis(::Type{<:LatLongPoint}) = (1, 2)

coordinate_axis(coord::AbstractPoint) = coordinate_axis(typeof(coord))

@inline idxin(I::Tuple{Int}, i::Int) = 1

@inline function idxin(I::Tuple{Int, Int}, i::Int)
    @inbounds begin
        if I[1] == i
            return 1
        else
            return 2
        end
    end
end

@inline function idxin(I::Tuple{Int, Int, Int}, i::Int)
    @inbounds begin
        if I[1] == i
            return 1
        elseif I[2] == i
            return 2
        else
            return 3
        end
    end
end

#= For avoiding JET failures =#
error_on_no_name_found() = true

@inline function symidx(ax::AbstractAxis{I}, name::Symbol) where {I}
    S = symbols(ax)
    if name == S[1]
        return idxin(I, 1)
    elseif name == S[2]
        return idxin(I, 2)
    elseif name == S[3]
        return idxin(I, 3)
    elseif error_on_no_name_found()
        error("$ax has no symbol $name")
    else
        return -1 # for type stability
    end
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

@inline AxisTensor(
    axes::A,
    components::S,
) where {
    A <: Tuple{Vararg{AbstractAxis}},
    S <: StaticArray{<:Tuple, T, N},
} where {T, N} = AxisTensor{T, N, A, S}(axes, components)

@inline AxisTensor(axes::Tuple{Vararg{AbstractAxis}}, components) =
    AxisTensor(axes, SArray{Tuple{map(length, axes)...}}(components))

# if the axes are already defined
@inline (AxisTensor{T, N, A, S} where {S})(
    components::AbstractArray{T, N},
) where {T, N, A} = AxisTensor(A.instance, components)
@inline (AxisTensor{T, N, A, S} where {T, S})(
    components::AbstractArray{<:Any, N},
) where {N, A} = AxisTensor(A.instance, components)


Base.axes(a::AxisTensor) = getfield(a, :axes)
Base.axes(::Type{AxisTensor{T, N, A, S}}) where {T, N, A, S} = A.instance
Base.size(a::AxisTensor) = map(length, axes(a))

Base.rand(::Type{AxisTensor{T, N, A, S}}) where {T, N, A, S} =
    AxisTensor{T, N, A, S}(A.instance, rand(S))

Base.zeros(::Type{AxisTensor{T, N, A, S}}) where {T, N, A, S} =
    AxisTensor{T, N, A, S}(A.instance, zeros(S))

function Base.show(io::IO, a::AxisTensor{T, N, A, S}) where {T, N, A, S}
    print(
        io,
        "AxisTensor{$T, $N, $A, $S}($(getfield(a, :axes)), $(getfield(a, :components)))",
    )
end

"""
    components(a::AxisTensor)

Returns a `StaticArray` containing the components of `a` in its stored basis.
"""
@inline components(a::AxisTensor) = getfield(a, :components)

@inline Base.getindex(v::AxisTensor, i...) =
    @inbounds getindex(components(v), i...)


@inline function Base.getindex(
    v::AxisTensor{<:Any, 2, Tuple{A1, A2}},
    ::Colon,
    i::Integer,
) where {A1, A2}
    AxisVector(axes(v, 1), @inbounds getindex(components(v), :, i))
end
@inline function Base.getindex(
    v::AxisTensor{<:Any, 2, Tuple{A1, A2}},
    i::Integer,
    ::Colon,
) where {A1, A2}
    AxisVector(axes(v, 2), @inbounds getindex(components(v), i, :))
end


@inline Base.map(f::F, a::AxisTensor) where {F} =
    AxisTensor(axes(a), map(f, components(a)))
@inline Base.map(
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

@inline AxisVector(ax::A1, v::SVector{N, T}) where {A1 <: AbstractAxis, N, T} =
    AxisVector{T, A1, SVector{N, T}}((ax,), v)

(AxisVector{T, A, SVector{1, T}} where {T})(arg1::Real) where {A} =
    AxisVector(A.instance, SVector(arg1))
(AxisVector{T, A, SVector{2, T}} where {T})(arg1::Real, arg2::Real) where {A} =
    AxisVector(A.instance, SVector(arg1, arg2))
(AxisVector{T, A, SVector{3, T}} where {T})(
    arg1::Real,
    arg2::Real,
    arg3::Real,
) where {A} = AxisVector(A.instance, SVector(arg1, arg2, arg3))

const CovariantVector{T, I, S} = AxisVector{T, CovariantAxis{I}, S}
const ContravariantVector{T, I, S} = AxisVector{T, ContravariantAxis{I}, S}
const CartesianVector{T, I, S} = AxisVector{T, CartesianAxis{I}, S}
const LocalVector{T, I, S} = AxisVector{T, LocalAxis{I}, S}

Base.propertynames(x::AxisVector) = symbols(axes(x, 1))
@inline function Base.getproperty(x::AxisVector, name::Symbol)
    n = symidx(axes(x, 1), name)
    if isnothing(n)
        zero(eltype(x))
    else
        @inbounds components(x)[n]
    end
end


const AdjointAxisVector{T, A1, S} = Adjoint{T, AxisVector{T, A1, S}}

@inline components(va::AdjointAxisVector) = components(parent(va))'
@inline Base.getindex(va::AdjointAxisVector, i::Int) =
    @inbounds getindex(components(va), i)
@inline Base.getindex(va::AdjointAxisVector, i::Int, j::Int) =
    @inbounds getindex(components(va), i, j)

# 2-tensors
const Axis2Tensor{T, A, S} = AxisTensor{T, 2, A, S}
Axis2Tensor(
    axes::Tuple{AbstractAxis, AbstractAxis},
    components::AbstractMatrix,
) = AxisTensor(axes, components)

const AdjointAxis2Tensor{T, A, S} = Adjoint{T, Axis2Tensor{T, A, S}}
@inline components(va::AdjointAxis2Tensor) = components(parent(va))'

const Axis2TensorOrAdj{T, A, S} =
    Union{Axis2Tensor{T, A, S}, AdjointAxis2Tensor{T, A, S}}

# based on 1st dimension
const Covariant2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {T, A <: Tuple{CovariantAxis, AbstractAxis}, S}
const Contravariant2Tensor{T, A, S} = Axis2Tensor{
    T,
    A,
    S,
} where {T, A <: Tuple{ContravariantAxis, AbstractAxis}, S}
const Cartesian2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {T, A <: Tuple{CartesianAxis, AbstractAxis}, S}
const Local2Tensor{T, A, S} =
    Axis2Tensor{T, A, S} where {T, A <: Tuple{LocalAxis, AbstractAxis}, S}

const CovariantTensor = Union{CovariantVector, Covariant2Tensor}
const ContravariantTensor = Union{ContravariantVector, Contravariant2Tensor}
const CartesianTensor = Union{CartesianVector, Cartesian2Tensor}
const LocalTensor = Union{LocalVector, Local2Tensor}

for I in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    strI = join(I)
    N = length(I)
    strUVW = join(map(i -> [:U, :V, :W][i], I))
    @eval begin
        const $(Symbol(:Covariant, strI, :Axis)) = CovariantAxis{$I}
        const $(Symbol(:Covariant, strI, :Vector)){T} =
            CovariantVector{T, $I, SVector{$N, T}}

        const $(Symbol(:Contravariant, strI, :Axis)) = ContravariantAxis{$I}
        const $(Symbol(:Contravariant, strI, :Vector)){T} =
            ContravariantVector{T, $I, SVector{$N, T}}

        const $(Symbol(:Cartesian, strI, :Axis)) = CartesianAxis{$I}
        const $(Symbol(:Cartesian, strI, :Vector)){T} =
            CartesianVector{T, $I, SVector{$N, T}}

        const $(Symbol(strUVW, :Axis)) = LocalAxis{$I}
        const $(Symbol(strUVW, :Vector)){T} = LocalVector{T, $I, SVector{$N, T}}
    end
end

# LinearAlgebra

check_axes(::A, ::A) where {A} = nothing
check_axes(ax1, ax2) = throw(DimensionMismatch("$ax1 and $ax2 do not match"))

check_dual(ax1, ax2) = _check_dual(ax1, ax2, dual(ax2))
_check_dual(::A, _, ::A) where {A} = nothing
_check_dual(ax1, ax2, _) =
    throw(DimensionMismatch("$ax1 is not dual with $ax2"))


@inline function LinearAlgebra.dot(x::AxisVector, y::AxisVector)
    check_dual(axes(x, 1), axes(y, 1))
    return LinearAlgebra.dot(components(x), components(y))
end

@inline function Base.:*(x::AxisVector, y::AdjointAxisVector)
    AxisTensor((axes(x, 1), axes(y, 2)), components(x) * components(y))
end
@inline function Base.:*(A::Axis2TensorOrAdj, x::AxisVector)
    check_dual(axes(A, 2), axes(x, 1))
    return AxisVector(axes(A, 1), components(A) * components(x))
end
@inline function Base.:*(A::Axis2TensorOrAdj, B::Axis2TensorOrAdj)
    check_dual(axes(A, 2), axes(B, 1))
    return AxisTensor((axes(A, 1), axes(B, 2)), components(A) * components(B))
end

@inline function Base.inv(A::Axis2TensorOrAdj)
    return AxisTensor((dual(axes(A, 2)), dual(axes(A, 1))), inv(components(A)))
end
@inline function Base.:\(A::Axis2TensorOrAdj, x::AxisVector)
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


LinearAlgebra.cross(x::Cartesian12Vector, y::Cartesian3Vector) =
    Cartesian12Vector(x.u2 * y.u3, -x.u1 * y.u3)
LinearAlgebra.cross(y::Cartesian3Vector, x::Cartesian12Vector) =
    Cartesian12Vector(-x.u2 * y.u3, x.u1 * y.u3)
LinearAlgebra.cross(x::UVVector, y::WVector) = UVVector(x.v * y.w, -x.u * y.w)
LinearAlgebra.cross(y::WVector, x::UVVector) = UVVector(-x.v * y.w, x.u * y.w)

@inline function Base.:(+)(A::Axis2Tensor, b::LinearAlgebra.UniformScaling)
    check_dual(axes(A)...)
    AxisTensor(axes(A), components(A) + b)
end
@inline function Base.:(-)(A::Axis2Tensor, b::LinearAlgebra.UniformScaling)
    check_dual(axes(A)...)
    AxisTensor(axes(A), components(A) - b)
end

"""
    transform(axis, tensor, local_geometry)
    project(axis, tensor, local_geometry)

Transform and project the first axis of a tensor to a different basis.

The conversion rules are defined as:

- `v::Covariant` => `Local`:     `∂ξ∂x' * v`
- `v::Local` => `Contravariant`: `∂ξ∂x  * v`
- `v::Contravariant` => `Local`: `∂x∂ξ  * v`
- `v::Local` => `Covariant`:     `∂x∂ξ' * v`
- `v::Covariant` => `Contravariant`:  `∂ξ∂x * (∂ξ∂x' * v) = gⁱʲ * v`
- `v::Contravariant` => `Covariant`:  `∂x∂ξ' * ∂x∂ξ * v   = gᵢⱼ * v`

# Example
Consider the conversion from a  `Covariant12Vector` to a `Contravariant12Axis`. Mathematically, we can write this as

```
[ v¹ ]   [g¹¹  g¹²  g¹³ ]   [ v₁ ]
[ v² ] = [g²¹  g²²  g²³ ] * [ v₂ ]
[ v³ ]   [g³¹  g³²  g³³ ]   [ 0  ]
```

`project` will drop v³ term no matter what the value is, i.e. it returns

```
[ v¹ ]   [g¹¹ v₁  + g¹² v₂ ]
[ v² ] = [g²¹ v₁  + g²² v₂ ]
[ 0  ]   [<drops this>]
```

`transform` will drop the v³ term, but throw an error if it is non-zero (i.e. if the conversion is not exact)

```
[ v¹ ]   [g¹¹ v₁  + g¹² v₂ ]
[ v² ] = [g²¹ v₁  + g²² v₂ ]
[ 0  ]   [<asserts g²³ v₁  + g²³ v₂ == 0>]
```

"""
function transform end


@inline function _transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end

@inline function _project(
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
            errcond = :($errcond || x[$n] != zero(T))
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
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        @inbounds AxisVector(ato, SVector($(vals...)))
    end
end

@generated function _project(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
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
    return :(@inbounds AxisVector(ato, SVector($(vals...))))
end

@inline function _transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@inline function _project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

#= Set `assert_exact_transform() = true` for debugging=#
assert_exact_transform() = false

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
    if assert_exact_transform()
        errcond = false
        for n in 1:N
            i = Ifrom[n]
            if i ∉ Ito
                for m in 1:M
                    errcond = :($errcond || x[$n, $m] != zero(T))
                end
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
        Base.@_propagate_inbounds_meta
        if assert_exact_transform()
            if $errcond
                throw(InexactError(:transform, Ato, x))
            end
        end
        @inbounds Axis2Tensor(
            (ato, axes(x, 2)),
            SMatrix{$(length(Ito)), $M}($(vals...)),
        )
    end
end

@generated function _project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
    N = length(Ifrom)
    M = length(J)
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
    return :(@inbounds Axis2Tensor(
        (ato, axes(x, 2)),
        SMatrix{$(length(Ito)), $M}($(vals...)),
    ))
end

@inline transform(ato::CovariantAxis, v::CovariantTensor) = _project(ato, v)
@inline transform(ato::ContravariantAxis, v::ContravariantTensor) =
    _project(ato, v)
@inline transform(ato::CartesianAxis, v::CartesianTensor) = _project(ato, v)
@inline transform(ato::LocalAxis, v::LocalTensor) = _project(ato, v)

@inline project(ato::CovariantAxis, v::CovariantTensor) = _project(ato, v)
@inline project(ato::ContravariantAxis, v::ContravariantTensor) =
    _project(ato, v)
@inline project(ato::CartesianAxis, v::CartesianTensor) = _project(ato, v)
@inline project(ato::LocalAxis, v::LocalTensor) = _project(ato, v)


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

@inline function outer(x::AbstractVector, y::AbstractVector)
    x * y'
end
@inline function outer(x::AbstractVector, y::Number)
    x * y
end
@inline function outer(x::AbstractVector, y)
    RecursiveApply.rmap(y -> x ⊗ y, y)
end
