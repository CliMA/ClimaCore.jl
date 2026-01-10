import UnrolledUtilities: unrolled_all, unrolled_flatmap, unrolled_map
import UnrolledUtilities: unrolled_maximum, unrolled_unique, StaticOneTo

"""
    math_operator_map(f, itrs...)

Custom `map` implementation specialized for math operators like `+`, `*`, etc.

Default methods are provided for `Tuple`s and `NamedTuple`s, both of which use
zero-padding for missing values. For example, mapping over two `Tuple`s with
different lengths will add zeros to the end of the shorter `Tuple`, and mapping
over two `NamedTuple`s will add zeros for names that appear in one `NamedTuple`
but not the other, with zero values for `itr` computed as `zero(eltype(itr))`.

The interface can be extended to handle iterators of any type `T` by defining a
method for `math_operator_map(f, itrs::T...)`.
"""
function math_operator_map(f::F, itrs::Tuple...) where {F}
    get_value(itr, i) = i <= length(itr) ? itr[i] : zero(eltype(itr))
    get_all_values(itrs, i) = unrolled_map(Base.Fix2(get_value, i), itrs)
    all_indices = StaticOneTo(unrolled_maximum(length, itrs))
    return unrolled_map(splat(f) ∘ Base.Fix1(get_all_values, itrs), all_indices)
end
function math_operator_map(f::F, itrs::NamedTuple...) where {F}
    get_value(itr, name) = haskey(itr, name) ? itr[name] : zero(eltype(itr))
    get_all_values(itrs, name) = unrolled_map(Base.Fix2(get_value, name), itrs)
    all_names = unrolled_unique(unrolled_flatmap(keys, itrs))
    return NamedTuple{all_names}(
        unrolled_map(splat(f) ∘ Base.Fix1(get_all_values, itrs), all_names),
    )
end

"""
    supports_math_operator_map(itr)

Indicates whether a method of [`math_operator_map`](@ref) is defined for `itr`.

By default, this is only true for `Tuple`s and `NamedTuple`s, but the interface
can be extended to handle iterators of any type `T` by defining
`supports_math_operator_map(::T) = true`.
"""
supports_math_operator_map(_) = false
supports_math_operator_map(::Tuple) = true
supports_math_operator_map(::NamedTuple) = true

"""
    MathWrapper(itr)

Wraps an iterator for which [`supports_math_operator_map`](@ref) is true, so
that [`math_operator_map`](@ref) is automatically called for any math operator.
Standard iterator methods like `iterate` and `getindex` are also supported.
"""
struct MathWrapper{I}
    itr::I
end

"""
    nested_math_wrapper(itr)

Recursively applies the [`MathWrapper`](@ref) constructor to an iterator and
every value it contains for which [`supports_math_operator_map`](@ref) is true.
Values for which `supports_math_operator_map` is false are left unmodified.
"""
nested_math_wrapper(itr) =
    supports_math_operator_map(itr) ?
    MathWrapper(unrolled_map(nested_math_wrapper, itr)) : itr

can_map_over(itrs...) = unrolled_all(supports_math_operator_map, itrs)

math_operator_broadcast(f::F, x) where {F} =
    can_map_over(x) ? math_operator_map(f, x) : f(x)
math_operator_broadcast(f::F, x, y) where {F} =
    can_map_over(x, y) ? math_operator_map(f, x, y) :
    can_map_over(y) ? math_operator_map(Base.Fix1(f, x), y) :
    can_map_over(x) ? math_operator_map(Base.Fix2(f, y), x) :
    f(x, y)
math_operator_broadcast(f::F, x, y, z) where {F} =
    can_map_over(x, y, z) ? math_operator_map(f, x, y, z) :
    can_map_over(y, z) ? math_operator_map(Base.Fix1(f, x), y, z) :
    can_map_over(x, z) ? math_operator_map(Base.Fix2(f, y), x, z) :
    can_map_over(x, y) ? math_operator_map((x, y) -> f(x, y, z), x, y) :
    can_map_over(z) ? math_operator_map(Base.Fix1(Base.Fix1(f, x), y), z) :
    can_map_over(y) ? math_operator_map(Base.Fix2(Base.Fix1(f, x), z), y) :
    can_map_over(x) ? math_operator_map(Base.Fix2(Base.Fix2(f, y), z), x) :
    f(x, y, z)
# TODO: Replace the closure above with Base.Fix{3} after upgrading to Julia 1.12

function broadcasted_math_wrapper(f::F, itrs...) where {F}
    unwrapped_itrs = unrolled_map(x -> x isa MathWrapper ? x.itr : x, itrs)
    return MathWrapper(math_operator_broadcast(f, unwrapped_itrs...))
end

# Add methods for all math operators that are handled in base/promotion.jl,
# along with some other commonly used operators like zero, one, and inv.
const one_or_two_arg_ops = (:+, :-, :&, :|, :*, :xor, :min, :max)
for f in (one_or_two_arg_ops..., :zero, :one, :inv)
    method = :(Base.$f(x::MathWrapper))
    @eval $method = broadcasted_math_wrapper($f, x)
end
for f in (one_or_two_arg_ops..., :/, :^, :(==), :<, :>, :<=, :>=, :rem, :mod)
    for method in (
        :(Base.$f(x::MathWrapper, y::MathWrapper)),
        :(Base.$f(x, y::MathWrapper)),
        :(Base.$f(x::MathWrapper, y)),
    )
        @eval $method = broadcasted_math_wrapper($f, x, y)
    end
end
for f in (:fma, :muladd)
    for method in (
        :(Base.$f(x::MathWrapper, y::MathWrapper, z::MathWrapper)),
        :(Base.$f(x, y::MathWrapper, z::MathWrapper)),
        :(Base.$f(x::MathWrapper, y, z::MathWrapper)),
        :(Base.$f(x::MathWrapper, y::MathWrapper, z)),
        :(Base.$f(x, y, z::MathWrapper)),
        :(Base.$f(x, y::MathWrapper, z)),
        :(Base.$f(x::MathWrapper, y, z)),
    )
        @eval $method = broadcasted_math_wrapper($f, x, y, z)
    end
end
# TODO: Add methods from ForwardDiff.jl, including the ones for AMBIGUOUS_TYPES

Base.propertynames(x::MathWrapper) = propertynames(x.itr)
Base.getproperty(x::MathWrapper, name) = getproperty(x.itr, name)

Base.length(x::MathWrapper) = length(x.itr)
Base.isempty(x::MathWrapper) = isempty(x.itr)
Base.iterate(x::MathWrapper, i...) = iterate(x.itr, i...)

Base.keys(x::MathWrapper) = keys(x.itr)
Base.values(x::MathWrapper) = values(x.itr)
Base.pairs(x::MathWrapper) = pairs(x.itr)

Base.firstindex(x::MathWrapper) = firstindex(x.itr)
Base.lastindex(x::MathWrapper) = lastindex(x.itr)
Base.getindex(x::MathWrapper, i) = getindex(x.itr, i)
Base.setindex(x::MathWrapper, value, i) =
    MathWrapper(Base.setindex(x.itr, value, i))

Base.show(io::IO, x::MathWrapper) = print(io, "MathWrapper(", x.itr, ")")
