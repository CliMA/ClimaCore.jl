using UnrolledUtilities
import ForwardDiff
import ForwardDiff: ForwardDiff, DiffRules

# Default types that can be used as arguments to auto-broadcasted math functions
const DefaultBroadcastable = Union{Tuple, NamedTuple}
const DefaultNonAutoBroadcaster =
    Union{DefaultBroadcastable, Number, AbstractArray}

"""
    AutoBroadcaster(itr)

Wrapper for an iterator that forces certain functions to be broadcasted over the
iterator's elements. This allows different types of broadcasting to be applied
simultaneously; e.g., ClimaCore's `Field`s and similar types use the standard
dot syntax to denote parallelized iteration over spatial locations, and they
wrap their values in `AutoBroadcaster`s for unrolled iteration over subfields.
All statically-sized iterators for which [`is_auto_broadcastable`](@ref) is true
are compatible with `AutoBroadcaster`s.

In the context of `AutoBroadcaster`s, broadcasting a function applies it with
[`unrolled_map`](https://clima.github.io/UnrolledUtilities.jl/dev/#Package-Features),
iterating over all arguments for which `is_auto_broadcastable` is true
(including those not wrapped in `AutoBroadcaster`s), while other arguments are
passed to the function directly. This behavior is triggered by using
`AutoBroadcaster`s, optionally in conjunction with compatible iterators that are
not wrapped in `AutoBroadcaster`s, in the following ways:
  - passing them to standard math functions or constructors
  - passing them to `ifelse` (for iterating over conditional values)
  - applying them as function calls (for iterating over functions)
  - explicitly broadcasting over them with [`nested_auto_broadcast`](@ref) or `@.`

Nested `AutoBroadcaster`s constructed with [`add_auto_broadcasters`](@ref)
evaluate broadcasts recursively, mapping across every layer of nested iterators
so that broadcasted functions are only applied to non-iterators in the innermost
layers. Aside from automatic broadcasting, `AutoBroadcaster`s are essentially
identical to their underlying iterators, with support for common operations like
`iterate`, `propertynames`, `getindex`, and `reduce`.

# Examples
```jldoctest; setup = :(import ClimaCore.Utilities, ClimaCore.Geometry.StaticArrays)
julia> x = Utilities.AutoBroadcaster((1, 2.0, StaticArrays.SVector(3, 4)))
(1, 2.0, [3, 4])

julia> zero(typeof(x))
(0, 0.0, [0, 0])

julia> 2 * x - (2, 3, [4, 5])
(0, 1.0, [2, 3])

julia> y = Utilities.add_auto_broadcasters((1, 2, (a = 3, b = 4, c = (5, 6, (7, 8)))))
(1, 2, (a = 3, b = 4, c = (5, 6, (7, 8))))

julia> min(y, abs(5 - y))
(1, 2, (a = 2, b = 1, c = (0, 1, (2, 3))))

julia> x' * y * x ÷ 5
(0, 1.0, (a = 15, b = 20, c = (25, 30, (35, 40))))
```
"""
struct AutoBroadcaster{I}
    itr::I
end

unwrap(::Type{AutoBroadcaster{I}}) where {I} = I
unwrap(x::AutoBroadcaster) = getfield(x, :itr) # getproperty is overwritten below
unwrap(x) = x

"""
    is_auto_broadcastable(::Type)
    is_auto_broadcastable(itr)

Indicates whether an [`AutoBroadcaster`](@ref) should broadcast over iterators
of the given type. By default, this is only true for `Tuple` and `NamedTuple`
types, but it can be extended to any statically-sized type compatible with
[UnrolledUtilities.jl](https://github.com/CliMA/UnrolledUtilities.jl).

For convenience, `is_auto_broadcastable` also supports passing a concrete
iterator instead of its type, but this method should not be extended directly.
"""
is_auto_broadcastable(::Type{<:DefaultBroadcastable}) = true
is_auto_broadcastable(::Type) = false
is_auto_broadcastable(::Type{Union{}}) = false # to resolve ambiguity
is_auto_broadcastable(itr) = is_auto_broadcastable(typeof(itr))

"""
    add_auto_broadcasters(itr)
    add_auto_broadcasters(I)

Recursively applies the [`AutoBroadcaster`](@ref) constructor to iterators for
which [`is_auto_broadcastable`](@ref) is true, as well as their elements for
which it is true, while leaving values for which it is false unmodified. Can
also be passed an iterator's type to infer the result type for such an iterator.
"""
add_auto_broadcasters(itr) =
    is_auto_broadcastable(itr) ?
    AutoBroadcaster(unrolled_map(add_auto_broadcasters, itr)) : itr
add_auto_broadcasters(::Type{I}) where {I} =
    Core.Compiler.return_type(add_auto_broadcasters, Tuple{I})

"""
    drop_auto_broadcasters(x)
    drop_auto_broadcasters(X)

Recursively unwraps constructors applied by [`add_auto_broadcasters`](@ref),
extracting the iterator from every [`AutoBroadcaster`](@ref) in `x`. Can also be
passed an iterator's type to infer the result type for such an iterator.
"""
drop_auto_broadcasters(x) =
    x isa AutoBroadcaster ? unrolled_map(drop_auto_broadcasters, unwrap(x)) : x
drop_auto_broadcasters(::Type{X}) where {X} =
    Core.Compiler.return_type(drop_auto_broadcasters, Tuple{X})

"""
    EnableAutoBroadcasting(f)

Wrapper for a function `f` that calls either [`add_auto_broadcasters`](@ref)
or [`drop_auto_broadcasters`](@ref) on its arguments if doing so may prevent
an error from being thrown.

When `EnableAutoBroadcasting(f)(args...)` is evaluated, if the `return_type` of
`f(args...)` indicates that an error will be thrown, the `return_type` is
recomputed after `add_auto_broadcasters` is called on every argument, and then
again after `drop_auto_broadcasters` is called on every argument. If either of
the new results no longer corresponds to a guaranteed error, `f` is applied
to the modified arguments. Otherwise, `f` is applied to its original arguments.
"""
struct EnableAutoBroadcasting{F}
    f::F
end

# Special handling of Type values, which cannot be inferred from their types
EnableAutoBroadcasting(::Type{T}) where {T} = EnableAutoBroadcasting{Type{T}}(T)

function ((; f)::EnableAutoBroadcasting)(args...)
    Core.Compiler.return_type(f, typeof(args)) != Union{} && return f(args...)
    args′ = unrolled_map(add_auto_broadcasters, args)
    Core.Compiler.return_type(f, typeof(args′)) != Union{} && return f(args′...)
    args′′ = unrolled_map(drop_auto_broadcasters, args)
    Core.Compiler.return_type(f, typeof(args′′)) != Union{} && return f(args′′...)
    return f(args...) # Error is not caused by missing or extra AutoBroadcasters
end

#########################################
## Automatic Unwrapping and Rewrapping ##
#########################################

Base.eltype(::Type{X}) where {X <: AutoBroadcaster} = eltype(unwrap(X))

for f in (:Tuple, :isempty, :length, :propertynames, :keys, :values, :pairs)
    @eval Base.$f(x::AutoBroadcaster) = $f(unwrap(x))
end
Base.NamedTuple{names}(x::AutoBroadcaster) where {names} =
    NamedTuple{names}(unwrap(x))
Base.show(io::IO, x::AutoBroadcaster) = show(io, unwrap(x))
Base.iterate(x::AutoBroadcaster, state...) = iterate(unwrap(x), state...)
Base.getproperty(x::AutoBroadcaster, name::Symbol) =
    getproperty(unwrap(x), name)
Base.merge(args::AutoBroadcaster...) =
    AutoBroadcaster(merge(unrolled_map(unwrap, args)...))

for f in (:axes, :size, :firstindex, :lastindex)
    @eval Base.$f(x::AutoBroadcaster) = $f(unwrap(x))
end
Base.axes(x::AutoBroadcaster, dim::Integer) = axes(unwrap(x), dim)
Base.size(x::AutoBroadcaster, dim::Integer) = size(unwrap(x), dim)
Base.@propagate_inbounds Base.getindex(x::AutoBroadcaster, index) =
    getindex(unwrap(x), index)
Base.@propagate_inbounds Base.setindex(x::AutoBroadcaster, value, index) =
    AutoBroadcaster(Base.setindex(unwrap(x), value, index))

#######################################
## Automatic Unrolling and Recursion ##
#######################################

Base.mapreduce(
    f::F,
    op::O,
    arg::AutoBroadcaster,
    args::AutoBroadcaster...;
    init...,
) where {F, O} =
    unrolled_mapreduce(f, op, unwrap(arg), unrolled_map(unwrap, args)...; init...)
Base.map(f::F, arg::AutoBroadcaster, args::AutoBroadcaster...) where {F} =
    AutoBroadcaster(unrolled_map(f, unwrap(arg), unrolled_map(unwrap, args)...))

# The built-in convert is unstable for nested Tuples/NamedTuples on Julia 1.10
nested_auto_convert(::Type{T}, arg) where {T} =
    _nested_auto_convert((new(T), arg))

# Turn types into values and zip the arguments to guarantee recursive inlining
_nested_auto_convert((x, y)) =
    x isa AutoBroadcaster ?
    AutoBroadcaster(_nested_auto_convert((unwrap(x), y))) :
    is_auto_broadcastable(x) ?
    unrolled_map(_nested_auto_convert, unrolled_map(tuple, x, unwrap(y))) :
    convert(typeof(x), unwrap(y))

Base.convert(::Type{I}, x::AutoBroadcaster) where {I <: DefaultBroadcastable} =
    nested_auto_convert(I, x)
Base.convert(::Type{X}, itr) where {X <: AutoBroadcaster} =
    nested_auto_convert(X, itr)
Base.convert(::Type{X}, x::X) where {X <: AutoBroadcaster} = x # to resolve ambiguity

"""
    nested_auto_broadcast(f, args...)

Analogue of `broadcast` that is applied recursively over nested iterators, as
long as at least one argument is an [`AutoBroadcaster`](@ref). This is always
the case for broadcast expressions that only contain `AutoBroadcaster`s and
scalars, arrays, or tuples; otherwise, this function must be used instead.
"""
nested_auto_broadcast(f::F, args...) where {F} = _nested_auto_broadcast(f, args)

# Zip the arguments instead of splatting them to guarantee recursive inlining
function _nested_auto_broadcast(f::F, args) where {F}
    unrolled_any(Base.Fix2(isa, AutoBroadcaster), args) || return f(args...)
    unwrapped_args = unrolled_map(unwrap, args)
    broadcastable_args = unrolled_filter(is_auto_broadcastable, unwrapped_args)
    lengths = unrolled_map(length, broadcastable_args)
    if !unrolled_allequal(lengths)
        lengths_str = join(unique(lengths), ", ", " and ")
        throw(DimensionMismatch("Arguments have unequal lengths $lengths_str"))
    end
    broadcast_axis = StaticOneTo(first(lengths))
    uniform_length_args = unrolled_map(unwrapped_args) do x
        is_auto_broadcastable(x) ? x : Iterators.map(Returns(x), broadcast_axis)
    end
    zipped_args = unrolled_map(tuple, uniform_length_args...)
    result_itr = unrolled_map(Base.Fix1(_nested_auto_broadcast, f), zipped_args)
    return AutoBroadcaster(result_itr)
end

apply(f::F, args...) where {F} = f(args...)

# If f is a type, wrap it in a Base.Fix1 struct to avoid type instabilities
nested_auto_broadcast(::Type{T}, args...) where {T} =
    nested_auto_broadcast(Base.Fix1(apply, T), args...)

struct NestedAutoStyle <: Base.BroadcastStyle end
Base.broadcasted(::NestedAutoStyle, f::F, args...) where {F} =
    nested_auto_broadcast(f, args...)

Base.broadcastable(x::AutoBroadcaster) = x

# Combining AutoBroadcasters with either scalars or arrays (which always use an
# AbstractArrayStyle) or tuples automatically triggers nested_auto_broadcast
Base.BroadcastStyle(::Type{<:AutoBroadcaster}) = NestedAutoStyle()
Base.BroadcastStyle(::NestedAutoStyle, ::Base.Broadcast.AbstractArrayStyle) =
    NestedAutoStyle()
Base.BroadcastStyle(::NestedAutoStyle, ::Base.Broadcast.Style{Tuple}) =
    NestedAutoStyle()

############################
## Automatic Broadcasting ##
############################

function type_constraints(n)
    available_types = (:AutoBroadcaster, :DefaultNonAutoBroadcaster)
    permutations = Iterators.product(map(Returns(available_types), 1:n)...)
    return Iterators.filter(Base.Fix1(any, ==(:AutoBroadcaster)), permutations)
end

# Boolean functions extended in ForwardDiff.jl
for f in ForwardDiff.UNARY_PREDICATES
    @eval Base.$f(x::AutoBroadcaster) = nested_auto_broadcast($f, x)
end
for f in (:<, :<=, :(==), :isless), (X, Y) in type_constraints(2)
    @eval Base.$f(x::$X, y::$Y) = nested_auto_broadcast($f, x, y)
end # FIXME: Adding a method for isequal causes invalidations

# Continuously differentiable functions from Base extended in ForwardDiff.jl
const diff_rules_from_base =
    Iterators.filter(==(:Base) ∘ first, ForwardDiff.DiffRules.diffrules())
for (_, f, n) in diff_rules_from_base, types in type_constraints(n)
    args = Base.mapany(Base.Fix1(Symbol, :arg), 1:n)
    typed_args = Base.mapany(((arg, type),) -> :($arg::$type), zip(args, types))
    @eval Base.$f($(typed_args...)) = nested_auto_broadcast(Base.$f, $(args...))
end

# Other math functions from Base extended in ForwardDiff.jl, excluding those
# that return multiple values (e.g., sincos or sincospi), so we avoid having to
# choose between an AutoBroadcaster of Tuples and a Tuple of AutoBroadcasters
for f in (:zero, :one, :eps, :float, :nextfloat, :prevfloat, :exponent)
    @eval Base.$f(x::AutoBroadcaster) = nested_auto_broadcast($f, x)
end
for f in (:floor, :ceil, :trunc, :round)
    @eval Base.$f(x::AutoBroadcaster) = nested_auto_broadcast($f, x)
    @eval Base.$f(::Type{T}, x::AutoBroadcaster) where {T} =
        nested_auto_broadcast(Base.Fix1($f, T), x)
end
(::Type{T})(x::AutoBroadcaster) where {T <: Integer} =
    nested_auto_broadcast(T, x)
Base.precision(x::AutoBroadcaster; base...) =
    nested_auto_broadcast(x -> precision(x; base...), x)
Base.literal_pow(::typeof(^), x::AutoBroadcaster, p::Val) =
    nested_auto_broadcast(x -> Base.literal_pow(^, x, p), x)
for (X, Y) in type_constraints(2)
    @eval Base.div(x::$X, y::$Y, r::RoundingMode) =
        nested_auto_broadcast((x, y) -> div(x, y, r), x, y)
    @eval Base.fld(x::$X, y::$Y) = nested_auto_broadcast(fld, x, y)
    @eval Base.cld(x::$X, y::$Y) = nested_auto_broadcast(cld, x, y)
end
for (X, Y, Z) in type_constraints(3)
    @eval Base.fma(x::$X, y::$Y, z::$Z) = nested_auto_broadcast(fma, x, y, z)
    @eval Base.muladd(x::$X, y::$Y, z::$Z) =
        nested_auto_broadcast(muladd, x, y, z)
end

# Common math functions absent from ForwardDiff.jl, excluding those that return
# multiple values (e.g., minmax, divrem, or fldmod), so we avoid having to
# choose between an AutoBroadcaster of Tuples and a Tuple of AutoBroadcasters
for f in (:!, :~, :adjoint, :angle, :cis, :cispi, :conj, :sign)
    @eval Base.$f(x::AutoBroadcaster) = nested_auto_broadcast($f, x)
end
for f in (://, :&, :|, :xor, :fld1, :mod1), (X, Y) in type_constraints(2)
    @eval Base.$f(x::$X, y::$Y) = nested_auto_broadcast($f, x, y)
end

# Passing nested AutoBroadcasters/DefaultBroadcastables to Base.sum or Base.prod 
for f in (:add_sum, :mul_prod), (X, Y) in type_constraints(2)
    @eval Base.$f(x::$X, y::$Y) = nested_auto_broadcast(Base.$f, x, y)
end

# Using AutoBroadcasters/DefaultBroadcastables as if-else statement conditionals
for C in (:AutoBroadcaster, :DefaultBroadcastable), (X, Y) in type_constraints(2)
    @eval Base.ifelse(cond::$C, x::$X, y::$Y) =
        nested_auto_broadcast(ifelse, cond, x, y)
end

# Applying AutoBroadcasters like functions
(f::AutoBroadcaster)(args...) = nested_auto_broadcast(apply, f, args...)

#######################################
## Automatic Broadcasting over Types ##
#######################################

# Value of f.(typeof.(arg1), typeof.(arg2), ...) for arguments of the given types
nested_auto_broadcast_over_argument_types(f::F, types...) where {F} =
    nested_auto_broadcast(unrolled_map(new, types)...) do args...
        f(unrolled_map(typeof, args)...)
    end

# Type of f_value.(arg1, arg2, ...) for arguments of the given types, where the
# function f_value satisfies f_value(x, y, ...) isa f(typeof(x), typeof(y), ...)
nested_auto_broadcast_result_type(f::F, types...) where {F} =
    typeof(nested_auto_broadcast_over_argument_types(new ∘ f, types...))

# Type functions extended in ForwardDiff.jl
for X in (:AutoBroadcaster,), Y in (:AutoBroadcaster, :DefaultBroadcastable)
    @eval Base.promote_rule(::Type{X}, ::Type{Y}) where {X <: $X, Y <: $Y} =
        nested_auto_broadcast_result_type(Base.promote_type, X, Y)
end # Only need one permutation for each pair of types passed to promote_rule
for f in (:zero, :one, :eps, :float)
    @eval Base.$f(::Type{X}) where {X <: AutoBroadcaster} =
        nested_auto_broadcast_over_argument_types($f, X)
end
Base.precision(::Type{X}; base...) where {X <: AutoBroadcaster} =
    nested_auto_broadcast_over_argument_types(x -> precision(x; base...), X)
