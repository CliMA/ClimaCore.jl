import ForwardDiff
using UnrolledUtilities
using UnrolledUtilities: output_type_for_promotion, constructor_from_tuple

const MathMapperCompatibleType = Union{Number, AbstractArray, Tuple, NamedTuple}

"""
    MathMapper(itr)

Wrapper for an iterator that forces math functions (`+`, `*`, `zero`, etc.) to
be broadcasted over the iterator's elements. Aside from automatic broadcasting,
`MathMapper`s act exactly like their underlying iterators, with support for
common operations like `propertynames`, `getindex`, `getindex`, and `reduce`.
All statically-sized iterators for which [`supports_math_mapper`](@ref) is true
are compatible with `MathMapper`s.

In the context of `MathMapper`s, broadcasting a function corresponds to mapping
it over all arguments for which `supports_math_mapper` is true (including those
not wrapped in `MathMapper`s), while other arguments are passed to it directly.
This behavior is triggered when a `MathMapper` is used in the following ways:
  - passed to `map` or `mapreduce` (only up to 4 arguments)
  - passed to a math function from `Base` (all common single-output functions)
  - passed to `ifelse` (for mapping over conditional expressions)
  - applied as a function call (for mapping over multiple functions)

Nested `MathMapper`s evaluate broadcasts recursively, mapping down through every
nested layer and only applying broadcasted functions to non-`MathMapper`s in the
innermost layers. Reductions of nested `MathMapper`s are similarly evaluated
over non-`MathMapper`s in the innermost layers. The `MathMapper` constructor
only wraps outer layers of nested iterators, but [`nested_math_mapper`](@ref)
recursively wraps all layers that support `MathMapper`s.

# Examples
```jldoctest; setup = :(import ClimaCore.Utilities, ClimaCore.Geometry.StaticArrays)
julia> x = Utilities.MathMapper((1, 2.0, StaticArrays.SVector(3, 4)))
(1, 2.0, [3, 4])

julia> zero(typeof(x))
(0, 0.0, [0, 0])

julia> 2 * x - (2, 3, [4, 5])
(0, 1.0, [2, 3])

julia> y = Utilities.nested_math_mapper((a = 1, b = (2, 3, (d = 4, e = (5,)))))
(a = 1, b = (2, 3, (d = 4, e = (5,))))

julia> min(y, abs(4 - y))
(a = 1, b = (2, 1, (d = 0, e = (1,))))

julia> mapreduce(min, +, y, abs(4 - y))
5

julia> maximum(log2, y - one(y))
2.0
```
"""
struct MathMapper{I}
    itr::I
end

unwrap(::Type{MathMapper{I}}) where {I} = I
unwrap(x::MathMapper) = getfield(x, :itr)

"""
    supports_math_mapper(itr)

Indicates whether a [`MathMapper`](@ref) should map over the given iterator. By
default, this is true for all `Tuple`s and `NamedTuple`s.

Defining the method `supports_math_mapper(::T) = true` allows `MathMapper`s to
map over iterators of any type `T`, as long as those iterators are statically
sized and compatible with the unrolled functions in
[UnrolledUtilities.jl](https://github.com/CliMA/UnrolledUtilities.jl).
"""
supports_math_mapper(_) = false
supports_math_mapper(::Tuple) = true
supports_math_mapper(::NamedTuple) = true

"""
    nested_math_mapper(itr)

Recursively applies the [`MathMapper`](@ref) constructor to an iterator and all
of its elements for which [`supports_math_mapper`](@ref) is true. Values for
which `supports_math_mapper` is false are left unmodified.
"""
nested_math_mapper(itr) =
    supports_math_mapper(itr) ?
    MathMapper(unrolled_map(nested_math_mapper, itr)) : itr

"""
    nested_math_mapper_type(T)

Type of a [`nested_math_mapper`](@ref) constructed from an iterator of type `T`.
"""
nested_math_mapper_type(::Type{T}) where {T} =
    inferred_result_type(nested_math_mapper, T)

# Unrolled analogue of tuple.(args...) for MathMapper broadcast arguments
function zip_math_mapper_args(args...)
    zip_args = unrolled_map(arg -> arg isa MathMapper ? unwrap(arg) : arg, args)
    possible_zip_lengths =
        unrolled_map(length, unrolled_filter(supports_math_mapper, zip_args))
    if !unrolled_allequal(possible_zip_lengths)
        two_lengths = join(unique(possible_zip_lengths)[1:2], " and ")
        throw(DimensionMismatch("Arguments have unequal lengths $two_lengths"))
    end
    zip_length = possible_zip_lengths[1]
    lazily_extrude = Base.Fix2(Iterators.map, StaticOneTo(zip_length))
    extruded_zip_args =
        unrolled_map(zip_args) do arg
            supports_math_mapper(arg) ? arg : lazily_extrude(Returns(arg))
        end
    return unrolled_map(tuple, extruded_zip_args...)
end

"""
    math_mapper_broadcast(f, args...)

Generic form of `map(f, args...)` for [`MathMapper`](@ref) arguments, which
performs automatic broadcasting with recursion over nested iterators. The
built-in `map` only uses methods for `MathMapper`s when given arguments of
specific types, but this function works with any arguments.
"""
math_mapper_broadcast(f::F, args...) where {F} = _math_mapper_broadcast(f, args)
function _math_mapper_broadcast(f::F, args) where {F}
    unrolled_any(Base.Fix2(isa, MathMapper), args) || return f(args...)
    nested_call = Base.Fix1(_math_mapper_broadcast, f)
    zipped_args = zip_math_mapper_args(args...)
    return MathMapper(unrolled_map(nested_call, zipped_args))
end

"""
    math_mapper_type_broadcast(f, types...)

Similar to [`math_mapper_broadcast`](@ref), but for argument types rather than
the arguments themselves. Instead of applying `f` to elements of arguments for
which [`supports_math_mapper`](@ref) is true, this function applies `f` to the
types of those elements (i.e., to the `fieldtypes` of the argument types).
"""
math_mapper_type_broadcast(f::F, types...) where {F} =
    _math_mapper_type_broadcast(f, Val(Tuple{types...}))
function _math_mapper_type_broadcast(f::F, types_val) where {F}
    types = fieldtypes(val_parameter(types_val))
    unrolled_any(Base.Fix2(<:, MathMapper), types) || return f(types...)
    nested_call = Base.Fix1(_math_mapper_type_broadcast, f)
    zipped_args_type = inferred_result_type(zip_math_mapper_args, types...)
    zipped_types_vals = unrolled_map(Val, fieldtypes(zipped_args_type))
    get_constructor = constructor_from_tuple ∘ output_type_for_promotion
    constructor = inferred_result_value(get_constructor, zipped_args_type)
    return MathMapper(constructor(unrolled_map(nested_call, zipped_types_vals)))
end

"""
    reduce_math_mapper_broadcast(f, op, args...; init...)

Generic form of `mapreduce(f, op, args...; init...)` for [`MathMapper`](@ref)
arguments, which performs automatic broadcasting with recursion over nested
iterators. The built-in `mapreduce` only uses methods for `MathMapper`s when
given arguments of specific types, but this function works with any arguments.
"""
reduce_math_mapper_broadcast(f::F, op::O, args...; init...) where {F, O} =
    _reduce_math_mapper_broadcast((; f, op, init), args)
function _reduce_math_mapper_broadcast((; f, op, init), args)
    unrolled_any(Base.Fix2(isa, MathMapper), args) || return f(args...)
    nested_call = Base.Fix1(_reduce_math_mapper_broadcast, (; f, op, init))
    zipped_args = zip_math_mapper_args(args...)
    return unrolled_mapreduce(nested_call, op, zipped_args; init...)
end

"""
    reduce_math_mapper_type_broadcast(f, types...)

Similar to [`reduce_math_mapper_broadcast`](@ref), but for argument types rather
than the arguments themselves. Instead of applying `f` to elements of arguments
for which [`supports_math_mapper`](@ref) is true, this function applies `f` to
the types of those elements (i.e., to the `fieldtypes` of the argument types).
"""
reduce_math_mapper_type_broadcast(f::F, op::O, types...; init...) where {F, O} =
    _reduce_math_mapper_type_broadcast((; f, op, init), Val(Tuple{types...}))
function _reduce_math_mapper_type_broadcast((; f, op, init), types_val)
    types = fieldtypes(val_parameter(types_val))
    unrolled_any(Base.Fix2(<:, MathMapper), types) || return f(types...)
    nested_call = Base.Fix1(_reduce_math_mapper_type_broadcast, (; f, op, init))
    zipped_args_type = inferred_result_type(zip_math_mapper_args, types...)
    zipped_types_vals = unrolled_map(Val, fieldtypes(zipped_args_type))
    return unrolled_mapreduce(nested_call, op, zipped_types_vals; init...)
end

"""
    @math_mapper_method [T] <method definition>

Adds type constraints to all positional arguments in a method definition (not
`Vararg` or keyword arguments) that do not already have constrained types, so
that at least one argument is a [`MathMapper`](@ref) and the rest have type `T`.

If [`supports_math_mapper`](@ref) is true for values of type `T`, this ensures
they will be promoted to behave like `MathMapper`s in broadcast operations. If
it is false, this instead allows them to be used as single values. By default,
`T` is set to `Union{Number, AbstractArray, Tuple, NamedTuple}`, so that `Tuple`
and `NamedTuple` iterators are mapped over while `Number` and `AbstractArray`
values are treated like scalars.

To avoid ambiguity errors, a separate method must be defined for every distinct
permutation of `MathMapper`s and `T`s, so that the number of methods actually
generated by `@math_mapper_method` for ``N`` type constraints is ``2^N - 1``:

```julia
julia> @macroexpand @math_mapper_method NonMathMapper foo(x, y) = map(foo, x, y)
quote
    foo(x::MathMapper, y::MathMapper) = map(foo, x, y)
    foo(x::NonMathMapper, y::MathMapper) = map(foo, x, y)
    foo(x::MathMapper, y::NonMathMapper) = map(foo, x, y)
end
```
"""
macro math_mapper_method(args...)
    function call_expr(method_expr)
        call_and_where_expr = method_expr.args[1]
        while Meta.isexpr(call_and_where_expr, :where)
            call_and_where_expr = call_and_where_expr.args[1]
        end
        return call_and_where_expr
    end

    usage = ArgumentError("Usage: @math_mapper_method [T] <method definition>")
    1 <= length(args) <= 2 || throw(usage)
    method_expr = args[end]
    Meta.isexpr(method_expr, [:(=), :function]) || throw(usage)
    Meta.isexpr(call_expr(method_expr), :call) || throw(usage)
    method_arg_exprs = @view call_expr(method_expr).args[2:end]
    untyped_positional_arg = Base.Fix2(!Meta.isexpr, [:(::), :..., :parameters])
    type_expr_indices = findall(untyped_positional_arg, method_arg_exprs)
    N = length(type_expr_indices)

    Mod = @__MODULE__
    T_expr = length(args) == 1 ? :($Mod.MathMapperCompatibleType) : args[1]
    valid_types = (:($Mod.MathMapper), T_expr)
    new_methods_block_expr = quote end
    for type_exprs in Iterators.product(Iterators.repeated(valid_types, N)...)
        any(==(:($Mod.MathMapper)), type_exprs) || continue
        new_method_expr = copy(method_expr)
        new_method_arg_exprs = @view call_expr(new_method_expr).args[2:end]
        for (i, type_expr) in zip(type_expr_indices, type_exprs)
            new_method_arg_exprs[i] = :($(new_method_arg_exprs[i])::$type_expr)
        end
        push!(new_methods_block_expr.args, new_method_expr)
    end
    return esc(new_methods_block_expr)
end

# Limit map and mapreduce to 4 arguments to avoid generating too many methods
for n in 1:4
    args = map(Base.Fix1(Symbol, :arg), 1:n)
    @eval @math_mapper_method Base.map(f::F, $(args...)) where {F} =
        math_mapper_broadcast(f, $(args...))
    @eval @math_mapper_method Base.mapreduce(
        f::F,
        op::O,
        $(args...);
        init...,
    ) where {F, O} = reduce_math_mapper_broadcast(f, op, $(args...); init...)
end

# Commutative reduction operations whose fallback methods do not call mapreduce
Base.any(f::F, x::MathMapper) where {F} = mapreduce(f, |, x; init = false)
Base.all(f::F, x::MathMapper) where {F} = mapreduce(f, &, x; init = true)
Base.reduce(op::O, x::MathMapper; init...) where {O} =
    mapreduce(identity, op, x; init...)

##################################
## Automatic Broadcasting Rules ##
##################################

# Type functions extended in ForwardDiff.jl
Base.zero(::Type{T}) where {T <: MathMapper} =
    math_mapper_type_broadcast(zero, T)
Base.one(::Type{T}) where {T <: MathMapper} =
    math_mapper_type_broadcast(one, T)
Base.eps(::Type{T}) where {T <: MathMapper} =
    math_mapper_type_broadcast(eps, T)
Base.float(::Type{T}) where {T <: MathMapper} =
    math_mapper_type_broadcast(float, T)
Base.precision(::Type{T}; base...) where {T <: MathMapper} =
    math_mapper_type_broadcast(T -> precision(T; base...), T)

Base.zero(x::MathMapper) = zero(typeof(x))
Base.one(x::MathMapper) = one(typeof(x))
Base.eps(x::MathMapper) = eps(typeof(x))
Base.float(x::MathMapper) = float(typeof(x))
Base.precision(x::MathMapper; base...) = precision(typeof(x); base...)

# Boolean functions extended in ForwardDiff.jl
for f in ForwardDiff.UNARY_PREDICATES
    @eval Base.$f(x::MathMapper) = map(Base.$f, x)
end
@math_mapper_method Base.:<(x, y) = map(<, x, y)
@math_mapper_method Base.:<=(x, y) = map(<=, x, y)
@math_mapper_method Base.:(==)(x, y) = map(==, x, y)
@math_mapper_method Base.isless(x, y) = map(isless, x, y)
# FIXME: Adding methods for isequal causes invalidations
# @math_mapper_method Base.isequal(x, y) = map(isequal, x, y)

# Math functions extended in ForwardDiff.jl by looping over diffrules(), along
# with several methods of those functions that are missing from diffrules()
for (Mod, f, n) in ForwardDiff.DiffRules.diffrules()
    if isdefined(@__MODULE__, Mod) && isdefined(getfield(@__MODULE__, Mod), f)
        args = map(Base.Fix1(Symbol, :arg), 1:n)
        @eval @math_mapper_method $Mod.$f($(args...)) = map($Mod.$f, $(args...))
    end
end
Base.:*(x::MathMapper) = map(*, x)
Base.min(x::MathMapper) = map(min, x)
Base.max(x::MathMapper) = map(max, x)

# Other common math functions extended in ForwardDiff.jl, excluding those that
# return tuples instead of single values (specifically sincos and sincospi)
Base.Int(x::MathMapper) = map(Int, x)
Base.Integer(x::MathMapper) = map(Integer, x)
Base.nextfloat(x::MathMapper) = map(nextfloat, x)
Base.prevfloat(x::MathMapper) = map(prevfloat, x)
Base.exponent(x::MathMapper) = map(exponent, x)
Base.floor(x::MathMapper) = map(floor, x)
Base.ceil(x::MathMapper) = map(ceil, x)
Base.trunc(x::MathMapper) = map(trunc, x)
Base.round(x::MathMapper) = map(round, x)
Base.floor(::Type{T}, x::MathMapper) where {T} = map(Base.Fix1(floor, T), x)
Base.ceil(::Type{T}, x::MathMapper) where {T} = map(Base.Fix1(ceil, T), x)
Base.trunc(::Type{T}, x::MathMapper) where {T} = map(Base.Fix1(trunc, T), x)
Base.round(::Type{T}, x::MathMapper) where {T} = map(Base.Fix1(round, T), x)
Base.literal_pow(::typeof(^), x::MathMapper, p::Val) =
    map(x -> Base.literal_pow(^, x, p), x)
@math_mapper_method Base.fld(x, y) = map(fld, x, y)
@math_mapper_method Base.cld(x, y) = map(cld, x, y)
@math_mapper_method Base.div(x, y, r::RoundingMode) =
    map((x, y) -> div(x, y, r), x, y)
@math_mapper_method Base.fma(x, y, z) = map(fma, x, y, z)
@math_mapper_method Base.muladd(x, y, z) = map(muladd, x, y, z)

# Common math functions not extended in ForwardDiff.jl, excluding those that
# return tuples instead of single values (e.g., divrem, fldmod, and fldmod1)
Base.:!(x::MathMapper) = map(!, x)
Base.:&(x::MathMapper) = map(&, x)
Base.:|(x::MathMapper) = map(|, x)
Base.xor(x::MathMapper) = map(xor, x)
@math_mapper_method Base.:&(x, y) = map(&, x, y)
@math_mapper_method Base.:|(x, y) = map(|, x, y)
@math_mapper_method Base.xor(x, y) = map(xor, x, y)
@math_mapper_method Base.fld1(x, y) = map(fld1, x, y)
@math_mapper_method Base.mod1(x, y) = map(mod1, x, y)

# Functional form of if-else statements with MathMappers as conditionals
@math_mapper_method Base.ifelse(x::MathMapper, y, z) = map(ifelse, x, y, z)
# FIXME: Adding a method with only one MathMapper argument causes invalidations
# Base.ifelse(
#     x::MathMapper,
#     y::MathMapperCompatibleType,
#     z::MathMapperCompatibleType,
# ) = map(ifelse, x, y, z)

# MathMappers applied as function calls (implemented using math_mapper_broadcast
# instead of map, since map only works correctly up to 4 arguments)
(f::MathMapper)(args...) =
    math_mapper_broadcast((f, args...) -> f(args...), f, args...)

################################
## Automatic Unwrapping Rules ##
################################

Base.convert(::Type{T}, itr) where {T <: MathMapper} =
    MathMapper(convert(unwrap(T), itr))
Base.convert(::Type{T}, x::MathMapper) where {T <: MathMapper} =
    MathMapper(convert(unwrap(T), unwrap(x)))

Base.show(io::IO, x::MathMapper) = show(io, unwrap(x))
Base.propertynames(x::MathMapper) = propertynames(unwrap(x))
Base.getproperty(x::MathMapper, s::Symbol) = getproperty(unwrap(x), s)

Base.length(x::MathMapper) = length(unwrap(x))
Base.isempty(x::MathMapper) = isempty(unwrap(x))
Base.iterate(x::MathMapper, i...) = iterate(unwrap(x), i...)

Base.keys(x::MathMapper) = keys(unwrap(x))
Base.values(x::MathMapper) = values(unwrap(x))
Base.pairs(x::MathMapper) = pairs(unwrap(x))

Base.firstindex(x::MathMapper) = firstindex(unwrap(x))
Base.lastindex(x::MathMapper) = lastindex(unwrap(x))
Base.getindex(x::MathMapper, i) = getindex(unwrap(x), i)
Base.setindex(x::MathMapper, value, i) =
    MathMapper(Base.setindex(unwrap(x), value, i))
