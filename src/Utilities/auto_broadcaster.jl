using UnrolledUtilities
import ForwardDiff

# Default types that can be used as arguments to auto-broadcasted math functions
const DefaultBroadcastable = Union{Tuple, NamedTuple}
const DefaultExtrudable = Union{Number, AbstractArray}
const DefaultNonAutoBroadcaster = Union{DefaultBroadcastable, DefaultExtrudable}

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
  - manually broadcasting over them (only works with unwrapped iterators when
    broadcasting functions that have no more than 4 arguments)

Nested `AutoBroadcaster`s constructed with [`enable_auto_broadcasting`](@ref)
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

julia> y = Utilities.enable_auto_broadcasting((1, 2, (a = 3, b = 4, c = (5, 6, (7, 8)))))
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
unwrap(x) = getfield(x, :itr) # use getfield as getproperty is overwritten below

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
is_auto_broadcastable(itr) = is_auto_broadcastable(typeof(itr))
is_auto_broadcastable(::Type{<:DefaultBroadcastable}) = true
is_auto_broadcastable(::Type{Union{}}) = false # to resolve method ambiguities
is_auto_broadcastable(::Type) = false

"""
    enable_auto_broadcasting(itr)

Recursively applies the [`AutoBroadcaster`](@ref) constructor to iterators for
which [`is_auto_broadcastable`](@ref) is true, as well as their elements for
which it is true, while leaving values for which it is false unmodified.
"""
enable_auto_broadcasting(itr) =
    is_auto_broadcastable(itr) ?
    AutoBroadcaster(unrolled_map(enable_auto_broadcasting, itr)) : itr

"""
    disable_auto_broadcasting(x)

Recursively unwraps constructors applied by [`enable_auto_broadcasting`](@ref),
extracting the iterator from every [`AutoBroadcaster`](@ref) in `x`.
"""
disable_auto_broadcasting(x) =
    x isa AutoBroadcaster ?
    unrolled_map(disable_auto_broadcasting, unwrap(x)) : x

"""
    auto_broadcast(f)

Modifies a function so that it is preceded by [`enable_auto_broadcasting`](@ref)
and followed by [`disable_auto_broadcasting`](@ref) every time it is called.
"""
auto_broadcast(f::F) where {F} =
    (args...) -> disable_auto_broadcasting(
        f(unrolled_map(enable_auto_broadcasting, args)...),
    )

"""
    @auto_broadcaster_args [T] <method definition>

Adds type constraints to all positional arguments in a method definition (not
`Vararg` or keyword arguments) that do not already have constrained types, so at
least one argument is a [`AutoBroadcaster`](@ref) and the rest have type `T`.

If [`is_auto_broadcastable`](@ref) is true for a positional argument, that
argument will behave like an `AutoBroadcaster` when broadcasting. Otherwise,
that argument will be passed along as a single value. By default, `T` is a union
over `Tuple`s and `NamedTuple`s, whose elements are mapped over, as well as
`Number`s and `AbstractArray`s, which are treated like single values.

To avoid ambiguity errors, a separate method must be defined for every distinct
permutation of `AutoBroadcaster`s and `T`s. When adding ``N`` type constraints,
this macro generates ``2^N - 1`` methods; e.g., for 2 new type constraints:

```julia
julia> @macroexpand @auto_broadcaster_args foo(x, y) = foo.(x, y)
quote
    foo(x::AutoBroadcaster, y::AutoBroadcaster) = foo.(x, y)
    foo(x::DefaultNonAutoBroadcaster, y::AutoBroadcaster) = foo.(x, y)
    foo(x::AutoBroadcaster, y::DefaultNonAutoBroadcaster) = foo.(x, y)
end
```
"""
macro auto_broadcaster_args(args...)
    function call_expr(method_expr)
        call_and_where_expr = method_expr.args[1]
        while Meta.isexpr(call_and_where_expr, :where)
            call_and_where_expr = call_and_where_expr.args[1]
        end
        return call_and_where_expr
    end

    usage_error =
        ArgumentError("Usage: @auto_broadcaster_args [T] <method definition>")
    1 <= length(args) <= 2 || throw(usage_error)

    M = @__MODULE__
    T_expr = length(args) > 1 ? args[1] : :($M.DefaultNonAutoBroadcaster)
    valid_types = (:($M.AutoBroadcaster), T_expr)

    method_expr = args[end]
    Meta.isexpr(method_expr, [:(=), :function]) || throw(usage_error)
    Meta.isexpr(call_expr(method_expr), :call) || throw(usage_error)
    method_arg_exprs = @view call_expr(method_expr).args[2:end]
    untyped_positional_arg = Base.Fix2(!Meta.isexpr, [:(::), :..., :parameters])
    type_expr_indices = findall(untyped_positional_arg, method_arg_exprs)
    N = length(type_expr_indices)

    new_methods_block_expr = quote end
    for type_exprs in Iterators.product(Iterators.repeated(valid_types, N)...)
        any(!=(T_expr), type_exprs) || continue
        new_method_expr = copy(method_expr)
        new_method_arg_exprs = @view call_expr(new_method_expr).args[2:end]
        for (i, type_expr) in zip(type_expr_indices, type_exprs)
            new_method_arg_exprs[i] = :($(new_method_arg_exprs[i])::$type_expr)
        end
        push!(new_methods_block_expr.args, new_method_expr)
    end
    return esc(new_methods_block_expr)
end

###################################
## Automatic Wrapping/Unwrapping ##
###################################

for f in (:propertynames, :length, :size, :isempty, :keys, :values, :pairs)
    @eval Base.$f(x::AutoBroadcaster) = $f(unwrap(x))
end
Base.show(io::IO, x::AutoBroadcaster) = show(io, unwrap(x))
Base.iterate(x::AutoBroadcaster, i...) = iterate(unwrap(x), i...)
Base.getproperty(x::AutoBroadcaster, s::Symbol) = getproperty(unwrap(x), s)
Base.getindex(x::AutoBroadcaster, i) = getindex(unwrap(x), i)
Base.setindex(x::AutoBroadcaster, value, i) =
    AutoBroadcaster(Base.setindex(unwrap(x), value, i))

Base.Tuple(x::AutoBroadcaster) = AutoBroadcaster(Tuple(unwrap(x)))
Base.NamedTuple{names}(x::AutoBroadcaster) where {names} =
    AutoBroadcaster(NamedTuple{names}(unwrap(x)))

for f in (:mapfoldl, :mapfoldr)
    @eval Base.$f(f::F, op::O, x::AutoBroadcaster; init...) where {F, O} =
        $f(f, op, unwrap(x); init...)
end
Base.mapreduce(f::F, op::O, args::AutoBroadcaster...; init...) where {F, O} =
    mapreduce(f, op, unrolled_map(unwrap, args)...; init...)
Base.map(f::F, args::AutoBroadcaster...) where {F} =
    map(f, unrolled_map(unwrap, args)...)

##########################
## Automatic Converting ##
##########################

Base.convert(::Type{X}, x) where {X <: AutoBroadcaster} =
    nested_convert((Val(X), x))
Base.convert(::Type{I}, x::AutoBroadcaster) where {I <: DefaultBroadcastable} =
    nested_convert((Val(I), x))

# Nested iterators need an optimized form of convert for type inference on Julia
# 1.10, where all arguments are zipped to avoid splatting, and where types are
# wrapped in Vals to avoid relying on constant propagation
nested_convert((_, x)::Tuple{Val{X}, Any}) where {X <: AutoBroadcaster} =
    AutoBroadcaster(nested_convert((Val(unwrap(X)), x)))
function nested_convert((_, itr)::Tuple{Val{I}, Any}) where {I}
    unwrapped_itr = itr isa AutoBroadcaster ? unwrap(itr) : itr
    is_auto_broadcastable(I) || return convert(I, unwrapped_itr)
    lazy_fieldtype_vals =
        Iterators.map(Base.Fix1(Val ∘ fieldtype, I), StaticOneTo(fieldcount(I)))
    return unrolled_map(nested_convert, zip(lazy_fieldtype_vals, unwrapped_itr))
end

# This method resolves an ambiguity with the default Base.convert(X, x::X) = x
Base.convert(::Type{X}, x::X) where {X <: AutoBroadcaster} = x

#########################
## Manual Broadcasting ##
#########################

struct AutoBroadcastStyle <: Base.BroadcastStyle end

Base.broadcasted(::AutoBroadcastStyle, f::F, args...) where {F} =
    nested_broadcast(f, args)

# Nested iterators need an optimized form of broadcasted for type inference on
# Julia 1.10, where all arguments are zipped to avoid splatting
function nested_broadcast(f::F, args) where {F}
    unrolled_any(Base.Fix2(isa, AutoBroadcaster), args) || return f(args...)
    map_args = unrolled_map(x -> x isa AutoBroadcaster ? unwrap(x) : x, args)
    lengths =
        unrolled_map(length, unrolled_filter(is_auto_broadcastable, map_args))
    if !unrolled_allequal(lengths)
        lengths_str = join(unique(lengths), ", ", " and ")
        throw(DimensionMismatch("Arguments have unequal lengths $lengths_str"))
    end
    extrusion_axis = StaticOneTo(first(lengths))
    uniform_length_args = unrolled_map(map_args) do x
        is_auto_broadcastable(x) ? x : Iterators.map(Returns(x), extrusion_axis)
    end
    zipped_args = unrolled_map(tuple, uniform_length_args...)
    broadcast_result = unrolled_map(Base.Fix1(nested_broadcast, f), zipped_args)
    return AutoBroadcaster(broadcast_result)
end

# Broadcasting over AutoBroadcasters and scalars or arrays (which always use
# AbstractArrayStyles) triggers the AutoBroadcastStyle
Base.BroadcastStyle(::Type{<:AutoBroadcaster}) = AutoBroadcastStyle()
Base.BroadcastStyle(::AutoBroadcastStyle, ::Base.Broadcast.AbstractArrayStyle) =
    AutoBroadcastStyle()

# Other DefaultNonAutoBroadcasters also trigger the AutoBroadcastStyle, but only
# up to 4 arguments in total (supporting 4 arguments requires generating 26 new
# methods, supporting 5 requires 57, supporting up to 10 requires 2036, etc.)
for n in 1:4
    names = map(Base.Fix1(Symbol, :arg), 1:n)
    @eval @auto_broadcaster_args Base.broadcasted(f::F, $(names...)) where {F} =
        Base.broadcasted(AutoBroadcastStyle(), f, $(names...))
end

############################
## Automatic Broadcasting ##
############################

# Type functions extended in ForwardDiff.jl
for f in (:zero, :one, :eps, :float)
    @eval Base.$f(::Type{X}) where {X <: AutoBroadcaster} =
        broadcast_over_element_types($f, X)
end
Base.precision(::Type{X}; base...) where {X <: AutoBroadcaster} =
    broadcast_over_element_types(X -> precision(X; base...), X)
Base.promote_rule(
    ::Type{X},
    ::Type{Y},
) where {X <: AutoBroadcaster, Y <: AutoBroadcaster} =
    broadcast_result_type(first ∘ promote, X, Y)

# Boolean functions extended in ForwardDiff.jl
for f in ForwardDiff.UNARY_PREDICATES
    @eval Base.$f(x::AutoBroadcaster) = $f.(x)
end
for f in (:<, :<=, :(==), :isless) # FIXME: Adding :isequal causes invalidations
    @eval @auto_broadcaster_args Base.$f(x, y) = $f.(x, y)
end

# Other base math functions extended in ForwardDiff.jl (excluding those that
# return multiple values, like sincos or sincospi, so we can avoid needing to
# choose between AutoBroadcasters of Tuples and Tuples of AutoBroadcasters)
for (M, f, n) in ForwardDiff.DiffRules.diffrules()
    M == :Base || continue
    names = map(Base.Fix1(Symbol, :arg), 1:n)
    @eval @auto_broadcaster_args Base.$f($(names...)) = Base.$f.($(names...))
end
for f in (:zero, :one, :eps, :float, :nextfloat, :prevfloat, :exponent)
    @eval Base.$f(x::AutoBroadcaster) = $f.(x)
end
for f in (:floor, :ceil, :trunc, :round)
    @eval Base.$f(x::AutoBroadcaster) = $f.(x)
    @eval Base.$f(::Type{T}, x::AutoBroadcaster) where {T} =
        Base.Fix1($f, T).(x)
end
Base.precision(x::AutoBroadcaster; base...) = (x -> precision(x; base...)).(x)
Base.literal_pow(::typeof(^), x::AutoBroadcaster, p::Val) =
    (x -> Base.literal_pow(^, x, p)).(x)
@auto_broadcaster_args Base.div(x, y, r::RoundingMode) =
    ((x, y) -> div(x, y, r)).(x, y)
@auto_broadcaster_args Base.fld(x, y) = fld.(x, y)
@auto_broadcaster_args Base.cld(x, y) = cld.(x, y)
@auto_broadcaster_args Base.fma(x, y, z) = fma.(x, y, z)
@auto_broadcaster_args Base.muladd(x, y, z) = muladd.(x, y, z)

# Important Base math functions absent from ForwardDiff.jl (excluding those that
# return multiple values, like minmax or divrem/fldmod, so we can avoid needing
# to choose between AutoBroadcasters of Tuples and Tuples of AutoBroadcasters)
for f in (:!, :~, :adjoint, :angle, :cis, :cispi, :conj, :sign)
    @eval Base.$f(x::AutoBroadcaster) = $f.(x)
end
for f in (://, :&, :|, :xor, :fld1, :mod1, :add_sum, :mul_prod)
    @eval @auto_broadcaster_args Base.$f(x, y) = Base.$f.(x, y)
end

# Applying number constructors to AutoBroadcsters (ForwardDiff.jl extends the
# constructors for Int and Integer in this way, but not any for other types)
(::Type{T})(x::AutoBroadcaster) where {T <: Number} = T.(x)

# Using AutoBroadcasters as conditionals for if-else statements
@auto_broadcaster_args Base.ifelse(x::AutoBroadcaster, y, z) = ifelse.(x, y, z)
# FIXME: Adding a method with only one AutoBroadcaster causes invalidations:
# Base.ifelse(x::AutoBroadcaster, y::T, z::T) where {T <: DefaultNonAutoBroadcaster} =
#     ifelse.(x, y, z)

# Calling AutoBroadcasters like functions (this is implemented with the internal
# form of broadcasted to avoid constraints on the number or types of arguments)
(x::AutoBroadcaster)(y...) =
    Base.broadcasted(AutoBroadcastStyle(), (x, y...) -> x(y...), x, y...)
