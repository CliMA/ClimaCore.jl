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
  - explicitly calling [`nested_broadcast`](@ref)

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
    add_auto_broadcasters(::Type)

Recursively applies the [`AutoBroadcaster`](@ref) constructor to iterators for
which [`is_auto_broadcastable`](@ref) is true, as well as their elements for
which it is true, while leaving values for which it is false unmodified. Can
also be passed an iterator's type to infer the result type for such an iterator.
"""
add_auto_broadcasters(itr) =
    itr isa AutoBroadcaster || is_auto_broadcastable(itr) ?
    AutoBroadcaster(unrolled_map(add_auto_broadcasters, unwrap(itr))) : itr
add_auto_broadcasters(::Type{T}) where {T} =
    Core.Compiler.return_type(add_auto_broadcasters, Tuple{T})

"""
    drop_auto_broadcasters(itr)
    drop_auto_broadcasters(::Type)

Recursively unwraps constructors applied by [`add_auto_broadcasters`](@ref),
extracting the iterator from every [`AutoBroadcaster`](@ref) in `itr`. Can also
be passed an iterator's type to infer the result type for such an iterator.
"""
drop_auto_broadcasters(itr) =
    itr isa AutoBroadcaster || is_auto_broadcastable(itr) ?
    unrolled_map(drop_auto_broadcasters, unwrap(itr)) : itr
drop_auto_broadcasters(::Type{T}) where {T} =
    Core.Compiler.return_type(drop_auto_broadcasters, Tuple{T})

"""
    auto_broadcasted([style], f, args, [axes])

Analogue of `Base.Broadcast.Broadcasted(style, f, args, axes)` that can pass the
arguments of `f` through either [`add_auto_broadcasters`](@ref) or
[`drop_auto_broadcasters`](@ref) if doing so will help avoid an inferred error.

When the [`unsafe_eltype`](@ref) of `Broadcasted(style, f, args, axes)`
indicates that `f` will throw an error, a new `Broadcasted` wrapper is
constructed with `add_auto_broadcasters` applied to every argument, and then
another is constructed with `drop_auto_broadcasters` applied to every argument.
If one of the new wrappers no longer corresponds to a guaranteed error, it is
returned instead of the original wrapper. Otherwise, the default result of
`Broadcasted(style, f, args, axes)` is returned without modifications.

# Examples
```jldoctest; setup = :(import ClimaCore.Utilities), filter = r"\\{.+\\}"
julia> x = (im, (1, 2.0), [3, 4])
(im, (1, 2.0), [3, 4])

julia> y = [x, x, x, x];

julia> bc = Base.Broadcast.Broadcasted(*, (Base.Broadcast.Broadcasted(adjoint, (y,)), y));

julia> sum(Base.materialize(bc))
ERROR: MethodError: no method matching adjoint(::Tuple{...})
[...]

julia> bc = Utilities.auto_broadcasted(*, (Utilities.auto_broadcasted(adjoint, (y,)), y));

julia> sum(Base.materialize(bc))
(4 + 0im, (4, 16.0), 100)
```
"""
auto_broadcasted(f::F, args, axes...) where {F} =
    auto_broadcasted(Base.Broadcast.combine_styles(args...), f, args, axes...)
function auto_broadcasted(style::Base.BroadcastStyle, f::F, args, axes...) where {F}
    wrapped_f(args...) = f(unrolled_map(add_auto_broadcasters, args)...)
    unwrapped_f(args...) = f(unrolled_map(drop_auto_broadcasters, args)...)
    bc = Base.Broadcast.Broadcasted(style, f, args, axes...)
    unsafe_eltype(bc) != Union{} && return bc
    bc′ = Base.Broadcast.Broadcasted(style, wrapped_f, args, axes...)
    unsafe_eltype(bc′) != Union{} && return bc′
    bc′′ = Base.Broadcast.Broadcasted(style, unwrapped_f, args, axes...)
    unsafe_eltype(bc′′) != Union{} && return bc′′
    return bc # error in bc is not caused by missing or extra AutoBroadcasters
end

"""
    nested_broadcast(f, args...)

Analogue of `broadcast` that is applied recursively over nested iterators, as
long as at least one argument is an [`AutoBroadcaster`](@ref). All loops over
iterator elements are unrolled and inlined to optimize performance.

This function is automatically called when an [`AutoBroadcaster`](@ref) is
passed to any standard math function or constructor, but for generic operations
it must be called explicitly.

# Examples
```jldoctest; setup = :(import ClimaCore.Utilities)
julia> x = Utilities.add_auto_broadcasters(((:a, :b, :c), (:d, :e, :f), :g))
((:a, :b, :c), (:d, :e, :f), :g)

julia> Utilities.nested_broadcast(string, x)
(("a", "b", "c"), ("d", "e", "f"), "g")

julia> y = Utilities.add_auto_broadcasters((1, 11, (111, 1111, 11111)))
(1, 11, (111, 1111, 11111))

julia> Utilities.nested_broadcast(Symbol, x, y * y)
((:a1, :b1, :c1), (:d121, :e121, :f121), (:g12321, :g1234321, :g123454321))
```
"""
nested_broadcast(f::F, args...) where {F} = _nested_broadcast(f, args)

# Zip the arguments instead of splatting them to guarantee recursive inlining
function _nested_broadcast(f::F, args) where {F}
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
    result_itr = unrolled_map(Base.Fix1(_nested_broadcast, f), zipped_args)
    return AutoBroadcaster(result_itr)
end

# Wrap each Type in a struct to guarantee recursive inlining
nested_broadcast(::Type{T}, args...) where {T} =
    nested_broadcast(Base.Fix1((T, args) -> T(args...), T) ∘ tuple, args...)

# Nested version of f.(typeof.(x), typeof.(y), ...) for x::type1, y::type2, etc.
nested_broadcast_over_types(f::F, types...) where {F} = nested_broadcast(
    (args...) -> f(unrolled_map(typeof, args)...),
    unrolled_map(new, types)...,
)

# Nested version of typeof(new.(f.(typeof.(x), typeof.(y), ...))) for x::type1...
nested_broadcast_result_type(f::F, types...) where {F} =
    typeof(nested_broadcast_over_types((types...) -> new(f(types...)), types...))

#########################################
## Automatic Unwrapping and Rewrapping ##
#########################################

Base.eltype(::Type{X}) where {X <: AutoBroadcaster} = eltype(unwrap(X))

Base.Tuple(x::AutoBroadcaster) = Tuple(unwrap(x))
Base.NamedTuple{names}(x::AutoBroadcaster) where {names} =
    NamedTuple{names}(unwrap(x))

Base.propertynames(x::AutoBroadcaster) = propertynames(unwrap(x))
Base.getproperty(x::AutoBroadcaster, name::Symbol) = getproperty(unwrap(x), name)

for f in (:keys, :values, :pairs, :isempty, :length, :firstindex, :lastindex)
    @eval Base.$f(x::AutoBroadcaster) = $f(unwrap(x))
end
Base.show(io::IO, x::AutoBroadcaster) = show(io, unwrap(x))
Base.axes(x::AutoBroadcaster, dim...) = axes(unwrap(x), dim...)
Base.size(x::AutoBroadcaster, dim...) = size(unwrap(x), dim...)
Base.iterate(x::AutoBroadcaster, state...) = iterate(unwrap(x), state...)
Base.merge(args::AutoBroadcaster...) =
    AutoBroadcaster(merge(unrolled_map(unwrap, args)...))
Base.@propagate_inbounds Base.getindex(x::AutoBroadcaster, index) =
    getindex(unwrap(x), index)
Base.@propagate_inbounds Base.setindex(x::AutoBroadcaster, value, index) =
    AutoBroadcaster(Base.setindex(unwrap(x), value, index))

# Broadcasts/maps/reductions are not recursive, unlike the math operations below
Base.broadcastable(x::AutoBroadcaster) = Base.broadcastable(unwrap(x))
Base.map(f::F, arg::AutoBroadcaster, args::AutoBroadcaster...) where {F} =
    AutoBroadcaster(map(f, unwrap(arg), unrolled_map(unwrap, args)...))
Base.mapreduce(
    f::F,
    op::O,
    arg::AutoBroadcaster,
    args::AutoBroadcaster...;
    init...,
) where {F, O} =
    mapreduce(f, op, unwrap(arg), unrolled_map(unwrap, args)...; init...)

# Circumvent the built-in convert function, which can introduce type
# instabilities for nested Tuples and NamedTuples on Julia 1.10
Base.convert(::Type{X}, x::X) where {X <: AutoBroadcaster} = x
Base.convert(::Type{I}, x::AutoBroadcaster) where {I <: DefaultBroadcastable} =
    nested_convert(I, x)
Base.convert(::Type{X}, itr) where {X <: AutoBroadcaster} =
    nested_convert(X, itr)
nested_convert(::Type{T}, arg) where {T} = _nested_convert((new(T), arg))

# Turn types into values and zip the arguments to guarantee recursive inlining
_nested_convert((x, y)) =
    x isa AutoBroadcaster ? AutoBroadcaster(_nested_convert((unwrap(x), y))) :
    is_auto_broadcastable(x) ?
    unrolled_map(_nested_convert, unrolled_map(tuple, x, unwrap(y))) :
    convert(typeof(x), unwrap(y))

###############################################
## Automatic Broadcasting of Math Operations ##
###############################################

const AutoBroadcasterOrSimilar = Union{AutoBroadcaster, DefaultBroadcastable}

# Type functions extended in ForwardDiff.jl
for f in (:zero, :one, :eps, :float)
    @eval Base.$f(::Type{X}) where {X <: AutoBroadcaster} =
        nested_broadcast_over_types($f, X)
end
Base.precision(::Type{X}; base...) where {X <: AutoBroadcaster} =
    nested_broadcast_over_types(x -> precision(x; base...), X)
Base.promote_rule(
    ::Type{X},
    ::Type{Y},
) where {X <: AutoBroadcaster, Y <: AutoBroadcasterOrSimilar} =
    nested_broadcast_result_type(Base.promote_type, X, Y)

# Common type functions absent from ForwardDiff.jl
for f in (:big, :real, :complex, :widen)
    @eval Base.$f(::Type{X}) where {X <: AutoBroadcaster} =
        nested_broadcast_result_type($f, X)
end

# Types of constructors for subtypes of T that have an unconstrained argument,
# leading to ambiguities with the method (::Type{<:T})(::AutoBroadcaster) = ...
function ambiguous_constructor_types(T)
    types = []
    if isabstracttype(T)
        for T_subtype in InteractiveUtils.subtypes(T)
            append!(types, ambiguous_constructor_types(T_subtype))
        end
    end
    vars = []
    empty_var = TypeVar(Symbol())
    while true
        new_type = reduce((T, var) -> UnionAll(var, T), vars; init = Type{T})
        constructor = reduce((T, _) -> UnionAll(empty_var, T), vars; init = T)
        hasmethod(constructor, Tuple{AutoBroadcaster}) && push!(types, new_type)
        T isa DataType && break
        push!(vars, T.var)
        T = T.body
    end
    return types
end

# All Number constructors (only defined for Integer and Dual in ForwardDiff.jl),
# with constructors for a few subtypes defined separately to avoid ambiguities
for constructor_type in ambiguous_constructor_types(Number)
    @eval (T::$constructor_type)(x::AutoBroadcaster) = nested_broadcast(T, x)
end
(T::Type{<:Number})(x::AutoBroadcaster) = nested_broadcast(T, x)

# Permutations of n type constraints that include at least one :AutoBroadcaster
function constraint_permutations(n)
    all_constraint_names = (:AutoBroadcaster, :DefaultNonAutoBroadcaster)
    permutations = Iterators.product(map(Returns(all_constraint_names), 1:n)...)
    return Iterators.filter(Base.Fix1(any, ==(:AutoBroadcaster)), permutations)
end

# Boolean functions extended in ForwardDiff.jl
for f in ForwardDiff.UNARY_PREDICATES
    @eval Base.$f(x::AutoBroadcaster) = nested_broadcast($f, x)
end
for f in (:<, :<=, :(==), :isless), (X, Y) in constraint_permutations(2)
    @eval Base.$f(x::$X, y::$Y) = nested_broadcast($f, x, y)
end # FIXME: Adding a method for isequal here causes invalidations

# Continuously differentiable functions from Base extended in ForwardDiff.jl
const base_function_diff_rules =
    Iterators.filter(==(:Base) ∘ first, ForwardDiff.DiffRules.diffrules())
for (_, f, n) in base_function_diff_rules, types in constraint_permutations(n)
    args = map(Base.Fix1(Symbol, :arg), 1:n)
    typed_args = map(((arg, type),) -> :($arg::$type), zip(args, types))
    @eval Base.$f($(typed_args...)) = nested_broadcast(Base.$f, $(args...))
end

# Other math functions from Base extended in ForwardDiff.jl, excluding those
# that return pairs of values (e.g., sincos or sincospi), so we avoid having to
# distinguish a Tuple of 2 AutoBroadcasters from an AutoBroadcaster of 2 Tuples
for f in (:zero, :one, :eps, :float, :nextfloat, :prevfloat, :exponent)
    @eval Base.$f(x::AutoBroadcaster) = nested_broadcast($f, x)
end
for f in (:floor, :ceil, :trunc, :round)
    @eval Base.$f(x::AutoBroadcaster) = nested_broadcast($f, x)
    @eval Base.$f(::Type{T}, x::AutoBroadcaster) where {T} =
        nested_broadcast(Base.Fix1($f, T), x)
end
Base.precision(x::AutoBroadcaster; base...) =
    nested_broadcast(x -> precision(x; base...), x)
Base.literal_pow(::typeof(^), x::AutoBroadcaster, p::Val) =
    nested_broadcast(x -> Base.literal_pow(^, x, p), x)
for (X, Y) in constraint_permutations(2)
    @eval Base.div(x::$X, y::$Y, r::RoundingMode) =
        nested_broadcast((x, y) -> div(x, y, r), x, y)
    @eval Base.fld(x::$X, y::$Y) = nested_broadcast(fld, x, y)
    @eval Base.cld(x::$X, y::$Y) = nested_broadcast(cld, x, y)
end
for (X, Y, Z) in constraint_permutations(3)
    @eval Base.fma(x::$X, y::$Y, z::$Z) = nested_broadcast(fma, x, y, z)
    @eval Base.muladd(x::$X, y::$Y, z::$Z) =
        nested_broadcast(muladd, x, y, z)
end

# Common math functions absent from ForwardDiff.jl, excluding those that return
# pairs of values (e.g., minmax, divrem, or fldmod), so we avoid having to
# distinguish a Tuple of 2 AutoBroadcasters from an AutoBroadcaster of 2 tuples
for f in (:!, :~, :adjoint, :angle, :cis, :cispi, :conj, :sign)
    @eval Base.$f(x::AutoBroadcaster) = nested_broadcast($f, x)
end
for f in (://, :&, :|, :xor, :fld1, :mod1), (X, Y) in constraint_permutations(2)
    @eval Base.$f(x::$X, y::$Y) = nested_broadcast($f, x, y)
end

# Internal functions called by Base.sum and Base.prod
for f in (:add_sum, :mul_prod), (X, Y) in constraint_permutations(2)
    @eval Base.$f(x::$X, y::$Y) = nested_broadcast(Base.$f, x, y)
end

# Using AutoBroadcasters/DefaultBroadcastables as if-else statement conditionals
for (X, Y) in constraint_permutations(2)
    @eval Base.ifelse(cond::AutoBroadcasterOrSimilar, x::$X, y::$Y) =
        nested_broadcast(ifelse, cond, x, y)
end

# Applying AutoBroadcasters like functions
(f::AutoBroadcaster)(args...) =
    nested_broadcast((f, args...) -> f(args...), f, args...)
