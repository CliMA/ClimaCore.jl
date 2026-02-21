module Utilities

import InteractiveUtils

include("auto_broadcaster.jl")
include("plushalf.jl")
include("cache.jl")

"""
    cart_ind(n::NTuple, i::Integer)

Returns a `CartesianIndex` from the list
`CartesianIndices(map(x->Base.OneTo(x), n))[i]`
given size `n` and location `i`.
"""
Base.@propagate_inbounds cart_ind(n::NTuple, i::Integer) =
    @inbounds CartesianIndices(map(x -> Base.OneTo(x), n))[i]

"""
    linear_ind(n::NTuple, ci::CartesianIndex)
    linear_ind(n::NTuple, t::NTuple)

Returns a linear index from the list
`LinearIndices(map(x->Base.OneTo(x), n))[ci]`
given size `n` and cartesian index `ci`.

The `linear_ind(n::NTuple, t::NTuple)` wraps `t`
in a `Cartesian` index and calls
`linear_ind(n::NTuple, ci::CartesianIndex)`.
"""
Base.@propagate_inbounds linear_ind(n::NTuple, ci::CartesianIndex) =
    @inbounds LinearIndices(map(x -> Base.OneTo(x), n))[ci]
Base.@propagate_inbounds linear_ind(n::NTuple, loc::NTuple) =
    linear_ind(n, CartesianIndex(loc))

"""
    unionall_type(::Type{T})

Extract the type of the input, and strip it of any type parameters.

This is useful when one needs the generic constructor for a given type.

Example
=======
```julia
julia> unionall_type(typeof([1, 2, 3]))
Array

julia> struct Foo{A, B}
               a::A
               b::B
       end

julia> unionall_type(typeof(Foo(1,2)))
Foo
```
"""
function unionall_type(::Type{T}) where {T}
    # NOTE: As of version 1.12, there is no simple, user-friendly way to extract
    # the generic type of T, so we need to reach for the internals in Julia.
    # Hopefully, Julia will introduce a simpler, more stable way to do this in a
    # future release.
    return T.name.wrapper
end

struct InferenceError <: Exception
    msg::AbstractString
    f::Any
    T::Type{<:Tuple}
end
function Base.showerror(io::IO, err::InferenceError)
    println(io, "InferenceError: ", err.msg)
    InteractiveUtils.code_warntype(io, err.f, err.T)
end

function possibly_non_concrete_result_type(f::F, ::Type{T}) where {F, T}
    hasmethod(f, T) || throw(MethodError(f, T))
    result_type = Core.Compiler.return_type(f, T)
    result_type != Union{} ||
        throw(ArgumentError("Error in $f for argument types $(fieldtypes(T))"))
    return result_type
end

"""
    inferred_type(f, types...)
    inferred_type(f)

The type of value returned by `f` for arguments of the given types. An error is
thrown if the concrete result type cannot be inferred, or if there is no method
of `f` that can handle the specified types.

Omitting the type arguments causes `inferred_type` to generate the function
closure `(types...) -> inferred_type(f, types...)`.
"""
inferred_type(f::F) where {F} = (types...) -> inferred_type(f, types...)
inferred_type(f::F, types...) where {F} = _inferred_type(f, Tuple{types...})
function _inferred_type(f::F, ::Type{T}) where {F, T}
    result_type = possibly_non_concrete_result_type(f, T)
    isconcretetype(result_type) ||
        throw(InferenceError("Unable to infer concrete type of result", f, T))
    return result_type
end

"""
    inferred_const(f, types...)
    inferred_const(f)

The constant value returned by `f` for arguments of the given types, computed
similarly to [`inferred_type`](@ref), but with a `Val` wrapper. Only quantities
that are marked as `Core.Const` by `@code_warntype` can be inferred like this.

Omitting the type arguments causes `inferred_const` to generate the function
closure `(types...) -> inferred_const(f, types...)`.
"""
inferred_const(f::F) where {F} = (types...) -> inferred_const(f, types...)
inferred_const(f::F, types...) where {F} = _inferred_const(f, Tuple{types...})
function _inferred_const(f::F, ::Type{T}) where {F, T}
    result_type = possibly_non_concrete_result_type(f, T)
    val_type_with_result_value = Core.Compiler.return_type(Val ∘ f, T)
    val_type_with_result_value != Union{} ||
        throw(ArgumentError("Unable to infer value of $result_type"))
    isconcretetype(val_type_with_result_value) ||
        throw(InferenceError("Unable to infer Core.Const for result", f, T))
    return first(val_type_with_result_value.parameters)
end

"""
    broadcast_eltype(bc)

Optimized analogue of `eltype` that has better type stability on Julia 1.10 for
nested `Base.Broadcast.Broadcasted` wrappers than the built-in function
`Base.Broadcast.combine_eltypes(bc.f, bc.args)`.
"""
broadcast_eltype(bc) = eltype(bc)
function broadcast_eltype(bc::Base.Broadcast.Broadcasted)
    arg_eltypes = unrolled_map(broadcast_eltype, bc.args)
    unrolled_all(!=(Union{}), arg_eltypes) || return Union{}
    return Core.Compiler.return_type(bc.f, Tuple{arg_eltypes...})
end

"""
    broadcast_result_type(f, X, [Y])
    
Inferred result of `typeof(f.(x, [y]))` for arguments `x`/`y`of types `X`/`Y`.
"""
broadcast_result_type(f::F, ::Type{X}) where {F, X} =
    inferred_type(broadcast, typeof(f), X)
broadcast_result_type(f::F, ::Type{X}, ::Type{Y}) where {F, X, Y} =
    inferred_type(broadcast, typeof(f), X, Y)

"""
    broadcast_over_element_types(f, X, [Y])

Value of `f.(typeof.(x), [typeof.(y)])` for arguments `x`/`y` of types `X`/`Y`.
"""
broadcast_over_element_types(f::F, ::Type{X}) where {F, X} =
    inferred_const(broadcast, typeof(f ∘ typeof), X)
broadcast_over_element_types(f::F, ::Type{X}, ::Type{Y}) where {F, X, Y} =
    inferred_const(broadcast, typeof((x, y) -> f(typeof(x), typeof(y))), X, Y)

end # module
