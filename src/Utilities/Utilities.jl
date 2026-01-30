module Utilities

import InteractiveUtils

include("math_mapper.jl")
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

"""
    inferred_result_type(f, types...)

The type of value returned by `f` for arguments of the given types. An error is
thrown if the concrete result type cannot be inferred, or if no method can be
found to handle the argument types.
"""
function inferred_result_type(f::F, types...) where {F}
    unrolled_all(!=(Union{}), types) ||
        throw(ArgumentError("Type of argument $(findfirst(==(Union{}), types)) \
                             to $f is unknown"))
    hasmethod(f, Tuple{types...}) || throw(MethodError(f, Tuple{types...}))
    inferred_type = Core.Compiler.return_type(f, Tuple{types...})
    isconcretetype(inferred_type) ||
        throw(ErrorException("Cannot infer type of value returned by $f: \
                              \n$(inference_string(f, types...))"))
    return inferred_type
end

"""
    inferred_result_value(f, types...)

The value returned by `f` for arguments of the given types, computed by using
[`inferred_result_type`](@ref) in conjunction with a `Val` wrapper. The value
returned by a function can only be inferred when it is a compile-time constant
(i.e., when it is marked as a `Core.Const` in the output of `@code_warntype`).
"""
function inferred_result_value(f::F, types...) where {F}
    inferred_result_type(f, types...) # First check whether the type is inferred
    inferred_val_type = Core.Compiler.return_type(Val ∘ f, Tuple{types...})
    isconcretetype(inferred_val_type) ||
        throw(ErrorException("Cannot infer constant value returned by $f: \
                              \n$(inference_string(f, types...))"))
    return val_type_parameter(inferred_val_type)
end

function inference_string(f, types...)
    io = IOBuffer()
    InteractiveUtils.code_warntype(io, f, Tuple{types...})
    return String(take!(io))
end

# Wrap values passed between functions in Vals to guarantee constant-propagation
val_parameter(::Val{constant}) where {constant} = constant
val_type_parameter(::Type{Val{constant}}) where {constant} = constant

end # module
