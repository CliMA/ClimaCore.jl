# Treat array elements as if they are fields of a single tuple value
field_expr(f, value_expr) = :(Core.getfield($value_expr, $f))
field_expr(f, array_expr, index_expr) =
    :(@inbounds $array_expr[struct_index($f, $array_expr, $index_expr...)])

# Keep array element read instructions separate unless the full value is needed
full_value_expr(@nospecialize(S), value_expr) = value_expr
full_value_expr(@nospecialize(S), inputs...) =
    Expr(:tuple, (field_expr(f, inputs...) for f in 1:fieldcount(S))...)

# Manually inline all read instructions into a generated function body, instead
# of relying on Julia's compiler to inline them during the code lowering stage
bitcast_struct_expr(@nospecialize(T), @nospecialize(S), inputs...) =
    if T === S
        full_value_expr(S, inputs...)
    elseif sizeof(T) != sizeof(S)
        error("Cannot bitcast $S ($(sizeof(S)) bytes) to $T ($(sizeof(T)) bytes)")
    elseif Base.issingletontype(T)
        T.instance
    elseif iszero(fieldcount(T)) && iszero(fieldcount(S)) && T !== Bool
        # Use Core.bitcast to convert primitive types, but not to turn non-Bools
        # into Bools, since that zeros out their first seven bits; for example,
        # Core.bitcast(Int8, Core.bitcast(Bool, Int8(3))) returns 1 instead of 3
        :(Core.bitcast($T, $(full_value_expr(S, inputs...))))
    elseif isone(count(!Base.issingletontype, fieldtypes(S)))
        # Ignore the singleton fields of S
        f = findfirst(!Base.issingletontype, fieldtypes(S))
        bitcast_struct_expr(T, fieldtype(S, f), field_expr(f, inputs...))
    elseif isone(count(!Base.issingletontype, fieldtypes(T)))
        # Use instances to get the singleton fields of T
        T_expr = Expr(:new, T)
        for field_type in fieldtypes(T)
            field_value_expr =
                Base.issingletontype(field_type) ? field_type.instance :
                bitcast_struct_expr(field_type, S, inputs...)
            push!(T_expr.args, field_value_expr)
        end
        T_expr
    else
        # Use unsafe_load to convert composite types, and to losslessly turn
        # non-Bools into Bools; implemented like getindex(v::MArray, i) from
        # https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.0.0/src/MArray.jl#L85,
        # but with an LLVMPtr instead of a Ptr so the LLVM compiler can apply
        # SROA/DCE (scalar replacement of aggregates and dead code elimination)
        isbitstype(T) || error("Cannot allocate $T in stack memory for bitcast")
        isbitstype(S) || error("Cannot allocate $S in stack memory for bitcast")
        quote
            stack_memory = Ref($(full_value_expr(S, inputs...)))
            pointer = Core.LLVMPtr{$T, 0}(pointer_from_objref(stack_memory))
            GC.@preserve stack_memory unsafe_load(pointer)
        end
    end

"""
    bitcast_struct(T, value)
    bitcast_struct(T, array, Val(num_indices), index...)

Converts `value` into an `isbits` type `T` that spans the same number of bytes
(counting all bytes that are used as padding; see extended help for details).
Serves as a GPU-compatible generalization of the native `Core.bitcast` function,
losslessly converting between arbitrary data types, including composite types.

Instead of converting a single value, it is also possible to convert a subset of
an array corresponding to the result of [`get_struct`](@ref). This is equivalent
to converting the array elements after first loading them into a tuple, but with
guaranteed inlining for arbitrary data types. Inlining is necessary for the
compiler's [`getfield_elim_pass!`](https://hackmd.io/bZz8k6SHQQuNUW-Vs7rqfw) to
eliminate reads of array elements for unused fields of `T` (a key optimization
in GPU kernels, where reads from global memory can be relatively expensive).

# Examples

```julia-repl
julia> bitcast_struct(NTuple{4, Int8}, Int32(1))
(1, 0, 0, 0)

julia> bitcast_struct(NTuple{6, Int32}, (2 * eps(0.0), eps(0.0), 0.0))
(2, 0, 1, 0, 0, 0)

julia> bitcast_struct(Tuple{Int32, Int32, Int128}, (2, 0, 1, 0))
(2, 0, 1)
```

# Extended help

The output of `bitcast_struct(T, value)` is similar to the output of
`reinterpret(T, value)`, with both functions interpreting sequential bytes in
[little-endian order](https://en.wikipedia.org/wiki/Endianness):

```julia-repl
julia> reinterpret(NTuple{4, Int8}, Int32(1))
(1, 0, 0, 0)

julia> reinterpret(NTuple{6, Int32}, (2 * eps(0.0), eps(0.0), 0.0))
(2, 0, 1, 0, 0, 0)

julia> reinterpret(Tuple{Int32, Int32, Int128}, (2, 1, 0))
(2, 0, 1)
```

As the last example shows, `bitcast_struct` and `reinterpret` can behave
differently when converting between data structures with nonuniform field sizes.
Specifically, they differ for data structures that are stored with
[padding](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Structure-Layout.html),
which the C code underlying Julia uses to ensure that fields are efficiently
aligned in stack memory.

Unlike `reinterpret(T, value)`, which avoids mixing padding with non-padding
(it recursively traverses fields of `value` and `T`, introducing offsets when
their padding bytes are in different positions), `bitcast_struct(T, value)`
makes no distinction between padding and non-padding. Although `reinterpret` is
therefore less likely to produce unexpected outputs, it also performs runtime
allocations in heap memory, making it unsuitable for GPU kernels that do not
support such allocations. In contrast, `bitcast_struct` has a much simpler
implementation, with all of its allocations confined to stack memory. Moreover,
as long as `bitcast_struct` is only called within [`set_struct!`](@ref) and
[`get_struct`](@ref), potentially unexpected outputs will be hidden from users.

In addition to the low-level method of `reinterpret` for `isbits` inputs, there
is another method for `AbstractArray` inputs that behaves exactly like
`bitcast_struct` when it comes to padding:

```julia-repl
julia> reinterpret(reshape, NTuple{4, Int8}, Int32[1])[1]
(1, 0, 0, 0)

julia> reinterpret(reshape, NTuple{6, Int32}, [2 * eps(0.0), eps(0.0), 0.0])[1]
(2, 0, 1, 0, 0, 0)

julia> reinterpret(reshape, Tuple{Int32, Int32, Int128}, [2, 0, 1, 0])[1]
(2, 0, 1)
```

This method of `reinterpret` reads bytes from heap memory without distinguishing
padding and non-padding, in the same way as `bitcast_struct` reads bytes from
stack memory. So, while the method of `reinterpret` for `isbits` inputs can
construct the nonuniform type `Tuple{Int32, Int32, Int128}` from three `Int64`s,
`bitcast_struct` and the method for arrays both require a fourth `Int64`,
spanning the eight padding bytes inserted between the `Int32`s and the `Int128`.

For more information about `reinterpret` and padding, see the following:
- https://discourse.julialang.org/t/reinterpret-returns-wrong-values
- https://discourse.julialang.org/t/reinterpret-vector-into-single-struct
- https://discourse.julialang.org/t/reinterpret-vector-of-mixed-type-tuples
"""
@generated bitcast_struct(::Type{T}, value::S) where {T, S} =
    Expr(:block, :@inline, bitcast_struct_expr(T, S, :value))

@generated function bitcast_struct(
    ::Type{T},
    array,
    ::Val{num_indices},
    index...,
) where {T, num_indices}
    S = NTuple{num_indices, eltype(array)}
    return Expr(:block, :@inline, bitcast_struct_expr(T, S, :array, :index))
end
