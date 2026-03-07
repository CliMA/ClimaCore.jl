# Treat array elements as if they are fields of a single tuple value
field_expr(f, value_expr) = :(Core.getfield($value_expr, $f))
field_expr(f, array_expr, index_expr) =
    :(@inbounds $array_expr[struct_index($f, $array_expr, $index_expr...)])

all_field_exprs(@nospecialize(S), inputs...) =
    Base.mapany(f -> field_expr(f, inputs...), 1:fieldcount(S))

# Keep array element read instructions separate unless the full value is needed
full_value_expr(@nospecialize(S), value_expr) = value_expr
full_value_expr(@nospecialize(S), array_expr, index_expr) =
    Expr(:tuple, all_field_exprs(S, array_expr, index_expr)...)

# Manually perform SROA (scalar replacement of aggregates) through a recursively
# generated expression, instead of relying on the compiler to guarantee inlining
bitcast_struct_expr(@nospecialize(T), @nospecialize(S), inputs...) =
    if T === S
        full_value_expr(S, inputs...)
    elseif sizeof(T) != sizeof(S)
        error("Cannot bitcast $S ($(sizeof(S)) bytes) to $T ($(sizeof(T)) bytes)")
    elseif issingletontype(T)
        T.instance
    elseif iszero(fieldcount(T)) && iszero(fieldcount(S)) && T !== Bool
        # Use Core.bitcast to convert primitive types, but not to turn non-Bools
        # into Bools, since that zeros out their first seven bits; for example,
        # Core.bitcast(Int8, Core.bitcast(Bool, Int8(3))) returns 1 instead of 3
        :(Core.bitcast($T, $(full_value_expr(S, inputs...))))
    elseif isone(count(!issingletontype, fieldtypes(S)))
        # Ignore the singleton fields of S
        f = findfirst(!issingletontype, fieldtypes(S))
        bitcast_struct_expr(T, fieldtype(S, f), field_expr(f, inputs...))
    elseif isone(count(!issingletontype, fieldtypes(T)))
        # Use instances to get the singleton fields of T
        T_expr = Expr(:new, T)
        for F_type in fieldtypes(T)
            issingletontype(F_type) && push!(T_expr.args, F_type.instance)
            issingletontype(F_type) && continue
            push!(T_expr.args, bitcast_struct_expr(F_type, S, inputs...))
        end
        T_expr
    else
        T_end = sizeof(T)
        S_end = sizeof(S)
        F_starts = map(Base.Fix1(fieldoffset, T), 1:fieldcount(T))
        F_ends = map(+, F_starts, map(sizeof, fieldtypes(T)))
        f_starts = map(Base.Fix1(fieldoffset, S), 1:fieldcount(S))
        f_ends = map(+, f_starts, map(sizeof, fieldtypes(S)))
        f_ends_padded = push!(f_starts[2:end], S_end)
        f_exprs = all_field_exprs(S, inputs...)
        if fieldcount(T) > 1 && (
            all(F_start -> F_start in f_starts || F_start == S_end, F_starts) &&
            all(F_end -> F_end in f_ends || F_end == 0, F_ends)
        )
            # Partition the fields of S if they are all aligned with fields of T
            T_expr = Expr(:new, T)
            for (F_type, F_start, F_end) in zip(fieldtypes(T), F_starts, F_ends)
                issingletontype(F_type) && push!(T_expr.args, F_type.instance)
                issingletontype(F_type) && continue
                f1 = findlast(==(F_start), f_starts)
                f2 = findfirst(==(F_end), f_ends)
                f12_type =
                    f1 == f2 ? fieldtype(S, f1) : Tuple{fieldtypes(S)[f1:f2]...}
                f12_expr =
                    f1 == f2 ? f_exprs[f1] : Expr(:tuple, f_exprs[f1:f2]...)
                F_expr = bitcast_struct_expr(F_type, f12_type, f12_expr)
                push!(T_expr.args, F_expr)
            end
            T_expr
        elseif fieldcount(S) > 1 && (
            all(f_start -> f_start in F_starts || f_start == T_end, f_starts) &&
            all(f_end -> f_end in F_ends || f_end == 0, f_ends)
        )
            # Partition the fields of T if they are all aligned with fields of S
            T_expr = Expr(:splatnew, T, Expr(:tuple))
            for (f_type, f_start, f_end, f_end_padded, f_expr) in
                zip(fieldtypes(S), f_starts, f_ends, f_ends_padded, f_exprs)
                issingletontype(f_type) && continue
                F1 = findfirst(==(f_start), F_starts)
                F2 = findfirst(==(f_end), F_ends)
                F2_padded = findlast(==(f_end_padded), F_ends)
                F12_type =
                    F1 == F2 ? fieldtype(T, F1) : Tuple{fieldtypes(T)[F1:F2]...}
                F12_expr = bitcast_struct_expr(F12_type, f_type, f_expr)
                non_padding_expr = F1 == F2 ? F12_expr : Expr(:..., F12_expr)
                push!(T_expr.args[2].args, non_padding_expr)
                for F_type in fieldtypes(T)[(F2 + 1):F2_padded]
                    # Leave fields that correspond to padding in S uninitialized
                    padding_expr = Expr(:new, F_type)
                    push!(T_expr.args[2].args, padding_expr)
                end
            end
            T_expr
        else
            # Use unsafe_load for all other conversions, including turning
            # non-Bools into Bools; implemented like getindex(v::MArray, i) from
            # https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.0.0/src/MArray.jl#L85,
            # but with an LLVMPtr instead of a Ptr for better LLVM optimization
            isbitstype(T) || error("Cannot bitcast mutable $T in stack memory")
            isbitstype(S) || error("Cannot bitcast mutable $S in stack memory")
            quote
                stack_memory = Ref($(full_value_expr(S, inputs...)))
                pointer = Core.LLVMPtr{$T, 0}(pointer_from_objref(stack_memory))
                GC.@preserve stack_memory unsafe_load(pointer)
            end
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
    bitcast_expr = bitcast_struct_expr(T, S, :array, :index)
    return Expr(:block, :(Base.@_propagate_inbounds_meta), bitcast_expr)
end
