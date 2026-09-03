"""
    DebugOnly

Module for debugging tools. The tools in this module are subject to change without
warning and are not supported for production use.
"""
module DebugOnly

"""
    post_op_callback(result, args...; kwargs...)

Callback applied to the result of every data operation when
[`call_post_op_callback`](@ref) returns `true`.

The function has no methods by default: it is a debugging hook, and users add the
method that checks what they need (see [`example_debug_post_op_callback`](@ref)).

This function is called from every data operation, so expensive work performed in
`post_op_callback` slows down the whole code.
"""
function post_op_callback end

"""
    call_post_op_callback()

Return whether [`post_op_callback`](@ref) is called after every data operation.
The default method returns `false`; overload it to return `true` to enable the
callback:

```julia
ClimaCore.DebugOnly.call_post_op_callback() = true
```
"""
call_post_op_callback() = false

# TODO: define a convenience macro to inject `post_op_hook`

"""
    example_debug_post_op_callback(result, args...; kwargs...)

Example [`post_op_callback`](@ref) implementation that throws an error if `result`
contains a `NaN` or an `Inf`.
"""
function example_debug_post_op_callback(result, args...; kwargs...)
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    if has_nans || has_inf
        has_nans && error("NaNs found!")
        has_inf && error("Infs found!")
    end
end

"""
    depth_limited_stack_trace([io::IO,] st::Base.StackTraces.StackTrace; maxtypedepth = 3)

Return a vector of strings, one per frame of the stack trace `st`, with type
parameters printed to a depth of at most `maxtypedepth`. The width of `io` (default
`stdout`) determines where types are truncated.
"""
depth_limited_stack_trace(st::Base.StackTraces.StackTrace; maxtypedepth = 3) =
    depth_limited_stack_trace(stdout, st; maxtypedepth)

function depth_limited_stack_trace(
    io::IO,
    st::Base.StackTraces.StackTrace;
    maxtypedepth = 3,
)
    return map(s -> type_depth_limit(io, string(s); maxtypedepth), st)
end

function type_depth_limit(io::IO, s::String; maxtypedepth::Union{Nothing, Int})
    sz = get(io, :displaysize, displaysize(io))::Tuple{Int, Int}
    return Base.type_depth_limit(s, max(sz[2], 120); maxdepth = maxtypedepth)
end

"""
    print_depth_limited_stack_trace([io::IO,] st::Base.StackTraces.StackTrace; maxtypedepth = 3)

Print the stack trace `st` to `io` (default `stdout`), with type parameters printed
to a depth of at most `maxtypedepth`.
"""
print_depth_limited_stack_trace(
    st::Base.StackTraces.StackTrace;
    maxtypedepth = 3,
) = print_depth_limited_stack_trace(stdout, st; maxtypedepth)

function print_depth_limited_stack_trace(
    io::IO,
    st::Base.StackTraces.StackTrace;
    maxtypedepth = 3,
)
    for t in depth_limited_stack_trace(st; maxtypedepth)
        println(io, t)
    end
end


"""
    allow_mismatched_spaces_unsafe()

Return whether the check for consistent spaces in broadcasted operations is
disabled. The default method returns `false`.

By default, `ClimaCore` checks that broadcasted in-place expressions use
consistent spaces (i.e., the destination space is the same as the space that the
expression returns). When debugging, it can be convenient to disable this check.

The most common use is to combine spaces that were `deepcopy`ed: the consistency
check compares the spaces by identity, not by contents, so it rejects spaces that
are identical but not the same object.

To allow combining mismatched spaces, override this function so that it returns
`true`.

!!! warning

    `ClimaCore` checks for consistency of spaces to protect against nonsensical
    results. If you disable this check, you are responsible for ensuring that the
    results make sense.

# Examples

```julia
julia> import ClimaCore;

julia> using ClimaCore.CommonSpaces;

julia> space = ExtrudedCubedSphereSpace(;
           z_elem = 10,
           z_min = 0,
           z_max = 1,
           radius = 10,
           h_elem = 10,
           n_quad_points = 4,
           staggering = CellCenter(),
       );

julia> other_space = deepcopy(space);

julia> other_space == space
false

julia> one = ones(space);

julia> other_one = ones(other_space);

julia> one .+ other_one
ERROR: Broacasted spaces are not the same.
Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:35
 [2] error_mismatched_spaces(space1::Type, space2::Type)
   @ ClimaCore.Fields ~/repos/ClimaCore.jl/src/Fields/broadcast.jl:227

# Turning `allow_mismatched_spaces_unsafe` on

julia> ClimaCore.DebugOnly.allow_mismatched_spaces_unsafe() = true;

julia> one .+ other_one
Float64-valued Field:
  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0  …  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
```
"""
function allow_mismatched_spaces_unsafe()
    return false
end

end
