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

end
