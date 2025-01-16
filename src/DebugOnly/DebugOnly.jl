"""
	DebugOnly

A module for debugging tools. Note that any tools in here are subject to sudden
changes without warning. So, please, do _not_ use any of these tools in
production as support for them are not guaranteed.
"""
module DebugOnly

"""
    post_op_callback(result, args...; kwargs...)

A callback that is called, if `ClimaCore.DataLayouts.call_post_op_callback() =
true`, on the result of every data operation.

There is purposely no implementation-- this is a debugging tool, and users may
want to check different things.

Note that, since this method is called in so many places, this is a
performance-critical code path and expensive operations performed in
`post_op_callback` may significantly slow down your code.
"""
function post_op_callback end

"""
    call_post_op_callback()

Returns a Bool. Meant to be overloaded so that
`ClimaCore.DataLayouts.post_op_callback(result, args...; kwargs...)` is called
after every data operation.
"""
call_post_op_callback() = false

# TODO: define a convenience macro to inject `post_op_hook`

"""
	example_debug_post_op_callback(result, args...; kwargs...)

An example `post_op_callback` method, that checks for `NaN`s and `Inf`s.
"""
function example_debug_post_op_callback(result, args...; kwargs...)
    if any(isnan, parent(result))
        error("NaNs found!")
    elseif any(isinf, parent(result))
        error("Inf found!")
    end
end

"""
    depth_limited_stack_trace(
        [io::IO, ]
        st::Base.StackTraces.StackTrace;
        maxtypedepth=3
    )

Given a stacktrace `st`, return a vector of strings containing the trace with
depth-limited printing.
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
    print_depth_limited_stack_trace(
        [io::IO, ]
        st::Base.StackTraces.StackTrace;
        maxtypedepth=3
    )

Given a stacktrace `st`, print the stack trace with depth-limited types, given
by `maxtypedepth`.
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
