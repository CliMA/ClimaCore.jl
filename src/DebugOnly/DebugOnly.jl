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
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    if has_nans || has_inf
        has_nans && error("NaNs found!")
        has_inf && error("Infs found!")
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


"""
    allow_mismatched_spaces_unsafe()

When returning `true`, disable check for consistency among spaces in broadcasted
operations.

By default, `ClimaCore` checks that broadcasted in-place expressions use
consistent spaces (ie, the destination space is the same as the space that the
expression returns). Sometimes, when debugging, it is convenient to disable this
check.

The most common case for this is to allow combining spaces that were
`deepcopied`, given that the consistency check in performed by comparing the
pointer of the spaces, not their contents. In other words, allowing for
mismatched spaces allows one to work with spaces that are identical, but not the
same.

To allow combining mismatched spaces, override this function so that it returns
`true`.

!!! warn

    `ClimaCore` checks for consistency of spaces to protect you from non-sense
    results. If you disable this check, you are responsible to ensure that the
    results make sense.

Example
=======

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
           staggering = CellCenter()
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
"""
function allow_mismatched_spaces_unsafe()
    return false
end

end
