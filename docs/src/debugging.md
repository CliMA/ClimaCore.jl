# Debugging

One of the most challenging tasks that users have is: debug a large simulation
that is breaking, e.g., yielding `NaN`s somewhere. This is especially complex
for large models with many terms and implicit time-stepping with all the bells
and whistles that the CliMA ecosystem offers.

ClimaCore has a module, [`ClimaCore.DebugOnly`](@ref), which contains tools for
debugging simulations for these complicated situations.

Because so much data (for example, the solution state, and many cached fields)
is typically contained in ClimaCore data structures, we offer a hook to inspect
this data after any operation that ClimaCore performs.

## Example

### Print `NaNs` when they are found

In this example, we add a callback that simply prints `NaNs found` every
instance when they are detected in a `ClimaCore` operation.

To do this, we need two ingredients:

First, we need to enable the callback system:
```@example clima_debug
import ClimaCore
ClimaCore.DebugOnly.call_post_op_callback() = true
```

The line `ClimaCore.DebugOnly.call_post_op_callback() = true` means that at the
end of every `ClimaCore` operation, the function
`ClimaCore.DebugOnly.post_op_callback` is called. By default, this function does
nothing. So, the second ingredient is to define a method:
```@example clima_debug
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    if any(isnan, parent(result))
        println("NaNs found!")
    end
end
```
If needed, `post_op_callback` can be specialized or behave differently in
different cases, but here, it only checks if `NaN`s are in the given that.

Note that, due to dispatch, `post_op_callback` will likely need a very general
method signature, and using `post_op_callback
(result::DataLayouts.VIJFH, args...; kwargs...)` above fails (on the CPU),
because `post_op_callback` ends up getting called multiple times with different
datalayouts.

Now, let us put everything together and demonstrate a complete example:

```@example clima_debug
import ClimaCore
ClimaCore.DebugOnly.call_post_op_callback() = true
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    if any(isnan, parent(result))
        println("NaNs found!")
    end
end

FT = Float64
data = ClimaCore.DataLayouts.VIJFH{FT}(Array{FT}, zeros; Nv=5, Nij=2, Nh=2)
@. data = NaN
ClimaCore.DebugOnly.call_post_op_callback() = false # hide
```
This example should print `NaN` on your standard output.

### Infiltrating

[Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) is a simple
debugging tool for Julia packages.

Here is an example, where we can use Infiltrator.jl to find where NaNs is coming
from interactively.

```julia
import ClimaCore
import Infiltrator # must be in your default environment
ClimaCore.DebugOnly.call_post_op_callback() = true
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    if any(isnan, parent(result))
        println("NaNs found!")
        # Let's define the stack trace so that we know where this came from
        st = stacktrace()

        # Let's use Infiltrator.jl to exfiltrate to drop into the REPL.
        # Now, `Infiltrator.safehouse` will be a NamedTuple
        # containing `result`, `args` and `kwargs`.
        Infiltrator.@exfiltrate
    end
end

FT = Float64
data = ClimaCore.DataLayouts.VIJFH{FT}(Array{FT}, zeros; Nv=5, Nij=2, Nh=2)
@. data = NaN
# Let's see what happened
(;result, args, kwargs, st) = Infiltrator.safehouse;

# You can print the stack trace, to see where the NaNs were found:
ClimaCore.DebugOnly.print_depth_limited_stack_trace(st;maxtypedepth=1)

# Once there, you can see that the call lead you to `copyto!`,
# Inspecting `args` shows that the `Broadcasted` object used to populate the
# result was:
julia> args[2]
Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}(identity, (NaN,))

# And there's your problem, NaNs is on the right-hand-side of that assignment.
```

### Caveats

!!! warn

    While `post_op_callback` may be helpful, it's not bullet proof. NaNs can
    infiltrate user data any time internals are used. For example `parent
    (data) .= NaN` will not be caught by ClimaCore.DebugOnly, and errors can be
    observed later than expected.

!!! note

    `post_op_callback` is called in many places, so this is a
    performance-critical code path and expensive operations performed in
    `post_op_callback` may significantly slow down your code.
