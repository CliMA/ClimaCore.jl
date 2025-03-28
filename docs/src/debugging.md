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
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    if has_nans || has_inf
        has_nans && println("NaNs found!")
        has_inf && println("Infs found!")
    end
end
```

If needed, multiple methods of `post_op_callback` can be defined, but here, we
define a general method that checks if `NaN`s are in the given object.

Note that we need `post_op_callback` to be called for a wide variety of inputs
because it is called by many many different functions with many different
objects. Therefore, we recommend that you define `post_op_callback` with a very
general method signature, like the one above and perhaps use `Infiltrator` to
inspect the arguemtns.

Now, let us put everything together and demonstrate a complete example:

```@example clima_debug
import ClimaCore
ClimaCore.DebugOnly.call_post_op_callback() = true
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    if has_nans || has_inf
        has_nans && println("NaNs found!")
        has_inf && println("Infs found!")
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
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    if has_nans || has_inf
        has_nans && println("NaNs found!")
        has_inf && println("Infs found!")
        # Let's define the stack trace so that we know where this came from
        st = stacktrace()

        # Let's use Infiltrator.jl to exfiltrate to drop into the REPL.
        # Now, `Infiltrator.safehouse` will be a NamedTuple
        # containing `result`, `args`, `kwargs` and `st`.
        Infiltrator.@exfiltrate
        # sometimes code execution doesn't stop, I'm not sure why. Let's
        # make sure we exfiltrate immediately with the data we want.
        error("Exfiltrating.")
    end
end

FT = Float64
data = ClimaCore.DataLayouts.VIJFH{FT}(Array{FT}, zeros; Nv=5, Nij=2, Nh=2)
x = ClimaCore.DataLayouts.VIJFH{FT}(Array{FT}, zeros; Nv=5, Nij=2, Nh=2)
parent(x)[1] = NaN # emulate incorrect initialization
@. data = x + 1
# Let's see what happened
(;result, args, kwargs, st) = Infiltrator.safehouse;

# You can print the stack trace, to see where the NaNs were found:
ClimaCore.DebugOnly.print_depth_limited_stack_trace(st; maxtypedepth=1)

# Once there, you can see that the call lead you to `copyto!`,
# Inspecting `args` shows that the `Broadcasted` object used to populate the
# result was:
julia> args[2]
Base.Broadcast.Broadcasted{ClimaCore.DataLayouts.VIJFHStyle{5, 2, Array{Float64}}}(+, (ClimaCore.DataLayouts.VIJFH{Float64, 5, 2, Array{Float64, 5}}
  [NaN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1))

# And there's your problem, NaNs is on the right-hand-side of that assignment.
```

If your broadcasted object is very long, it can be a bit overwhelming to figure
out which part of the expression contains NaNs (if any). To make this process
more manageable, we can use [`StructuredPrinting.jl`]
(https://github.com/charleskawczynski/StructuredPrinting.jl) to highlight which
parts of the broadcasted object contains NaNs:

```julia
using StructuredPrinting
import ClimaCore: DataLayouts
highlight_nans(x::DataLayouts.AbstractData) = any(y->isnan(y), parent(x));
highlight_nans(_) = false;
bc = Infiltrator.safehouse.args[2]; # we know that argument 2 is the broadcasted object
(; result) = Infiltrator.safehouse; # get the result
@structured_print bc Options(; highlight = x->highlight_nans(x))
```
This last line results in:

```julia
julia> @structured_print bc Options(; highlight = x->highlight_nans(x))
bc
bc.style::ClimaCore.DataLayouts.VIJFHStyle{5, 2, Array{Float64}}
bc.f::typeof(+)
bc.args::Tuple{ClimaCore.DataLayouts.VIJFH{Float64, 5, 2, Array{…}}, Int64}
bc.args.1::ClimaCore.DataLayouts.VIJFH{Float64, 5, 2, Array{Float64, 5}}       # highlighted in RED
bc.args.2::Int64
bc.axes::NTuple{5, Base.OneTo{Int64}}
bc.axes.1::Base.OneTo{Int64}
bc.axes.1.stop::Int64
bc.axes.2::Base.OneTo{Int64}
bc.axes.2.stop::Int64
bc.axes.3::Base.OneTo{Int64}
bc.axes.3.stop::Int64
bc.axes.4::Base.OneTo{Int64}
bc.axes.4.stop::Int64
bc.axes.5::Base.OneTo{Int64}
bc.axes.5.stop::Int64
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

!!! warn

    It is _highly_ recommended to use `post_op_callback` _without_ `@testset`,
    as Test.jl may continue running through code execution, until all of the
    tests in a given `@testset` are complete, and the result will be that you
    will get the _last_ observed instance of `NaN` or `Inf`.
