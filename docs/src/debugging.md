# Debugging

`ClimaCore` comes with a set of debugging tools for a variety of situations.
These tools are collected in the [`ClimaCore.DebugOnly`](@ref) modules and are
presented in this page.

## Finding where `NaN` arise

One of the most challenging tasks that users have is: debug a large simulation
that is breaking, e.g., yielding `NaN`s somewhere. This is especially complex
for large models with many terms and implicit time-stepping with all the bells
and whistles that the CliMA ecosystem offers.

Because so much data (for example, the solution state, and many cached fields)
is typically contained in `ClimaCore` data structures, we offer a hook to
inspect this data after any operation that `ClimaCore` performs.

### Example

#### Print `NaNs` when they are found

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
because it is called by many different functions with many different objects.
Therefore, we recommend that you define `post_op_callback` with a very general
method signature, like the one above and perhaps use `Infiltrator` to inspect
the arguments.

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

As you see, this only tells us that a `NaN` was found, but not which function
triggered the `NaN`. A simple way to find that is to extend this example with
`Infiltrator`.

#### Infiltrating and Exfiltrating

[Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) is a simple
debugging tool for Julia packages.

Here is an example, where we can use Infiltrator.jl to find where `NaN`s is coming
from interactively.

##### Infiltrating

Suppose you have a `NaN` in your simulation and what to look at the variables
right before the expression caused the `NaN`. One of the challenges in this is
that you probably have a tower of function being called, many of which belong to
other packages.

Let us simulate this case (note this is a toy example optimized for clarity and
not for performance: the functions below have unnecessary allocations)
```julia
import ClimaCore
using ClimaCore.CommonSpaces
space = CubedSphereSpace(; radius = 10, n_quad_points = 4, h_elem = 10)
myrho = ones(space)
myP = ones(space)
myu = ones(space)

kinetic_energy(field) = field .* field ./ 2
other_energy(field) = field ./ sum(field)
offset_by_one(field) = field .- 1

function specific_energy(rho, P, u)
    density_without_restmass = offset_by_one(rho)
    return (kinetic_energy(u) .+ other_energy(P)) ./ density_without_restmass
end

function renormalized_energy(rho, P, u)
    energy = specific_energy(rho, P, u)
    return energy ./ sum(energy)
end

any(isnan, renormalized_energy(myrho, myP, myu)) # true
```

To debug this, we first need to identify where the first `NaN` is produced. We
use `DebugOnly.call_post_op_callback` and infiltrate.
```julia
import Infiltrator # must be in your default environment
ClimaCore.DebugOnly.call_post_op_callback() = true
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    has_nans = result isa Number ? isnan(result) : any(isnan, parent(result))
    has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
    @infiltrate has_nans || has_inf
end
```

"Infiltrating" means being dropped into a new REPL where in the scope of the
`@infiltrate` macro. `@infiltrate condition` means that we want to infiltrate
only when the `condition` is true (in this case `has_nans || has_inf`).

Now, when we run our example, we will see
```julia
julia> renormalized_energy(myrho, myP, myu)
Infiltrating post_op_callback(::ClimaCore.DataLayouts.IJFH{Float64, 4, Array{Float64, 4}}, ::ClimaCore.DataLayouts.IJFH{Float64, 4, Array{Float64, 4}}, ::Vararg{Any}; kwargs::@Kwargs{})
  at REPL[40]:4
infil>
```
Here, we are dropped into a new REPL with full access to the variables in the scope where the `NaN` occurred. However, because of how `post_op_callback`, this is at a low level within `ClimaCore`, which is typically not useful. Hence, the next step is to type `@trace`, which prints out
```julia
[1] post_op_callback(::ClimaCore.DataLayouts.IJFH{…}, ::ClimaCore.DataLayouts.IJFH{…}, ::Vararg{…}; kwargs::@Kwargs{})
    at REPL[40]:4
[2] post_op_callback
    at REPL[40]:1
[3] copyto!
    at ClimaCore.jl/src/DataLayouts/copyto.jl:18
[4] copyto!
    at ClimaCore.jl/src/Fields/broadcast.jl:190
[5] copy
    at ClimaCore.jl/src/Fields/broadcast.jl:97
[6] materialize
    at .julia/juliaup/julia-1.11.4+0.x64.linux.gnu/share/julia/base/broadcast.jl:872
[7] specific_energy(rho::ClimaCore.Fields.Field{…}, P::ClimaCore.Fields.Field{…}, u::ClimaCore.Fields.Field{…})
    at REPL[31]:2
[8] renormalized_energy(rho::ClimaCore.Fields.Field{…}, P::ClimaCore.Fields.Field{…}, u::ClimaCore.Fields.Field{…})
    at REPL[36]:2
[9] top-level scope
```

`@trace` returns a type-limited stacktrace that we can read backwards until we
see our functions. In this case, we see that the first `NaN` is in
`specific_energy`, so we will investigate that function. We leave the
Infiltrator REPL with `@exit`, disable the `call_post_op_callback`, and move our
`infiltrate` call within the target function:
```julia
ClimaCore.DebugOnly.call_post_op_callback() = false
function specific_energy(rho, P, u)
    @infiltrate
    density_without_restmass = offset_by_one(rho)
    return (kinetic_energy(u) .+ other_energy(P)) ./ density_without_restmass
end
```
Now, when we evaluate our problematic expression (the one at the top level, in this case `renormalized_energy(myrho, myP, myu)`), we will be dropped in a REPL inside `specific_energy`. Here, we have access to `density_without_restmass`, and we notice that it can be zero, leading to the `NaN`.

!!! tip

    The infiltrator REPL is different from the normal Julia repl. Type `?` for
    some useful commands. You can fetch objects defined in the main REPL by
    prepending their name with `Main`. Similarly, if you want to infiltrate inside a
    module, prepend `@infiltrate` with `Main` (`Main.@infiltrate`).

Looking at this code, we could have probably guess that the `NaN` comes from a
division from 0, and the real question is how do we get that?. In more complex
cases, the functions are spread across different packages and involve several
different variables. This approach allows one to systematically identify where
things go wrong.

##### Exfiltrating and StructuredPrinting

Let's now see a different way to use Infiltrator, where we move the variables in specific scope to the Main scope in the REPL and do some analysis on it.

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

#### Caveats

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

#### Faster explorations when the initialization is expensive

Sometimes, we want to start from a given simulation state and explore different
ideas. For example, we want to run a simulation for 10 days, and then test how
different approaches affect its stability. Sometimes, checkpoints offer a way to
do this, but not everything can be checkpointed. 

A simple way to "checkpoint" a simulation is to `deepcopy` its state. This
allows one to step the copy instead of the original one, which can be re-used to
make new copies, allowing for various explorations. `ClimaCore` does not support
this workflow out-of-the-box. The reason for this is that `ClimaCore` uses
pointers to perform certain safety checks, and deepcopies return new pointers
(by definition). To enable this, override the
`DebugOnly.allow_mismatched_spaces_unsafe` function so that it returns true. When
`DebugOnly.allow_mismatched_spaces_unsafe` returns true, `ClimaCore` can mix fields
defined on space that are not identically the same.

Let us look at an example of this.
```julia
import ClimaCore
using ClimaCore.CommonSpaces
other_space = deepcopy(space)

one = ones(space)
other_one = ones(other_space)

one .+ other_one  # This throws an error 

ClimaCore.DebugOnly.allow_mismatched_spaces_unsafe() = true
one .+ other_one # Now it's fine!
```

!!! warn

    `ClimaCore` checks for consistency of spaces to protect you from non-sense
    results. If you disable this check, you are responsible to ensure that the
    results make sense.
