# Debug NaNs and broadcasts

`ClimaCore.DebugOnly` holds hooks for locating where a simulation first
produces a `NaN` or `Inf` and for inspecting the broadcast expression that
produced it. A large model evaluates hundreds of broadcasts per step, most of
them inside other packages, so the hooks are placed at the one point they all
pass through: the end of every `ClimaCore` operation.

## Prerequisites

Optional: [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) in
the default environment for the interactive steps, and
[StructuredPrinting.jl](https://github.com/CliMA/StructuredPrinting.jl) for
inspecting broadcast objects.

## Steps

 1. Switch the hook on and give it a method. When
    `DebugOnly.call_post_op_callback()` returns `true`, every `ClimaCore`
    operation ends by calling `DebugOnly.post_op_callback(result, args...; kwargs...)`
    with its result and arguments. The function has no methods by default;
    define one with a general signature, since it is called from many places
    with many argument types:

    ```@example clima_debug
    import ClimaCore
    ClimaCore.DebugOnly.call_post_op_callback() = true
    function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
        has_nan = result isa Number ? isnan(result) : any(isnan, parent(result))
        has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
        has_nan && println("NaN found")
        has_inf && println("Inf found")
    end
    data = ClimaCore.DataLayouts.VIJFH{Float64, 5, 2, 2, 2}(Array{Float64})
    @. data = NaN
    ```

    The hook applies to every `ClimaCore` operation in the session, including
    code unrelated to the problem, so switch it off once the `NaN` is located:

    ```@example clima_debug
    ClimaCore.DebugOnly.call_post_op_callback() = false
    nothing # hide
    ```

 2. Find the operation that produced it. The message above says that a `NaN`
    appeared, not where. With Infiltrator, drop into a REPL at the first
    occurrence instead of printing:

    ```julia
    import Infiltrator
    ClimaCore.DebugOnly.call_post_op_callback() = true
    function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
        has_nan = result isa Number ? isnan(result) : any(isnan, parent(result))
        has_inf = result isa Number ? isinf(result) : any(isinf, parent(result))
        @infiltrate has_nan || has_inf
    end
    ```

    `@infiltrate condition` opens the `infil>` REPL in the scope of the macro
    when the condition holds. That scope is inside `ClimaCore`'s `copyto!`,
    which is rarely informative by itself; type `@trace` for a stack trace with
    type-limited signatures and read it upward until your own functions appear:

    ```text
    [3] copyto!         at ClimaCore.jl/src/DataLayouts/copyto.jl:18
    [4] copyto!         at ClimaCore.jl/src/Fields/broadcast.jl:190
    [5] copy            at ClimaCore.jl/src/Fields/broadcast.jl:97
    [6] materialize     at base/broadcast.jl:872
    [7] specific_energy(rho::Field, P::Field, u::Field)   at REPL[31]:2
    [8] renormalized_energy(rho::Field, P::Field, u::Field)   at REPL[36]:2
    ```

    Here the first `NaN` appears in `specific_energy`. Leave the REPL with
    `@exit`, switch the hook off, and place `@infiltrate` inside that function
    to inspect its local variables before the offending expression runs. In
    the `infil>` REPL, `?` lists the commands; objects from the main session
    are reached by prefixing `Main`, and `Main.@infiltrate` is the form to use
    inside a module.

 3. Alternatively, exfiltrate the arguments to the main session and inspect
    them there. `Infiltrator.@exfiltrate` copies the local variables into
    `Infiltrator.safehouse`; raising an error afterwards stops at the first
    occurrence:

    ```julia
    import Infiltrator
    ClimaCore.DebugOnly.call_post_op_callback() = true
    function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
        has_nan = result isa Number ? isnan(result) : any(isnan, parent(result))
        if has_nan
            st = stacktrace()
            Infiltrator.@exfiltrate   # result, args, kwargs, and st
            error("exfiltrated at the first NaN")
        end
    end
    ```

    After the error, `(; result, args, st) = Infiltrator.safehouse` holds the
    data. `ClimaCore.DebugOnly.print_depth_limited_stack_trace(st; maxtypedepth = 1)`
    prints the trace with the field and space types abbreviated. When the
    trace leads to `copyto!`, `args[2]` is the `Broadcasted` object whose
    evaluation produced the result, and StructuredPrinting highlights the parts
    of it that contain `NaN`s:

    ```julia
    using StructuredPrinting
    import ClimaCore: DataLayouts
    has_nan(x::DataLayouts.DataLayout) = any(isnan, parent(x))
    has_nan(_) = false
    bc = Infiltrator.safehouse.args[2]
    @structured_print bc Options(; highlight = has_nan)
    ```

    The output lists the fields of `bc` (`f`, `args`, `axes`, …) with their
    types, and the argument that carries the `NaN` is printed in red.

## Caveats

  - The hook sees `ClimaCore` operations only. A `NaN` written through
    internals, such as `parent(data) .= NaN`, is not caught until a later
    `ClimaCore` operation reads it.
  - `post_op_callback` runs after every operation, so an expensive callback
    slows the run in proportion.
  - Do not combine the hook with `@testset`: Test.jl keeps running after an
    error until the set completes, so the state you inspect is the last
    occurrence, not the first.

## Reuse a state after `deepcopy`

Exploring alternatives from a spun-up state is easiest by advancing a
`deepcopy` of it, so that the original is kept for the next copy. `ClimaCore`
checks that fields in one broadcast live on the same space by object identity,
and a `deepcopy` creates a new space object, so a broadcast that mixes the copy
with fields on the original space raises a mismatched-spaces error.
`DebugOnly.allow_mismatched_spaces_unsafe` turns that check off:

```julia
import ClimaCore
other_space = deepcopy(space)
ones(space) .+ ones(other_space)                          # error: mismatched spaces
ClimaCore.DebugOnly.allow_mismatched_spaces_unsafe() = true
ones(space) .+ ones(other_space)                          # allowed
```

The check exists to prevent meaningless results from fields on different
grids; with it off, you are responsible for making sure the spaces are in fact
equivalent.
