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

```@example
import ClimaCore
using ClimaCore: DataLayouts
ClimaCore.DebugOnly.call_post_op_callback() = true
function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
  if any(isnan, parent(data))
    println("NaNs found!")
  end
end

FT = Float64;
data = DataLayouts.VIJFH{FT}(Array{FT}, zeros; Nv=5, Nij=2, Nh=2)
@. data = NaN
```

Note that, due to dispatch, `post_op_callback` will likely need a very general
method signature, and using `post_op_callback
(result::DataLayouts.VIJFH, args...; kwargs...)` above fails (on the CPU),
because `post_op_callback` ends up getting called multiple times with different
datalayouts.

!!! warn

    While this debugging tool may be helpful, it's not bullet proof. NaNs can
    infiltrate user data any time internals are used. For example `parent
    (data) .= NaN` will not be caught by ClimaCore.DebugOnly, and errors can be
    observed later than expected.

!!! note

    This method is called in many places, so this is a performance-critical code
    path and expensive operations performed in `post_op_callback` may
    significantly slow down your code.
