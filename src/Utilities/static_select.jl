import StaticArrays

"""
    static_select(v, i)

`v[i]` for a tuple or `StaticArray` `v` and a run-time index `i`, evaluated as a
branch-free chain of `ifelse`s over the compile-time indices `1:length(v)`, so
that no dynamic index into the underlying tuple is generated. An out-of-range
`i` returns `v[1]`; the caller is responsible for the bounds.

# Why this exists

Inside a GPU kernel, `v[i]` with a run-time `i` compiles to a pointer
computation with a run-time offset into the memory that holds `v`. LLVM's
scalar-replacement pass (SROA) refuses to promote any stack object that is
addressed that way, so the *whole* object stays in local memory. In a ClimaCore
broadcast kernel every argument of the broadcast is repacked into one tuple
before the point function runs, so a single `v[i]` on a small `SVector` that
arrived as a broadcast argument puts that entire argument tuple, parameter
structs included, into local memory, and every later parameter read becomes a
local-memory load. On an A100 this cost a 1.4 kB local frame and 176 local
loads per point in a microphysics kernel, and it made register capping
counter-productive. The same `v[i]` in a hand-written `@cuda` kernel costs
nothing, because there `v` is its own kernel parameter and NVPTX indexes
parameter space directly, which is why the problem is easy to miss in a
standalone reproducer.

The `ifelse` chain compiles to `N - 1` selects on constant-index loads, which
for the small `N` this is meant for (quadrature nodes, stencil coefficients) is
cheaper than the local-memory round trip and keeps SROA happy.

# Examples

```julia
julia> v = SVector(10.0, 20.0, 30.0);

julia> static_select(v, 2)
20.0

julia> static_select((1, 2, 3), 3)
3
```
"""
@inline static_select(v::Tuple, i::Integer) = _static_select(v, i, Val(length(v)))
@inline static_select(v::StaticArrays.StaticArray, i::Integer) =
    _static_select(Tuple(v), i, Val(length(v)))

@generated function _static_select(v::Tuple, i, ::Val{N}) where {N}
    N >= 1 || return :(throw(ArgumentError("static_select needs a non-empty collection")))
    ex = :(@inbounds v[1])
    for k in 2:N
        ex = :(ifelse(i == $k, @inbounds(v[$k]), $ex))
    end
    return ex
end
