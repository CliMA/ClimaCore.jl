# Performance tips and monitoring

This document outlines some performance tips and performance monitoring strategies.

For the most part, this document focuses on some of the common performance gotcha's that we've observed in the CliMA codebase.

There is a very good and thorough overview of performance tips in [Julia's docs](https://docs.julialang.org/en/v1/manual/performance-tips/).

## Avoiding global variables

Julia allows for [function closures](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured), which can be very handy, but can also result in performance cliffs, specifically if the captured variable is a non-constant global variable. So, for that reason, it's recommended to avoid closures when possible.

## Dynamic memory allocations

Some Julia functions incur allocations. For example, `push!` dynamically allocates memory. Sometimes, we can avoid using `push!` if the length of the container we're pushing to is known. If the length is unknown, then one can use alternative methods, for example, `map`. In addition, if `push!` is the only viable option, it's recommended to specify (if possible) the container type. For example, `Float64[]` and not `[]`. see [these docs](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-abstract-container) for more details.

## Tracking allocations

Julia's performance docs above recommends to pay close attention to allocations. Allocations can be coarsely reported with the `@time` macro and more finely reported by using `julia --track-allocation=all`. From [CodeCov.jl's docs](https://github.com/JuliaCI/Coverage.jl#memory-allocation):

Start julia with
```sh
julia --track-allocation=user
```
Then:
- Run whatever commands you wish to test. This first run is to ensure that everything is compiled (because compilation allocates memory).
- Call `Profile.clear_malloc_data()`
- Run your commands again
- Quit julia

Finally, navigate to the directory holding your source code. Start julia (without command-line flags), and analyze the results using
```julia
using Coverage
analyze_malloc(dirnames)  # could be "." for the current directory, or "src", etc.
```
This will return a vector of `MallocInfo` objects, specifying the number of bytes allocated, the file name, and the line number.
These are sorted in increasing order of allocation size.

## ReportMetrics.jl

CliMA's [ReportMetrics.jl](https://github.com/CliMA/ReportMetrics.jl) applies the strategy in the above section and provides a re-useable interface for reporting the top-most important allocations. Here is an example of it in use:

 - rep_workload.jl
 - perf.jl

```julia
# File: rep_workload.jl
import Profile

x = rand(1000)

function foo()
    s = 0.0
    for i in x
        s += i - rand()
    end
    return s
end

for i in 1:100
    foo()
end
Profile.clear_malloc_data()
for i in 1:100
    foo()
end
```

```
# perf.jl
import ReportMetrics
ReportMetrics.report_allocs(;
    job_name = "RA_example",
    run_cmd = `$(Base.julia_cmd()) --track-allocation=all rep_workload.jl`,
    dirs_to_monitor = [pwd()],
)
```

This will print out something like the following:
```
[ Info: RA_example: Number of unique allocating sites: 2
┌───────────────┬─────────────┬─────────────────────────────────────────┐
│ Allocations % │ Allocations │                    <file>:<line number> │
│       (xᵢ/∑x) │     (bytes) │                                         │
├───────────────┼─────────────┼─────────────────────────────────────────┤
│            77 │     7996800 │ ReportMetrics.jl/test/rep_workload.jl:7 │
│            23 │     2387200 │ ReportMetrics.jl/test/rep_workload.jl:6 │
└───────────────┴─────────────┴─────────────────────────────────────────┘
```

From here, one can investigate where the most important allocations are coming from. Often, allocations arise from either:
 - Using functions that inherently allocate
   - For example, `push!` inherently allocates
   - Another example: defining a new variable `a = c .+ b`. Here, `a` is a newly allocated variable. It could be put into a cache and computed in-place via `a .= c .+ b`, which is non-allocating for Julia-native types (e.g., Arrays).
 - Type instabilities. Sometimes type-instabilities can trigger the compiler to perform runtime inference, which results in allocations. So, fixing type instabilities is one way to fix / remove allocations.

## Geometry traits for broadcast

Broadcasted field operations can be optimized to use minimal local geometry data
(coordinates, J, WJ, invJ) instead of the full LocalGeometry structure, reducing
memory traffic on GPUs.

**By default, operations use full geometry** (conservative approach). To opt into
the optimization for operations that don't need metric tensors or coordinate
transformations, declare:

```julia
import ClimaCore.Geometry: geometry_requirement, NeedsMinimal

geometry_requirement(::typeof(simple_arithmetic_op)) = NeedsMinimal()
```

### When to use NeedsMinimal

Use `NeedsMinimal()` only for operations that:
- Perform simple arithmetic on field values
- Only need point coordinates (e.g., for height-dependent formulas)
- Don't perform vector coordinate transformations

### When to keep default (NeedsFull)

Keep the default for operations that:
- Convert between coordinate systems (Covariant ↔ Contravariant ↔ Local)
- Compute norms, cross products, or projections
- Use metric tensors (∂x∂ξ, ∂ξ∂x, gⁱʲ)

### Incremental optimization approach

Start with the default (`NeedsFull`) and incrementally mark safe operations as
`NeedsMinimal()` after validating correctness and measuring performance gains.

## References

 - General julia-specific [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
 - [Code-coverage while tracking allocations](https://github.com/JuliaCI/Coverage.jl#code-coverage)

 - CliMA's [ReportMetrics.jl](https://github.com/CliMA/ReportMetrics.jl)
