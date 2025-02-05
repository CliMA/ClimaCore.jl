# Shared memory design

ClimaCore stencil operators support staggered (or collocated) finite difference
and interpolation operations. For example, the `DivergenceF2C` operator takes
an argument that lives on the cell faces and the resulting divergence
calculation lives on the cell centers. Such operations are effectively
matrix-vector multiplication and are often a significant portion of the runtime
cost for users.

Here, we outline an optimization, shared memory (or, "shmem" for short), that we
use to improve the performance of these operations.

## Motivation

A naive and simplified implementation of this operation looks like `div[i] = (f
[i+1] - f[i]) / dz[i]`. Such a calculation on the gpu (or cpu) requires `f[i]`
be read from global memory to compute the result of `div[i]` and `div[i-1]`. Not
to mention, if `f` is a `Broadcasted` object (`Broadcasted` objects behave like
arrays, and support `f[i]` behavior), then `f[i]` may require several reads and
or computations.

Reading data from global memory is often the main bottleneck for
bandwidth-limited cuda kernels. As such, we use shmem to reduce the number of global memory reads (and compute) in our kernels.

## High-level design

The high-level view of the design is:

 - The `bc::StencilBroadcasted` type has a `work` field, which is used to store
   shmem for the `bc.op` operator. The element type of the `work`
   (or parts of `work` if there are multiple parts) is the type returned by the
   `bc.op`'s `Operator.return_eltype`.
 - Recursively reconstruct the broadcasted object, allocating shmem for
   each `StencilBroadcasted` along the way that supports shmem
   (different operators require different arguments, and therefore different
   types and amounts of shmem).
 - Recursively fill the shmem for all `StencilBroadcasted`. This is done
   by reading the argument data from `getidx`
 - The destination field is filled with the result of `getidx` (as it is without
   shmem), except that we overload `getidx` (for supported `StencilBroadcasted`
   types) to retrieve the result of `getidx` via `fd_operator_evaluate`, which
   retrieves the result from the shmem, instead of global memory.




