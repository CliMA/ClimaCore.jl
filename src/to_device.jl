import Adapt
import ClimaComms

"""
    to_device(device, x)

Move `x` to the given `device`.

`x` is typically a `DataLayouts.DataLayout`, `Spaces.AbstractSpace`, `Fields.Field`,
or `Fields.FieldVector`, but can be anything that `Adapt.adapt` moves; this moves the
backing arrays between CPUs and GPUs in either direction. Moving to a
`ClimaComms.CPUMultiThreaded` device is not supported.

# Returns

A version of `x` with its backing arrays on `device`, as a freshly built wrapper (so
`out === x` does not hold). A move between CPU and GPU allocates new arrays; when `x`
already lives on `device`, `out` may share `x`'s arrays rather than copy them.
"""
to_device(device::ClimaComms.AbstractDevice, x) =
    Adapt.adapt(ClimaComms.array_type(device), x)

# The destination device is derived from its array type, and
# `Adapt.adapt(Array, device)` is `CPUSingleThreaded()` for every device: an
# array type carries no thread count. Moving to a CPUMultiThreaded device
# through the generic method above would therefore hand back a single-threaded
# result without saying so.
to_device(::ClimaComms.CPUMultiThreaded, _) = error(
    "to_device cannot move to a CPUMultiThreaded device: the destination is \
     derived from its array type, which gives CPUSingleThreaded for Array. \
     Use to_device(ClimaComms.CPUSingleThreaded(), x) or to_cpu(x).",
)

"""
    to_cpu(x)

Move the backing data of `x` to the CPU.

`x` is typically a `DataLayouts.DataLayout`, `Spaces.AbstractSpace`, `Fields.Field`,
or `Fields.FieldVector`, as for [`to_device`](@ref). Equivalent to
`to_device(ClimaComms.CPUSingleThreaded(), x)`.

# Returns

A version of `x` with its backing data on the CPU, as a freshly built wrapper (so
`out === x` does not hold). When `x` already lives on the CPU, `out` may share `x`'s
arrays rather than copy them.
"""
to_cpu(x) = to_device(ClimaComms.CPUSingleThreaded(), x)
