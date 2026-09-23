import Adapt
import ClimaComms

# The argument types that `to_device`/`to_cpu` know how to move. Named so that
# the CPUMultiThreaded methods below can repeat it without drift.
const MovableToDevice = Union{
    DataLayouts.DataLayout,
    Spaces.AbstractSpace,
    Fields.Field,
    Fields.FieldVector,
}

"""
    to_device(device, x)

Move `x` to the given `device`.

`x` is a `DataLayouts.DataLayout`, `Spaces.AbstractSpace`, `Fields.Field`, or
`Fields.FieldVector`; this moves the backing arrays between CPUs and GPUs in either
direction.

# Returns

A version of `x` with its backing arrays on `device`, as a freshly built wrapper (so
`out === x` does not hold). A move between CPU and GPU allocates new arrays; when `x`
already lives on `device`, `out` may share `x`'s arrays rather than copy them.
"""
function to_device(device::ClimaComms.AbstractDevice, x::MovableToDevice)
    return Adapt.adapt(ClimaComms.array_type(device), x)
end

# The destination device is derived from its array type, and
# `Adapt.adapt(Array, device)` is `CPUSingleThreaded()` for every device: an
# array type carries no thread count. Moving to a CPUMultiThreaded device
# through the generic method above would therefore hand back a single-threaded
# result without saying so. The two methods are needed because neither the
# generic method nor a single `(::CPUMultiThreaded, _)` method is more specific
# than the other for a movable `x` (one wins on the device, the other on `x`).
_to_multithreaded_unsupported() = error(
    "to_device cannot move to a CPUMultiThreaded device: the destination is \
     derived from its array type, which gives CPUSingleThreaded for Array. \
     Use to_device(ClimaComms.CPUSingleThreaded(), x) or to_cpu(x).",
)
to_device(::ClimaComms.CPUMultiThreaded, _) = _to_multithreaded_unsupported()
to_device(::ClimaComms.CPUMultiThreaded, ::MovableToDevice) =
    _to_multithreaded_unsupported()


"""
    to_cpu(x)

Move the backing data of `x` to the CPU.

`x` is a `DataLayouts.DataLayout`, `Spaces.AbstractSpace`, `Fields.Field`, or
`Fields.FieldVector`. Equivalent to `to_device(ClimaComms.CPUSingleThreaded(), x)`.

# Returns

A version of `x` with its backing data on the CPU, as a freshly built wrapper (so
`out === x` does not hold). When `x` already lives on the CPU, `out` may share `x`'s
arrays rather than copy them.
"""
to_cpu(x::MovableToDevice) = to_device(ClimaComms.CPUSingleThreaded(), x)
