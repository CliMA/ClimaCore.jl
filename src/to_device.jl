import Adapt
import ClimaComms

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
function to_device(
    device::ClimaComms.AbstractDevice,
    x::Union{
        DataLayouts.DataLayout,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    },
)
    return Adapt.adapt(ClimaComms.array_type(device), x)
end

to_device(::ClimaComms.CPUMultiThreaded, _) = error("Not supported")


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
to_cpu(
    x::Union{
        DataLayouts.DataLayout,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    },
) = to_device(ClimaComms.CPUSingleThreaded(), x)
