import Adapt
import ClimaComms

"""
    to_device(device, x)

Move `x` to the given `device`.

`x` is a `DataLayouts.DataLayout`, `Spaces.AbstractSpace`, `Fields.Field`, or
`Fields.FieldVector`; this moves the backing arrays between CPUs and GPUs in either
direction.

# Returns

A copy of `x` on `device`, also when `x` already lives on `device`; the result is never
`===` to `x`.
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

A copy of `x` on the CPU; the result is never `===` to `x`.
"""
to_cpu(
    x::Union{
        DataLayouts.DataLayout,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    },
) = to_device(ClimaComms.CPUSingleThreaded(), x)
