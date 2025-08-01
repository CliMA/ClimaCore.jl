import Adapt
import ClimaComms

"""
    out = to_device(device, x::Union{
        DataLayouts.AbstractData,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    })
Move `x` to the given `device`.
This is particularly useful to move different types of `Space.AbstractSpace`s,
`Fields.Field`s, and `Fields.FieldVector`s from CPUs to GPUs and viceversa.
If the input is already defined on the target device, returns a copy.
This means that `out === x` will not in general be satisfied.
"""
function to_device(
    device::ClimaComms.AbstractDevice,
    x::Union{
        DataLayouts.AbstractData,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    },
)
    return Adapt.adapt(ClimaComms.array_type(device), x)
end

# Generic fallback for other types that might need device adaptation
function to_device(device::ClimaComms.AbstractDevice, x)
    return Adapt.adapt(ClimaComms.array_type(device), x)
end

to_device(::ClimaComms.CPUMultiThreaded, _) = error("Not supported")

"""
    out = to_cpu(x::Union{
        DataLayouts.AbstractData,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    })
Move `x` backing data to the CPU.
This is particularly useful for `Space.AbstractSpace`s,
`Fields.Field`s, and `Fields.FieldVector`s.
Returns a copy.
This means that `out === x` will not in general be satisfied.
"""
to_cpu(
    x::Union{
        DataLayouts.AbstractData,
        Spaces.AbstractSpace,
        Fields.Field,
        Fields.FieldVector,
    },
) = to_device(ClimaComms.CPUSingleThreaded(), x)

# Generic to_cpu fallback
to_cpu(x) = to_device(ClimaComms.CPUSingleThreaded(), x)
