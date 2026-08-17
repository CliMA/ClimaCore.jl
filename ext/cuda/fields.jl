import ClimaComms
using CUDA: @cuda
import ClimaCore: Spaces, Fields
import ClimaCore.Fields: Field, FieldStyle
import ClimaCore.Fields: AbstractFieldStyle, bycolumn
import ClimaCore.Spaces: AbstractSpace, cuda_synchronize

# Reductions over Fields (sum, maximum, minimum, mean, and norm) are handled by
# the device-agnostic methods in src/Fields/mapreduce.jl, which parallelize
# over the DataScope of each Field's underlying DataLayout.

function bycolumn(fn, space::AbstractSpace, ::ClimaComms.CUDADevice)
    fn(:)
    return nothing
end

function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractFieldStyle}
    Base.Broadcast.Broadcasted{Style}(
        Adapt.adapt(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )
end

function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    bc::Base.Broadcast.Broadcasted{Style, <:Any, Type{T}},
) where {Style <: AbstractFieldStyle, T}
    Base.Broadcast.Broadcasted{Style}(
        (x...) -> T(x...),
        Adapt.adapt(to, bc.args),
        bc.axes,
    )
end

cuda_synchronize(device::ClimaComms.CUDADevice; kwargs...) =
    CUDA.synchronize(; kwargs...)
