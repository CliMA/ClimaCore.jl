import ClimaComms
using CUDA: @cuda
import LinearAlgebra, Statistics
import ClimaCore: DataLayouts, Spaces, Grids, Fields
import ClimaCore.Fields: Field, FieldStyle
import ClimaCore.Fields: AbstractFieldStyle, bycolumn
import ClimaCore.Spaces: AbstractSpace, cuda_synchronize

function bycolumn(fn, space::AbstractSpace, ::ClimaComms.CUDADevice)
    fn(:)
    return nothing
end

function Base.sum(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    dev::ClimaComms.CUDADevice,
)
    context = ClimaComms.context(axes(field))
    localsum = mapreduce_cuda(identity, +, field, weighting = true)
    ClimaComms.allreduce!(context, parent(localsum), +)
    call_post_op_callback() && post_op_callback(localsum[], field, dev)
    return localsum[]
end

function Base.sum(fn, field::Field, dev::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localsum = mapreduce_cuda(fn, +, field, weighting = true)
    ClimaComms.allreduce!(context, parent(localsum), +)
    call_post_op_callback() && post_op_callback(localsum[], fn, field, dev)
    return localsum[]
end

function Base.maximum(fn, field::Field, dev::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmax = mapreduce_cuda(fn, max, field)
    ClimaComms.allreduce!(context, parent(localmax), max)
    call_post_op_callback() && post_op_callback(localmax[], fn, field, dev)
    return localmax[]
end

function Base.maximum(field::Field, dev::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmax = mapreduce_cuda(identity, max, field)
    ClimaComms.allreduce!(context, parent(localmax), max)
    call_post_op_callback() && post_op_callback(localmax[], fn, field, dev)
    return localmax[]
end

function Base.minimum(fn, field::Field, dev::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmin = mapreduce_cuda(fn, min, field)
    ClimaComms.allreduce!(context, parent(localmin), min)
    call_post_op_callback() && post_op_callback(localmin[], fn, field, dev)
    return localmin[]
end

function Base.minimum(field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmin = mapreduce_cuda(identity, min, field)
    ClimaComms.allreduce!(context, parent(localmin), min)
    call_post_op_callback() && post_op_callback(localmin[], fn, field, dev)
    return localmin[]
end

Statistics.mean(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.CUDADevice,
) = Base.sum(field) ./ Base.sum(ones(axes(field)))

Statistics.mean(fn, field::Field, ::ClimaComms.CUDADevice) =
    Base.sum(fn, field) ./ Base.sum(ones(axes(field)))

function LinearAlgebra.norm(
    field::Field,
    ::ClimaComms.CUDADevice,
    p::Real = 2;
    normalize = true,
)
    if p == 2
        # currently only one which supports structured types
        # TODO: perform map without allocation new field
        if normalize
            sqrt.(Statistics.mean(LinearAlgebra.norm_sqr.(field)))
        else
            sqrt.(sum(LinearAlgebra.norm_sqr.(field)))
        end
    elseif p == 1
        if normalize
            Statistics.mean(abs, field)
        else
            mapreduce_cuda(abs, +, field)
        end
    elseif p == Inf
        Base.maximum(abs, field)
    else
        if normalize
            Statistics.mean(x -> x^p, field) .^ (1 / p)
        else
            mapreduce_cuda(x -> x^p, +, field) .^ (1 / p)
        end
    end
end

function mapreduce_cuda(f, op, field::Field; weighting = false, opargs...)
    data = Fields.field_values(field)
    weighted_jacobian =
        weighting ? Spaces.weighted_jacobian(axes(field)) :
        OnesArray(parent(data))
    return mapreduce_cuda(f, op, data; weighted_jacobian, opargs...)
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
