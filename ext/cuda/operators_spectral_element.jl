import ClimaCore: Spaces, Quadratures, Topologies
import ClimaComms
using CUDA: @cuda
import ClimaCore.Operators: AbstractSpectralStyle, strip_space
import ClimaCore.Operators: SpectralBroadcasted, set_node!, get_node
import Base.Broadcast: Broadcasted

"""
    CUDASpectralStyle()

Applies spectral-element operations by using threads for each node, and
synchronizing when they occur. This is used for GPU kernels.
"""
struct CUDASpectralStyle <: AbstractSpectralStyle end

AbstractSpectralStyle(::ClimaComms.CUDADevice) = CUDASpectralStyle

Base.@propagate_inbounds function get_node(
    space,
    sbc::SpectralBroadcasted{CUDASpectralStyle},
    ij,
    slabidx,
)
    Operators.operator_evaluate(sbc.op, sbc.work, sbc.axes, ij, slabidx)
end

function Base.copyto!(
    out::Field,
    sbc::Union{
        SpectralBroadcasted{CUDASpectralStyle},
        Broadcasted{CUDASpectralStyle},
    },
)
    space = axes(out)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    max_threads = 256
    @assert Nq * Nq ≤ max_threads
    Nvthreads = fld(max_threads, Nq * Nq)
    Nvblocks = cld(Nv, Nvthreads)
    # executed
    @cuda always_inline = true threads = (Nq, Nq, Nvthreads) blocks =
        (Nh, Nvblocks) copyto_spectral_kernel!(
        strip_space(out, space),
        strip_space(sbc, space),
        space,
        Val(Nvthreads),
    )
    return out
end


function copyto_spectral_kernel!(
    out::Fields.Field,
    sbc,
    space,
    ::Val{Nvt},
) where {Nvt}
    @inbounds begin
        i = threadIdx().x
        j = threadIdx().y
        k = threadIdx().z
        h = blockIdx().x
        vid = k + (blockIdx().y - 1) * blockDim().z
        # allocate required shmem

        sbc_reconstructed = reconstruct_placeholder_broadcasted(space, sbc)
        sbc_shmem = allocate_shmem(Val(Nvt), sbc_reconstructed)


        # can loop over blocks instead?
        if space isa Spaces.AbstractSpectralElementSpace
            v = nothing
        elseif space isa Spaces.FaceExtrudedFiniteDifferenceSpace
            v = vid - half
        elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace
            v = vid
        else
            error("Invalid space")
        end
        ij = CartesianIndex((i, j))
        slabidx = Fields.SlabIndex(v, h)
        # v may potentially be out-of-range: any time memory is accessed, it
        # should be checked by a call to is_valid_index(space, ij, slabidx)

        # resolve_shmem! needs to be called even when out of range, so that 
        # sync_threads() is invoked collectively
        resolve_shmem!(sbc_shmem, ij, slabidx)

        isactive = is_valid_index(space, ij, slabidx)
        if isactive
            result = get_node(space, sbc_shmem, ij, slabidx)
            set_node!(space, out, ij, slabidx, result)
        end
    end
    return nothing
end


"""
    allocate_shmem(Val(Nvt), b)

Create a new broadcasted object with necessary share memory allocated,
using `Nvt` slabs per block.
"""
@inline function allocate_shmem(::Val{Nvt}, obj) where {Nvt}
    obj
end
@inline function allocate_shmem(
    ::Val{Nvt},
    bc::Broadcasted{Style},
) where {Nvt, Style}
    Broadcasted{Style}(bc.f, _allocate_shmem(Val(Nvt), bc.args...), bc.axes)
end
@inline function allocate_shmem(
    ::Val{Nvt},
    sbc::SpectralBroadcasted{Style},
) where {Nvt, Style}
    args = _allocate_shmem(Val(Nvt), sbc.args...)
    work = operator_shmem(sbc.axes, Val(Nvt), sbc.op, args...)
    SpectralBroadcasted{Style}(sbc.op, args, sbc.axes, work)
end

@inline _allocate_shmem(::Val{Nvt}) where {Nvt} = ()
@inline _allocate_shmem(::Val{Nvt}, arg, xargs...) where {Nvt} =
    (allocate_shmem(Val(Nvt), arg), _allocate_shmem(Val(Nvt), xargs...)...)





"""
    resolve_shmem!(obj, ij, slabidx)

Recursively stores the arguments to all operators into shared memory, at the
given indices (if they are valid).

As this calls `sync_threads()`, it should be called collectively on all threads
at the same time.
"""
Base.@propagate_inbounds function resolve_shmem!(
    sbc::SpectralBroadcasted,
    ij,
    slabidx,
)
    space = axes(sbc)
    isactive = is_valid_index(space, ij, slabidx)

    _resolve_shmem!(ij, slabidx, sbc.args...)

    # we could reuse shmem if we split this up
    #==
    if isactive
        temp = compute thing to store in shmem
    end
    CUDA.sync_threads()
    if isactive
        shmem[i,j] = temp
    end
    CUDA.sync_threads()
    ===#

    if isactive
        operator_fill_shmem!(
            sbc.op,
            sbc.work,
            space,
            ij,
            slabidx,
            _get_node(space, ij, slabidx, sbc.args...)...,
        )
    end
    CUDA.sync_threads()
    return nothing
end

@inline _resolve_shmem!(ij, slabidx) = nothing
@inline function _resolve_shmem!(ij, slabidx, arg, xargs...)
    resolve_shmem!(arg, ij, slabidx)
    _resolve_shmem!(ij, slabidx, xargs...)
end


Base.@propagate_inbounds function resolve_shmem!(bc::Broadcasted, ij, slabidx)
    _resolve_shmem!(ij, slabidx, bc.args...)
    return nothing
end
Base.@propagate_inbounds function resolve_shmem!(obj, ij, slabidx)
    nothing
end
