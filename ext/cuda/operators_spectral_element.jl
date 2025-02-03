import ClimaCore: Spaces, Quadratures, Topologies
import ClimaCore: Operators, Geometry, Quadratures, RecursiveApply
import ClimaComms
using CUDA
import ClimaCore.Operators: AbstractSpectralStyle, strip_space
import ClimaCore.Operators: SpectralBroadcasted, set_node!, get_node
import ClimaCore.Operators: get_local_geometry
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
    operator_evaluate(sbc.op, sbc.work, sbc.axes, ij, slabidx)
end

function Base.copyto!(
    out::Field,
    sbc::Union{
        SpectralBroadcasted{CUDASpectralStyle},
        Broadcasted{CUDASpectralStyle},
    },
)
    space = axes(out)
    us = UniversalSize(Fields.field_values(out))
    # executed
    p = spectral_partition(us)
    args = (
        strip_space(out, space),
        strip_space(sbc, space),
        space,
        Val(p.Nvthreads),
    )
    auto_launch!(
        copyto_spectral_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(out, out, sbc)
    return out
end


function copyto_spectral_kernel!(
    out::Fields.Field,
    sbc,
    space,
    ::Val{Nvt},
) where {Nvt}
    @inbounds begin
        # allocate required shmem
        sbc_reconstructed =
            Operators.reconstruct_placeholder_broadcasted(space, sbc)
        sbc_shmem = allocate_shmem(Val(Nvt), sbc_reconstructed)

        # can loop over blocks instead?
        (ij, slabidx) = spectral_universal_index(space)
        # v in `slabidx` may potentially be out-of-range: any time memory is
        # accessed, it should be checked by a call to is_valid_index(space, ij, slabidx)

        # resolve_shmem! needs to be called even when out of range, so that 
        # sync_threads() is invoked collectively
        resolve_shmem!(sbc_shmem, ij, slabidx)

        isactive = Operators.is_valid_index(space, ij, slabidx)
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
    isactive = Operators.is_valid_index(space, ij, slabidx)

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
            Operators._get_node(space, ij, slabidx, sbc.args...)...,
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

Base.@propagate_inbounds function operator_evaluate(
    op::Divergence{(1,)},
    (Jv¹,),
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    DJv = D[i, 1] ⊠ Jv¹[1, vt]
    for k in 2:Nq
        DJv = DJv ⊞ D[i, k] ⊠ Jv¹[k, vt]
    end
    return RecursiveApply.rmul(DJv, local_geometry.invJ)
end
Base.@propagate_inbounds function operator_evaluate(
    op::Divergence{(1, 2)},
    (Jv¹, Jv²),
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    DJv = D[i, 1] ⊠ Jv¹[1, j, vt]
    for k in 2:Nq
        DJv = DJv ⊞ D[i, k] ⊠ Jv¹[k, j, vt]
    end
    for k in 1:Nq
        DJv = DJv ⊞ D[j, k] ⊠ Jv²[i, k, vt]
    end
    return RecursiveApply.rmul(DJv, local_geometry.invJ)
end

Base.@propagate_inbounds function operator_evaluate(
    op::WeakDivergence{(1,)},
    (WJv¹,),
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    Dᵀ₁WJv¹ = D[1, i] ⊠ WJv¹[1, vt]
    for k in 2:Nq
        Dᵀ₁WJv¹ = Dᵀ₁WJv¹ ⊞ D[k, i] ⊠ WJv¹[k, vt]
    end
    return ⊟(RecursiveApply.rdiv(Dᵀ₁WJv¹, local_geometry.WJ))
end
Base.@propagate_inbounds function operator_evaluate(
    op::WeakDivergence{(1, 2)},
    (WJv¹, WJv²),
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    Dᵀ₁WJv¹ = D[1, i] ⊠ WJv¹[1, j, vt]
    Dᵀ₂WJv² = D[1, j] ⊠ WJv²[i, 1, vt]
    for k in 2:Nq
        Dᵀ₁WJv¹ = Dᵀ₁WJv¹ ⊞ D[k, i] ⊠ WJv¹[k, j, vt]
        Dᵀ₂WJv² = Dᵀ₂WJv² ⊞ D[k, j] ⊠ WJv²[i, k, vt]
    end
    return ⊟(RecursiveApply.rdiv(Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv², local_geometry.WJ))
end

Base.@propagate_inbounds function operator_evaluate(
    op::Gradient{(1,)},
    input,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    @inbounds begin
        ∂f∂ξ₁ = D[i, 1] ⊠ input[1, vt]
        for k in 2:Nq
            ∂f∂ξ₁ = ∂f∂ξ₁ ⊞ D[i, k] ⊠ input[k, vt]
        end
    end
    if eltype(input) <: Number
        return Geometry.Covariant1Vector(∂f∂ξ₁)
    elseif eltype(input) <: Geometry.AxisVector
        tensor_axes = (Geometry.Covariant1Axis(), axes(eltype(input))[1])
        tensor_components = hcat(Geometry.components(∂f∂ξ₁))'
        return Geometry.AxisTensor(tensor_axes, tensor_components)
    else
        error("Unsupported input type for gradient operator: $(eltype(input))")
    end
end
Base.@propagate_inbounds function operator_evaluate(
    op::Gradient{(1, 2)},
    input,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    @inbounds begin
        ∂f∂ξ₁ = D[i, 1] ⊠ input[1, j, vt]
        ∂f∂ξ₂ = D[j, 1] ⊠ input[i, 1, vt]
        for k in 2:Nq
            ∂f∂ξ₁ = ∂f∂ξ₁ ⊞ D[i, k] ⊠ input[k, j, vt]
            ∂f∂ξ₂ = ∂f∂ξ₂ ⊞ D[j, k] ⊠ input[i, k, vt]
        end
    end
    if eltype(input) <: Number
        return Geometry.Covariant12Vector(∂f∂ξ₁, ∂f∂ξ₂)
    elseif eltype(input) <: Geometry.AxisVector
        tensor_axes = (Geometry.Covariant12Axis(), axes(eltype(input))[1])
        tensor_components =
            hcat(Geometry.components(∂f∂ξ₁), Geometry.components(∂f∂ξ₂))'
        return Geometry.AxisTensor(tensor_axes, tensor_components)
    else
        error("Unsupported input type for gradient operator: $(eltype(input))")
    end
end

Base.@propagate_inbounds function operator_evaluate(
    op::WeakGradient{(1,)},
    (Wf,),
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ

    Dᵀ₁Wf = D[1, i] ⊠ Wf[1, vt]
    for k in 2:Nq
        Dᵀ₁Wf = Dᵀ₁Wf ⊞ D[k, i] ⊠ Wf[k, vt]
    end
    return Geometry.Covariant1Vector(⊟(RecursiveApply.rdiv(Dᵀ₁Wf, W)))
end
Base.@propagate_inbounds function operator_evaluate(
    op::WeakGradient{(1, 2)},
    (Wf,),
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ

    Dᵀ₁Wf = D[1, i] ⊠ Wf[1, j, vt]
    Dᵀ₂Wf = D[1, j] ⊠ Wf[i, 1, vt]
    for k in 2:Nq
        Dᵀ₁Wf = Dᵀ₁Wf ⊞ D[k, i] ⊠ Wf[k, j, vt]
        Dᵀ₂Wf = Dᵀ₂Wf ⊞ D[k, j] ⊠ Wf[i, k, vt]
    end
    return Geometry.Covariant12Vector(
        ⊟(RecursiveApply.rdiv(Dᵀ₁Wf, W)),
        ⊟(RecursiveApply.rdiv(Dᵀ₂Wf, W)),
    )
end

Base.@propagate_inbounds function operator_evaluate(
    op::Curl{(1,)},
    work,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        _, v₂ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, vt]
        for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rmul(D₁v₂, local_geometry.invJ),
        )
    elseif length(work) == 1
        (v₃,) = work
        D₁v₃ = D[i, 1] ⊠ v₃[1, vt]
        for k in 2:Nq
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, vt]
        end
        return Geometry.Contravariant2Vector(
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
        )
    else
        _, v₂, v₃ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, vt]
        D₁v₃ = D[i, 1] ⊠ v₃[1, vt]
        @simd for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, vt]
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, vt]
        end
        return Geometry.Contravariant23Vector(
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
            RecursiveApply.rmul(D₁v₂, local_geometry.invJ),
        )
    end
end
Base.@propagate_inbounds function operator_evaluate(
    op::Curl{(1, 2)},
    work,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        v₁, v₂ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, j, vt]
        D₂v₁ = D[j, 1] ⊠ v₁[i, 1, vt]
        for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, j, vt]
            D₂v₁ = D₂v₁ ⊞ D[j, k] ⊠ v₁[i, k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rmul(D₁v₂ ⊟ D₂v₁, local_geometry.invJ),
        )
    elseif length(work) == 1
        (v₃,) = work
        D₁v₃ = D[i, 1] ⊠ v₃[1, j, vt]
        D₂v₃ = D[j, 1] ⊠ v₃[i, 1, vt]
        for k in 2:Nq
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, j, vt]
            D₂v₃ = D₂v₃ ⊞ D[j, k] ⊠ v₃[i, k, vt]
        end
        return Geometry.Contravariant12Vector(
            RecursiveApply.rmul(D₂v₃, local_geometry.invJ),
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
        )
    else
        v₁, v₂, v₃ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, j, vt]
        D₂v₁ = D[j, 1] ⊠ v₁[i, 1, vt]
        D₁v₃ = D[i, 1] ⊠ v₃[1, j, vt]
        D₂v₃ = D[j, 1] ⊠ v₃[i, 1, vt]
        @simd for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, j, vt]
            D₂v₁ = D₂v₁ ⊞ D[j, k] ⊠ v₁[i, k, vt]
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, j, vt]
            D₂v₃ = D₂v₃ ⊞ D[j, k] ⊠ v₃[i, k, vt]
        end
        return Geometry.Contravariant123Vector(
            RecursiveApply.rmul(D₂v₃, local_geometry.invJ),
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
            RecursiveApply.rmul(D₁v₂ ⊟ D₂v₁, local_geometry.invJ),
        )
    end
end

Base.@propagate_inbounds function operator_evaluate(
    op::WeakCurl{(1,)},
    work,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        _, Wv₂ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rdiv(⊟(Dᵀ₁Wv₂), local_geometry.WJ),
        )
    elseif length(work) == 1
        (Wv₃,) = work
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, vt]
        end
        return Geometry.Contravariant2Vector(
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
        )
    else
        _, Wv₂, Wv₃ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, vt]
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, vt]
        @simd for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, vt]
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, vt]
        end
        return Geometry.Contravariant23Vector(
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
            RecursiveApply.rdiv(⊟(Dᵀ₁Wv₂), local_geometry.WJ),
        )
    end
end
Base.@propagate_inbounds function operator_evaluate(
    op::WeakCurl{(1, 2)},
    work,
    space,
    ij,
    slabidx,
)
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        Wv₁, Wv₂ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, j, vt]
        Dᵀ₂Wv₁ = D[1, j] ⊠ Wv₁[i, 1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, j, vt]
            Dᵀ₂Wv₁ = Dᵀ₂Wv₁ ⊞ D[k, j] ⊠ Wv₁[i, k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rdiv(Dᵀ₂Wv₁ ⊟ Dᵀ₁Wv₂, local_geometry.WJ),
        )
    elseif length(work) == 1
        (Wv₃,) = work
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, j, vt]
        Dᵀ₂Wv₃ = D[1, j] ⊠ Wv₃[i, 1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, j, vt]
            Dᵀ₂Wv₃ = Dᵀ₂Wv₃ ⊞ D[k, j] ⊠ Wv₃[i, k, vt]
        end
        return Geometry.Contravariant12Vector(
            ⊟(RecursiveApply.rdiv(Dᵀ₂Wv₃, local_geometry.WJ)),
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
        )
    else
        Wv₁, Wv₂, Wv₃ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, j, vt]
        Dᵀ₂Wv₁ = D[1, j] ⊠ Wv₁[i, 1, vt]
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, j, vt]
        Dᵀ₂Wv₃ = D[1, j] ⊠ Wv₃[i, 1, vt]
        @simd for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, j, vt]
            Dᵀ₂Wv₁ = Dᵀ₂Wv₁ ⊞ D[k, j] ⊠ Wv₁[i, k, vt]
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, j, vt]
            Dᵀ₂Wv₃ = Dᵀ₂Wv₃ ⊞ D[k, j] ⊠ Wv₃[i, k, vt]
        end
        return Geometry.Contravariant123Vector(
            ⊟(RecursiveApply.rdiv(Dᵀ₂Wv₃, local_geometry.WJ)),
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
            RecursiveApply.rdiv(Dᵀ₂Wv₁ ⊟ Dᵀ₁Wv₂, local_geometry.WJ),
        )
    end
end
