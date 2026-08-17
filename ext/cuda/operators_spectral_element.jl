import ClimaCore: Spaces, Quadratures, Topologies
import ClimaCore: Operators, Geometry, Quadratures
import ClimaComms
using CUDA
import ClimaCore.Operators: AbstractSpectralStyle, strip_space
import ClimaCore.Operators: SpectralBroadcasted, set_node!, get_node
import ClimaCore.Operators: get_local_geometry
import ClimaCore.Operators: Divergence, SplitDivergence, Gradient, Curl
import ClimaCore.Operators:
    form_deriv_entry, form_jacobian_rescale, form_weight_rescale
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
    };
    mask = DataLayouts.NoMask(),
)
    space = axes(out)
    out_fv = Fields.field_values(out)
    # executed
    p = spectral_partition(out_fv)
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
        args = Operators._get_node(space, ij, slabidx, sbc.args)
        operator_fill_shmem!(
            sbc.op,
            sbc.work,
            space,
            ij,
            slabidx,
            args...,
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

# The methods below serve both forms of each operator: form_deriv_entry supplies the
# derivative matrix entries (transposed and sign-flipped for the weak form), and
# form_jacobian_rescale or form_weight_rescale divides the form's Jacobian or
# quadrature-weight factor back out of the result. Every operator keeps one accumulator
# per dimension and combines them once at the end, which keeps the accumulation loops as
# independent dependency chains.
Base.@propagate_inbounds function operator_evaluate(
    op::Divergence{(1,), F},
    (Jv¹,),
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    D₁Jv¹ = form_deriv_entry(F(), D, i, 1) * Jv¹[1, vt]
    for k in 2:Nq
        D₁Jv¹ += form_deriv_entry(F(), D, i, k) * Jv¹[k, vt]
    end
    return form_jacobian_rescale(F(), local_geometry, D₁Jv¹)
end
Base.@propagate_inbounds function operator_evaluate(
    op::Divergence{(1, 2), F},
    (Jv¹, Jv²),
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    D₁Jv¹ = form_deriv_entry(F(), D, i, 1) * Jv¹[1, j, vt]
    D₂Jv² = form_deriv_entry(F(), D, j, 1) * Jv²[i, 1, vt]
    for k in 2:Nq
        D₁Jv¹ += form_deriv_entry(F(), D, i, k) * Jv¹[k, j, vt]
        D₂Jv² += form_deriv_entry(F(), D, j, k) * Jv²[i, k, vt]
    end
    return form_jacobian_rescale(F(), local_geometry, D₁Jv¹ + D₂Jv²)
end

Base.@propagate_inbounds function operator_evaluate(
    op::SplitDivergence{(1,)},
    (Ju1, psi),
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
    RT = Geometry.mul_return_type(eltype(Ju1), eltype(psi))

    local_geometry = get_local_geometry(space, ij, slabidx)

    result = zero(RT)
    for j in 1:Nq
        j == i && continue
        result +=
            D[i, j] * (Ju1[i, vt] + Ju1[j, vt]) * (psi[i, vt] + psi[j, vt]) / 2
    end
    return result * local_geometry.invJ
end
Base.@propagate_inbounds function operator_evaluate(
    op::SplitDivergence{(1, 2)},
    (Ju1, Ju2, psi),
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
    RT = Geometry.mul_return_type(eltype(Ju1), eltype(psi))

    local_geometry = get_local_geometry(space, ij, slabidx)

    result = zero(RT)
    for k in 1:Nq
        k == i && continue
        result +=
            D[i, k] *
            (Ju1[i, j, vt] + Ju1[k, j, vt]) * (psi[i, j, vt] + psi[k, j, vt]) / 2
    end
    for k in 1:Nq
        k == j && continue
        result +=
            D[j, k] *
            (Ju2[i, j, vt] + Ju2[i, k, vt]) * (psi[i, j, vt] + psi[i, k, vt]) / 2
    end
    return result * local_geometry.invJ
end

Base.@propagate_inbounds function operator_evaluate(
    op::Gradient{(1,), F},
    (input,),
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    @inbounds begin
        ∂f∂ξ₁ = form_deriv_entry(F(), D, i, 1) * input[1, vt]
        for k in 2:Nq
            ∂f∂ξ₁ += form_deriv_entry(F(), D, i, k) * input[k, vt]
        end
    end
    result = if eltype(input) <: Number
        Geometry.Covariant1Vector(∂f∂ξ₁)
    elseif eltype(input) <: Geometry.AbstractTensor{1}
        tensor_axes = (Geometry.Covariant1Axis(), Geometry.tensor_axes(eltype(input))[1])
        tensor_components = hcat(parent(∂f∂ξ₁))'
        Geometry.Tensor(tensor_components, tensor_axes)
    else
        error("Unsupported input type for gradient operator: $(eltype(input))")
    end
    return form_weight_rescale(F(), local_geometry, result)
end
Base.@propagate_inbounds function operator_evaluate(
    op::Gradient{(1, 2), F},
    (input,),
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)

    @inbounds begin
        ∂f∂ξ₁ = form_deriv_entry(F(), D, i, 1) * input[1, j, vt]
        ∂f∂ξ₂ = form_deriv_entry(F(), D, j, 1) * input[i, 1, vt]
        for k in 2:Nq
            ∂f∂ξ₁ += form_deriv_entry(F(), D, i, k) * input[k, j, vt]
            ∂f∂ξ₂ += form_deriv_entry(F(), D, j, k) * input[i, k, vt]
        end
    end
    result = if eltype(input) <: Number
        Geometry.Covariant12Vector(∂f∂ξ₁, ∂f∂ξ₂)
    elseif eltype(input) <: Geometry.AbstractTensor{1}
        tensor_axes = (Geometry.Covariant12Axis(), Geometry.tensor_axes(eltype(input))[1])
        tensor_components =
            hcat(parent(∂f∂ξ₁), parent(∂f∂ξ₂))'
        Geometry.Tensor(tensor_components, tensor_axes)
    else
        error("Unsupported input type for gradient operator: $(eltype(input))")
    end
    return form_weight_rescale(F(), local_geometry, result)
end

Base.@propagate_inbounds function operator_evaluate(
    op::Curl{(1,), F},
    work,
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    _, v₂, v₃ = work
    D₁v₂ = form_deriv_entry(F(), D, i, 1) * v₂[1, vt]
    D₁v₃ = form_deriv_entry(F(), D, i, 1) * v₃[1, vt]
    @simd for k in 2:Nq
        D₁v₂ += form_deriv_entry(F(), D, i, k) * v₂[k, vt]
        D₁v₃ += form_deriv_entry(F(), D, i, k) * v₃[k, vt]
    end
    result = Geometry.Contravariant123Vector(zero(FT), -D₁v₃, D₁v₂)
    return form_jacobian_rescale(F(), local_geometry, result)
end
Base.@propagate_inbounds function operator_evaluate(
    op::Curl{(1, 2), F},
    work,
    space,
    ij,
    slabidx,
) where {F}
    vt = threadIdx().z
    i, j = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = get_local_geometry(space, ij, slabidx)

    v₁, v₂, v₃ = work
    D₁v₂ = form_deriv_entry(F(), D, i, 1) * v₂[1, j, vt]
    D₂v₁ = form_deriv_entry(F(), D, j, 1) * v₁[i, 1, vt]
    D₁v₃ = form_deriv_entry(F(), D, i, 1) * v₃[1, j, vt]
    D₂v₃ = form_deriv_entry(F(), D, j, 1) * v₃[i, 1, vt]
    @simd for k in 2:Nq
        D₁v₂ += form_deriv_entry(F(), D, i, k) * v₂[k, j, vt]
        D₂v₁ += form_deriv_entry(F(), D, j, k) * v₁[i, k, vt]
        D₁v₃ += form_deriv_entry(F(), D, i, k) * v₃[k, j, vt]
        D₂v₃ += form_deriv_entry(F(), D, j, k) * v₃[i, k, vt]
    end
    result = Geometry.Contravariant123Vector(D₂v₃, -D₁v₃, D₁v₂ - D₂v₁)
    return form_jacobian_rescale(F(), local_geometry, result)
end
