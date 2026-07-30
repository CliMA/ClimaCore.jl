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
import ClimaCore.Operators:
    axis_vals,
    axis_index,
    curl_term,
    replace_index,
    sum_axes
import UnrolledUtilities: unrolled_map
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

# The methods below serve every dimension and both forms of each operator:
# form_deriv_entry supplies the derivative matrix entries (transposed and
# sign-flipped for the weak form), form_jacobian_rescale or form_weight_rescale
# divides the form's Jacobian or quadrature-weight factor back out of the result,
# and `sum_axes` unrolls the loop over axes, keeping one accumulator per
# dimension.

"""
    apply_stencil(form, D, w, node, ::Val{d}, i, Nq)

``\\sum_k D[i, k] w_k``, where `w_k` is the value of the shared-memory array `w`
at `node` with its `d`th index replaced by `k`. This is the one-dimensional
spectral stencil for output node `i` applied along axis `d`; `form` selects the
matrix entry, so the weak form transposes `D` and flips its sign.
"""
Base.@propagate_inbounds function apply_stencil(form, D, w, node, vd, i, Nq)
    r = form_deriv_entry(form, D, i, 1) * w[replace_index(node, vd, 1)]
    for k in 2:Nq
        r += form_deriv_entry(form, D, i, k) * w[replace_index(node, vd, k)]
    end
    return r
end

"""
    curl_stencil(form, D, work, node, ::Val{d}, i, Nq)

``\\sum_k ε^{i d m} D[i, k] u_m``, the axis-`d` contribution of a curl summed over
the stencil, where `work` holds the covariant components `u_m` in shared memory
(`nothing` for the components the curl does not use).
"""
Base.@propagate_inbounds function curl_stencil(form, D, work, node, vd, i, Nq)
    r = curl_term(
        vd,
        form_deriv_entry(form, D, i, 1),
        shmem_components(work, replace_index(node, vd, 1)),
    )
    for k in 2:Nq
        r += curl_term(
            vd,
            form_deriv_entry(form, D, i, k),
            shmem_components(work, replace_index(node, vd, k)),
        )
    end
    return r
end

# The values of a curl's shared-memory component arrays at `node`, keeping the
# `nothing`s that mark the components the curl does not use.
Base.@propagate_inbounds shmem_components(work, node) =
    unrolled_map(w_k -> isnothing(w_k) ? nothing : (@inbounds w_k[node]), work)

Base.@propagate_inbounds function operator_evaluate(
    op::Divergence{I, F},
    Jv,
    space,
    ij,
    slabidx,
) where {I, F}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)

    DJv = sum_axes(op) do vd
        @inline
        d = axis_index(vd)
        @inbounds apply_stencil(F(), D, Jv[d], node, vd, ij[d], Nq)
    end
    return form_jacobian_rescale(F(), local_geometry, DJv)
end

Base.@propagate_inbounds function operator_evaluate(
    op::SplitDivergence{I},
    work,
    space,
    ij,
    slabidx,
) where {I}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)
    psi = last(work)

    result = sum_axes(op) do vd
        @inline
        d = axis_index(vd)
        i = ij[d]
        Juᵈ = work[d]
        # the two-point flux Fᵈ[i,k] vanishes for k == i
        r = zero(Geometry.mul_return_type(eltype(Juᵈ), eltype(psi)))
        @inbounds for k in 1:Nq
            k == i && continue
            node_k = replace_index(node, vd, k)
            r += D[i, k] * (Juᵈ[node] + Juᵈ[node_k]) * (psi[node] + psi[node_k]) / 2
        end
        r
    end
    return result * local_geometry.invJ
end

Base.@propagate_inbounds function operator_evaluate(
    op::Gradient{I, F},
    (input,),
    space,
    ij,
    slabidx,
) where {I, F}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)

    # the covariant components of the gradient are the derivatives along each axis
    ∂f∂ξ = unrolled_map(axis_vals(op)) do vd
        d = axis_index(vd)
        @inbounds apply_stencil(F(), D, input, node, vd, ij[d], Nq)
    end
    result = if eltype(input) <: Number
        Geometry.covariant_vector(Val(I), ∂f∂ξ)
    elseif eltype(input) <: Geometry.AbstractTensor{1}
        tensor_axes =
            (Geometry.covariant_axis(Val(I)), Geometry.tensor_axes(eltype(input))[1])
        tensor_components = hcat(unrolled_map(parent, ∂f∂ξ)...)'
        Geometry.Tensor(tensor_components, tensor_axes)
    else
        error("Unsupported input type for gradient operator: $(eltype(input))")
    end
    return form_weight_rescale(F(), local_geometry, result)
end

Base.@propagate_inbounds function operator_evaluate(
    op::Curl{I, F},
    work,
    space,
    ij,
    slabidx,
) where {I, F}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)

    result = sum_axes(op) do vd
        @inline
        d = axis_index(vd)
        @inbounds curl_stencil(F(), D, work, node, vd, ij[d], Nq)
    end
    return form_jacobian_rescale(F(), local_geometry, result)
end
