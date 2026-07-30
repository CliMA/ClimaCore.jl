import ClimaCore: Spaces, Operators, Quadratures
import CUDA
import UnrolledUtilities: unrolled_map, unrolled_foreach
import ClimaCore.Operators: Divergence, SplitDivergence, Gradient, Curl
import ClimaCore.Operators: operator_return_eltype, get_local_geometry
import ClimaCore.Operators: form_jacobian, form_weighted_arg
import ClimaCore.Operators:
    axis_vals,
    axis_index,
    contravariant,
    curl_covariant_components,
    curl_uses_component,
    slab_dims

# These methods serve every dimension and both forms of each operator. The
# shared-memory work arrays are dimension-generic: an operator over axes `I` uses
# arrays with one `Nq` per axis in `I`, indexed by the node index truncated to
# those axes (on the GPU `ij` is always two-dimensional, with `j = 1` for
# one-dimensional spaces) plus this thread's slab index. See the
# "Dimension-generic building blocks" section of
# `src/Operators/spectralelement.jl`, and the `FormType` helpers above it for the
# strong/weak differences.

"""
    shmem_dims(op, space, Val(Nvt))

Shape of a shared-memory work array for `op`: one `Nq` per axis it works over,
plus `Nvt` slabs per block.
"""
@inline function shmem_dims(
    op::Operators.SpectralElementOperator,
    space,
    ::Val{Nvt},
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    return (slab_dims(op, Nq)..., Nvt)
end

"""
    sem_shmem(T, op, space, Val(Nvt))

A shared-memory work array with element type `T`, holding one value per node of
`op`'s slab for each of the `Nvt` slabs in the block.
"""
@inline sem_shmem(::Type{T}, op, space, valNvt) where {T} =
    CUDA.CuStaticSharedArray(T, shmem_dims(op, space, valNvt))

"""
    sem_shmem_per_axis(T, op, space, Val(Nvt))

One [`sem_shmem`](@ref) array per axis that `op` works over. The tuple is built
with `unrolled_map`, so each element comes from its own `CuStaticSharedArray`
call and the allocations are distinct.
"""
@inline sem_shmem_per_axis(::Type{T}, op, space, valNvt) where {T} =
    unrolled_map(_ -> sem_shmem(T, op, space, valNvt), axis_vals(op))

"""
    shmem_index(op, ij)

Index of node `ij` in a work array allocated by [`sem_shmem`](@ref): the node
index truncated to the axes `op` works over, plus this thread's slab index.
"""
@inline shmem_index(::Operators.SpectralElementOperator{I}, ij) where {I} =
    CartesianIndex(ntuple(d -> ij[d], Val(length(I)))..., CUDA.threadIdx().z)

# Both forms of the divergence hold one scaled contravariant component per
# dimension in shared memory; they differ only in the Jacobian factor that scales
# the components (see form_jacobian), so the arrays named Jv hold J uⁱ for the
# strong form and WJ uⁱ for the weak form.
Base.@propagate_inbounds function operator_shmem(
    space,
    valNvt::Val{Nvt},
    op::Divergence{I},
    arg,
) where {Nvt, I}
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    return sem_shmem_per_axis(RT, op, space, valNvt)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Divergence{I, F},
    Jv,
    space,
    ij,
    slabidx,
    arg,
) where {I, F}
    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)
    jacobian = form_jacobian(F(), local_geometry)
    unrolled_foreach(axis_vals(op)) do vd
        @inbounds Jv[axis_index(vd)][node] =
            jacobian * contravariant(vd, arg, local_geometry)
    end
end

Base.@propagate_inbounds function operator_shmem(
    space,
    valNvt::Val{Nvt},
    op::SplitDivergence{I},
    arg1,
    arg2,
) where {Nvt, I}
    FT = Spaces.undertype(space)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    # allocate temp output for the mass flux Juᵈ along each axis, and for psi
    Ju = sem_shmem_per_axis(JT, op, space, valNvt)
    psi = sem_shmem(eltype(arg2), op, space, valNvt)
    return (Ju..., psi)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::SplitDivergence{I},
    work,
    space,
    ij,
    slabidx,
    arg1,
    arg2,
) where {I}
    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)
    (; J) = local_geometry
    unrolled_foreach(axis_vals(op)) do vd
        @inbounds work[axis_index(vd)][node] =
            J * contravariant(vd, arg1, local_geometry)
    end
    @inbounds last(work)[node] = arg2
end

# Both forms of the gradient hold the argument in shared memory; they differ only
# in the quadrature weighting applied to it (see form_weighted_arg), so the array
# holds f for the strong form and W f for the weak form. It is wrapped in a tuple
# to match the other operators.
Base.@propagate_inbounds function operator_shmem(
    space,
    valNvt::Val{Nvt},
    op::Gradient{I},
    arg,
) where {Nvt, I}
    f = sem_shmem(eltype(arg), op, space, valNvt)
    return (f,)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{I, F},
    (f,),
    space,
    ij,
    slabidx,
    arg,
) where {I, F}
    local_geometry = get_local_geometry(space, ij, slabidx)
    @inbounds f[shmem_index(op, ij)] = form_weighted_arg(F(), local_geometry, arg)
end

# Both forms of the curl hold the covariant components of the argument in shared
# memory; they differ only in the quadrature weighting applied to those
# components (see form_weighted_arg). `curl_result_type` always returns a
# Contravariant123Vector, but a curl over axes `I` only reads the components that
# `curl_uses_component` selects: the entries of `work` for the others are
# `nothing`.
Base.@propagate_inbounds function operator_shmem(
    space,
    valNvt::Val{Nvt},
    op::Curl{I},
    arg,
) where {Nvt, I}
    ET = eltype(eltype(arg))
    return unrolled_map((Val(1), Val(2), Val(3))) do vk
        curl_uses_component(op, vk) ? sem_shmem(ET, op, space, valNvt) : nothing
    end
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Curl{I, F},
    work,
    space,
    ij,
    slabidx,
    arg,
) where {I, F}
    local_geometry = get_local_geometry(space, ij, slabidx)
    node = shmem_index(op, ij)
    u = curl_covariant_components(op, F(), arg, local_geometry)
    unrolled_foreach(work, u) do w_k, u_k
        @inbounds isnothing(w_k) || (w_k[node] = u_k)
    end
end
