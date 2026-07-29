import ClimaCore: DataLayouts, Spaces, Geometry, Operators, Quadratures
import CUDA
import ClimaCore.Operators:
    Divergence,
    SplitDivergence,
    Gradient,
    Curl
import ClimaCore.Operators: operator_return_eltype, get_local_geometry
import ClimaCore.Operators: form_jacobian, form_weighted_arg

# Both forms of the divergence hold one scaled contravariant component per dimension in
# shared memory, so they share these methods; they differ only in the Jacobian factor that
# scales the components (see form_jacobian), so the shared arrays named Jv hold J uⁱ for the
# strong form and WJ uⁱ for the weak form.
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Divergence{(1,)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    Jv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nvt))
    return (Jv¹,)
end
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Divergence{(1, 2)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    Jv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    Jv² = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    return (Jv¹, Jv²)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Divergence{(1,), F},
    (Jv¹,),
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    jacobian = form_jacobian(F(), local_geometry)
    Jv¹[i, vt] = jacobian * Geometry.contravariant1(arg, local_geometry)
end
Base.@propagate_inbounds function operator_fill_shmem!(
    op::Divergence{(1, 2), F},
    (Jv¹, Jv²),
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, j = ij.I
    jacobian = form_jacobian(F(), local_geometry)
    Jv¹[i, j, vt] = jacobian * Geometry.contravariant1(arg, local_geometry)
    Jv²[i, j, vt] = jacobian * Geometry.contravariant2(arg, local_geometry)
end

# Both forms of the gradient hold the argument in shared memory, so they share these
# methods; they differ only in the quadrature weighting applied to it (see
# form_weighted_arg), so the shared array holds f for the strong form and W f for the weak
# form. It is wrapped in a tuple to match the other operators.
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Gradient{(1,)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    f = CUDA.CuStaticSharedArray(eltype(arg), (Nq, Nvt))
    return (f,)
end
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Gradient{(1, 2)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    f = CUDA.CuStaticSharedArray(eltype(arg), (Nq, Nq, Nvt))
    return (f,)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{(1,), F},
    (f,),
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    f[i, vt] = form_weighted_arg(F(), local_geometry, arg)
end
Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{(1, 2), F},
    (f,),
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, j = ij.I
    f[i, j, vt] = form_weighted_arg(F(), local_geometry, arg)
end

# Both forms of the curl hold the covariant components of the argument in shared memory, so
# they share these methods; they differ only in the quadrature weighting applied to those
# components (see form_weighted_arg). `curl_result_type` always returns a
# Contravariant123Vector, so all three components are allocated in 2D, while the first is
# unused in 1D.
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Curl{(1,)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    ET = eltype(eltype(arg))
    v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
    v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
    return (nothing, v₂, v₃)
end
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Curl{(1, 2)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    ET = eltype(eltype(arg))
    v₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
    v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
    v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
    return (v₁, v₂, v₃)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Curl{(1,), F},
    work,
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    i, _ = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    _, v₂, v₃ = work
    weighted(x) = form_weighted_arg(F(), local_geometry, x)
    v₂[i, vt] = weighted(Geometry.covariant2(arg, local_geometry))
    v₃[i, vt] = weighted(Geometry.covariant3(arg, local_geometry))
end
Base.@propagate_inbounds function operator_fill_shmem!(
    op::Curl{(1, 2), F},
    work,
    space,
    ij,
    slabidx,
    arg,
) where {F}
    vt = threadIdx().z
    i, j = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    v₁, v₂, v₃ = work
    weighted(x) = form_weighted_arg(F(), local_geometry, x)
    v₁[i, j, vt] = weighted(Geometry.covariant1(arg, local_geometry))
    v₂[i, j, vt] = weighted(Geometry.covariant2(arg, local_geometry))
    v₃[i, j, vt] = weighted(Geometry.covariant3(arg, local_geometry))
end
