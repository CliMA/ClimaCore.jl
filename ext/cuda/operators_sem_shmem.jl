import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators:
    Divergence, WeakDivergence, Gradient, WeakGradient, Curl, WeakCurl
import ClimaCore.Operators: operator_return_eltype, get_local_geometry

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Divergence{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    Jv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    Jv² = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    return (Jv¹, Jv²)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Divergence{(1, 2)},
    (Jv¹, Jv²),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, j = ij.I

    Jv¹[i, j, vt] =
        local_geometry.J ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant1(v, local_geometry),
            arg,
        )
    Jv²[i, j, vt] =
        local_geometry.J ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant2(v, local_geometry),
            arg,
        )
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::WeakDivergence{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    Nf = DataLayouts.typesize(FT, RT)
    WJv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    WJv² = CUDA.CuStaticSharedArray(RT, (Nq, Nq, Nvt))
    return (WJv¹, WJv²)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::WeakDivergence{(1, 2)},
    (WJv¹, WJv²),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, j = ij.I

    WJv¹[i, j, vt] =
        local_geometry.WJ ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant1(v, local_geometry),
            arg,
        )
    WJv²[i, j, vt] =
        local_geometry.WJ ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant2(v, local_geometry),
            arg,
        )
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Gradient{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    IT = eltype(arg)
    input = CUDA.CuStaticSharedArray(IT, (Nq, Nq, Nvt))
    return (input,)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{(1, 2)},
    (input,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, j = ij.I
    input[i, j, vt] = arg
end


Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::WeakGradient{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    IT = eltype(arg)
    Wf = CUDA.CuStaticSharedArray(IT, (Nq, Nq, Nvt))
    return (Wf,)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::WeakGradient{(1, 2)},
    (Wf,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    i, j = ij.I
    Wf[i, j, vt] = W ⊠ arg
end


Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Curl{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector
        # input data is a Covariant12Vector field
        v₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₁, v₂)
    elseif RT <: Geometry.Contravariant12Vector
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₃,)
    elseif RT <: Geometry.Contravariant123Vector
        v₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₁, v₂, v₃)
    else
        error("invalid return type")
    end
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Curl{(1, 2)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, j = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    RT = operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        v₁, v₂ = work
        v₁[i, j, vt] = Geometry.covariant1(arg, local_geometry)
        v₂[i, j, vt] = Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant12Vector
        (v₃,) = work
        v₃[i, j, vt] = Geometry.covariant3(arg, local_geometry)
    else
        v₁, v₂, v₃ = work
        v₁[i, j, vt] = Geometry.covariant1(arg, local_geometry)
        v₂[i, j, vt] = Geometry.covariant2(arg, local_geometry)
        v₃[i, j, vt] = Geometry.covariant3(arg, local_geometry)
    end
end



Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::WeakCurl{(1, 2)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector
        # input data is a Covariant12Vector field
        Wv₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₁, Wv₂)
    elseif RT <: Geometry.Contravariant12Vector
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₃,)
    elseif RT <: Geometry.Contravariant123Vector
        Wv₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₁, Wv₂, Wv₃)
    else
        error("invalid return type")
    end
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::WeakCurl{(1, 2)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, j = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    RT = operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        Wv₁, Wv₂ = work
        Wv₁[i, j, vt] = W ⊠ Geometry.covariant1(arg, local_geometry)
        Wv₂[i, j, vt] = W ⊠ Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant12Vector
        (Wv₃,) = work
        Wv₃[i, j, vt] = W ⊠ Geometry.covariant3(arg, local_geometry)
    else
        Wv₁, Wv₂, Wv₃ = work
        Wv₁[i, j, vt] = W ⊠ Geometry.covariant1(arg, local_geometry)
        Wv₂[i, j, vt] = W ⊠ Geometry.covariant2(arg, local_geometry)
        Wv₃[i, j, vt] = W ⊠ Geometry.covariant3(arg, local_geometry)
    end
end
