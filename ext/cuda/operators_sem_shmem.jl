import ClimaCore: DataLayouts, Spaces, Geometry, DataLayouts
import CUDA
import ClimaCore.Operators:
    Divergence,
    WeakDivergence,
    SplitDivergence,
    Gradient,
    WeakGradient,
    Curl,
    WeakCurl
import ClimaCore.Operators: operator_return_eltype, get_local_geometry

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.Divergence{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = Operators.operator_return_eltype(op, eltype(arg))
    Jv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nvt))
    return (Jv¹,)
end
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
    op::Divergence{(1,)},
    (Jv¹,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    (; J) = local_geometry
    Jv¹[i, vt] = J * Geometry.contravariant1(arg, local_geometry)
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
    (; J) = local_geometry
    Jv¹[i, j, vt] = J * Geometry.contravariant1(arg, local_geometry)
    Jv²[i, j, vt] = J * Geometry.contravariant2(arg, local_geometry)
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::WeakDivergence{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    Nf = DataLayouts.typesize(FT, RT)
    WJv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nvt))
    return (WJv¹,)
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
    op::WeakDivergence{(1,)},
    (WJv¹,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    (; WJ) = local_geometry
    WJv¹[i, vt] = WJ * Geometry.contravariant1(arg, local_geometry)
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
    (; WJ) = local_geometry
    WJv¹[i, j, vt] = WJ * Geometry.contravariant1(arg, local_geometry)
    WJv²[i, j, vt] = WJ * Geometry.contravariant2(arg, local_geometry)
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::SplitDivergence{(1,)},
    arg1,
    arg2,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    # allocate temp output for Ju1 and psi
    Ju1 = CUDA.CuStaticSharedArray(JT, (Nq, Nvt))
    psi = CUDA.CuStaticSharedArray(eltype(arg2), (Nq, Nvt))
    return (Ju1, psi)
end
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::SplitDivergence{(1, 2)},
    arg1,
    arg2,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    # allocate temp output for Ju1, Ju2, and psi
    Ju1 = CUDA.CuStaticSharedArray(JT, (Nq, Nq, Nvt))
    Ju2 = CUDA.CuStaticSharedArray(JT, (Nq, Nq, Nvt))
    psi = CUDA.CuStaticSharedArray(eltype(arg2), (Nq, Nq, Nvt))
    return (Ju1, Ju2, psi)
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::SplitDivergence{(1,)},
    (Ju1, psi),
    space,
    ij,
    slabidx,
    arg1,
    arg2,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    (; J) = local_geometry
    Ju1[i, vt] = J * Geometry.contravariant1(arg1, local_geometry)
    psi[i, vt] = arg2
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::SplitDivergence{(1, 2)},
    (Ju1, Ju2, psi),
    space,
    ij,
    slabidx,
    arg1,
    arg2,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    i, j = ij.I
    (; J) = local_geometry
    Ju1[i, j, vt] = J * Geometry.contravariant1(arg1, local_geometry)
    Ju2[i, j, vt] = J * Geometry.contravariant2(arg1, local_geometry)
    psi[i, j, vt] = arg2
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Gradient{(1,)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    return CUDA.CuStaticSharedArray(eltype(arg), (Nq, Nvt))
end
Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Gradient{(1, 2)},
    arg,
) where {Nvt}
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    return CUDA.CuStaticSharedArray(eltype(arg), (Nq, Nq, Nvt))
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{(1,)},
    input,
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, _ = ij.I
    input[i, vt] = arg
end
Base.@propagate_inbounds function operator_fill_shmem!(
    op::Gradient{(1, 2)},
    input,
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
    op::WeakGradient{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    IT = eltype(arg)
    Wf = CUDA.CuStaticSharedArray(IT, (Nq, Nvt))
    return (Wf,)
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
    op::WeakGradient{(1,)},
    (Wf,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    i, _ = ij.I
    Wf[i, vt] = W * arg
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
    Wf[i, j, vt] = W * arg
end

Base.@propagate_inbounds function operator_shmem(
    space,
    ::Val{Nvt},
    op::Curl{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector # input is a Covariant12Vector
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, v₂)
    elseif RT <: Geometry.Contravariant2Vector # input is a Covariant3Vector
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (v₃,)
    elseif RT <: Geometry.Contravariant23Vector # input is a Covariant123Vector
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, v₂, v₃)
    else
        error("invalid return type")
    end
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
    if RT <: Geometry.Contravariant3Vector # input is a Covariant12Vector
        v₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₁, v₂)
    elseif RT <: Geometry.Contravariant12Vector # input is a Covariant3Vector
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₃,)
    elseif RT <: Geometry.Contravariant123Vector # input is a Covariant123Vector
        v₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (v₁, v₂, v₃)
    else
        error("invalid return type")
    end
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::Curl{(1,)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, _ = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    RT = operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        _, v₂ = work
        v₂[i, vt] = Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant2Vector
        (v₃,) = work
        v₃[i, vt] = Geometry.covariant3(arg, local_geometry)
    else
        _, v₂, v₃ = work
        v₂[i, vt] = Geometry.covariant2(arg, local_geometry)
        v₃[i, vt] = Geometry.covariant3(arg, local_geometry)
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
    op::WeakCurl{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector # input is a Covariant12Vector
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, Wv₂)
    elseif RT <: Geometry.Contravariant2Vector # input is a Covariant3Vector
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (Wv₃,)
    elseif RT <: Geometry.Contravariant23Vector # input is a Covariant123Vector
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, Wv₂, Wv₃)
    else
        error("invalid return type")
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
    if RT <: Geometry.Contravariant3Vector # input is a Covariant12Vector
        Wv₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₁, Wv₂)
    elseif RT <: Geometry.Contravariant12Vector # input is a Covariant3Vector
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₃,)
    elseif RT <: Geometry.Contravariant123Vector # input is a Covariant123Vector
        Wv₁ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nq, Nvt))
        return (Wv₁, Wv₂, Wv₃)
    else
        error("invalid return type")
    end
end

Base.@propagate_inbounds function operator_fill_shmem!(
    op::WeakCurl{(1,)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = threadIdx().z
    i, _ = ij.I
    local_geometry = get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    RT = operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        _, Wv₂ = work
        Wv₂[i, vt] = W * Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant2Vector
        (Wv₃,) = work
        Wv₃[i, vt] = W * Geometry.covariant3(arg, local_geometry)
    else
        _, Wv₂, Wv₃ = work
        Wv₂[i, vt] = W * Geometry.covariant2(arg, local_geometry)
        Wv₃[i, vt] = W * Geometry.covariant3(arg, local_geometry)
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
        Wv₁[i, j, vt] = W * Geometry.covariant1(arg, local_geometry)
        Wv₂[i, j, vt] = W * Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant12Vector
        (Wv₃,) = work
        Wv₃[i, j, vt] = W * Geometry.covariant3(arg, local_geometry)
    else
        Wv₁, Wv₂, Wv₃ = work
        Wv₁[i, j, vt] = W * Geometry.covariant1(arg, local_geometry)
        Wv₂[i, j, vt] = W * Geometry.covariant2(arg, local_geometry)
        Wv₃[i, j, vt] = W * Geometry.covariant3(arg, local_geometry)
    end
end
