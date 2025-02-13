import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: DivergenceF2C
import ClimaCore.Operators: return_eltype, get_local_geometry

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    ::Val{Nvt},
    op::DivergenceF2C,
    arg,
) where {Nvt}
    # allocate temp output
    RT = return_eltype(op, arg)
    Ju³ = CUDA.CuStaticSharedArray(RT, (Nvt,))
    return Ju³
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::DivergenceF2C,
    Ju³,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    vt = threadIdx().x
    lg = Geometry.LocalGeometry(space, idx - half, hidx)
    u³ = Operators.getidx(space, arg, loc, idx - half, hidx)
    Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::DivergenceF2C,
    Ju³,
    loc,
    space,
    idx,
    hidx,
    args...,
)
    vt = threadIdx().x
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₋ = Ju³[vt]   # corresponds to idx - half
    Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
    return (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end
