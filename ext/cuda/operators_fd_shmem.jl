import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.DivergenceF2C,
    args...,
) where {Nvt}
    # allocate temp output
    RT = return_eltype(op, args...)
    Ju³ = CUDA.CuStaticSharedArray(RT, (Nvt,))
    return Ju³
end

Base.@propagate_inbounds function fd_operator_fill_shmem_interior!(
    op::Operators.DivergenceF2C,
    Ju³,
    loc, # can be any location
    space,
    idx::Utilities.PlusHalf,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        u³ = Operators.getidx(space, arg, loc, idx, hidx)
        Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_fill_shmem_left_boundary!(
    op::Operators.DivergenceF2C,
    bc::Operators.SetValue,
    Ju³,
    loc,
    space,
    idx::Utilities.PlusHalf,
    hidx,
    arg,
)
    idx == Operators.left_face_boundary_idx(space) ||
        error("Incorrect left idx")
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        u³ = Operators.getidx(space, bc.val, loc, nothing, hidx)
        Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_fill_shmem_right_boundary!(
    op::Operators.DivergenceF2C,
    bc::Operators.SetValue,
    Ju³,
    loc,
    space,
    idx::Utilities.PlusHalf,
    hidx,
    arg,
)
    # The right boundary is called at `idx + 1`, so we need to subtract 1 from idx (shmem is loaded at vt+1)
    idx == Operators.right_face_boundary_idx(space) ||
        error("Incorrect right idx")
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        u³ = Operators.getidx(space, bc.val, loc, nothing, hidx)
        Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.DivergenceF2C,
    Ju³,
    loc,
    space,
    idx::Integer,
    hidx,
    args...,
)
    @inbounds begin
        vt = threadIdx().x
        local_geometry = Geometry.LocalGeometry(space, idx, hidx)
        Ju³₋ = Ju³[vt]   # corresponds to idx - half
        Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
        return (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
    end
end
