import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Geometry: ⊗

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

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.GradientC2F,
    args...,
) where {Nvt}
    # allocate temp output
    RT = return_eltype(op, args...)
    u = CUDA.CuStaticSharedArray(RT, (Nvt+1,))
    return u
end

Base.@propagate_inbounds function fd_operator_fill_shmem_interior!(
    op::Operators.GradientC2F,
    u,
    loc, # can be any location
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x
        u[vt] = Geometry.Covariant3Vector(1) ⊗ (Operators.getidx(space, arg, loc, idx, hidx))
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_fill_shmem_left_boundary!(
    op::Operators.GradientC2F,
    bc::Operators.SetValue,
    u,
    loc,
    space,
    idx::Integer,
    hidx,
    arg,
    )
    idx == Operators.left_center_boundary_idx(space) ||
    error("Incorrect left idx")
    @inbounds begin
        vt = threadIdx().x
        u[vt] = Geometry.Covariant3Vector(1) ⊗ (Operators.getidx(space, bc.val, loc, nothing, hidx))
        # u[vt] = Geometry.Covariant3Vector(2) ⊗ (Operators.getidx(space, bc.val, loc, nothing, hidx))
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_fill_shmem_right_boundary!(
    op::Operators.GradientC2F,
    bc::Operators.SetValue,
    u,
    loc,
    space,
    idx::Integer,
    hidx,
    arg,
)
    # The right boundary is called at `idx + 1`, so we need to subtract 1 from idx (shmem is loaded at vt+1)
    idx == Operators.right_center_boundary_idx(space) || error("Incorrect right idx")
    @inbounds begin
        vt = threadIdx().x
        u[vt] = Geometry.Covariant3Vector(1) ⊗ (Operators.getidx(space, bc.val, loc, nothing, hidx))
        # lg = Geometry.LocalGeometry(space, idx, hidx)
        # u³ = Operators.getidx(space, bc.val, loc, nothing, hidx)
        # Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.GradientC2F,
    u,
    loc,
    space,
    idx::PlusHalf,
    hidx,
    args...,
)
    @inbounds begin
        vt = threadIdx().x
        CUDA.@cuprintln("idx.i = $(idx.i)")
        if idx == Operators.right_face_boundary_idx(space)
            u₋ = 2*u[vt]   # corresponds to idx - half
            u₊ = 2*u[vt + 1] # corresponds to idx + half
            CUDA.@cuprintln("idx.i, vt = $(idx.i), $vt")
        elseif idx == Operators.left_face_boundary_idx(space)
            CUDA.@cuprintln("Eval LB vt = $vt")
            u₋ = 2*u[vt]   # corresponds to idx - half
            u₊ = 2*u[vt + 1] # corresponds to idx + half
        else
            u₋ = u[vt]   # corresponds to idx - half
            u₊ = u[vt+1] # corresponds to idx + half
            # u₋ = u[vt - 1]   # corresponds to idx - half
            # u₊ = u[vt] # corresponds to idx + half
        end
        return u₊ ⊟ u₋
    end
end
