import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Geometry: ⊗
import ClimaCore.RecursiveApply: ⊟, ⊞

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    shmem_params,
    op::Operators.DivergenceF2C,
    args...,
)
    # allocate temp output
    RT = return_eltype(op, args...)
    Ju³ = CUDA.CuStaticSharedArray(RT, interior_size(shmem_params))
    lJu³ = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params))
    rJu³ = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params))
    return (Ju³, lJu³, rJu³)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.DivergenceF2C,
    (Ju³, lJu³, rJu³),
    bc_bds,
    arg_space,
    space,
    idx::Utilities.PlusHalf,
    hidx,
    arg,
)
    @inbounds begin
        si = FDShmemIndex()
        bi = FDShmemBoundaryIndex()
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            u³ = Operators.getidx(space, arg, idx, hidx)
            Ju³[si] = Geometry.Jcontravariant3(u³, lg)
        elseif on_left_boundary(idx, space, op)
            bloc = Operators.left_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            bJu³ = on_left_boundary(idx, space) ? lJu³ : rJu³
            if bc isa Operators.SetValue
                bJu³[bi] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                bJu³[bi] = ub
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        elseif on_right_boundary(idx, space, op)
            bloc = Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            bJu³ = on_left_boundary(idx, space) ? lJu³ : rJu³
            if bc isa Operators.SetValue
                bJu³[bi] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                bJu³[bi] = ub
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        end
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.DivergenceF2C,
    (Ju³, lJu³, rJu³),
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        si = FDShmemIndex()
        bi = FDShmemBoundaryIndex()
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            Ju³₋ = Ju³[si]   # corresponds to idx - half
            Ju³₊ = Ju³[si + 1] # corresponds to idx + half
            return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetDivergence
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = lJu³[bi]   # corresponds to idx - half
                    Ju³₊ = Ju³[si + 1] # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
                else
                    # @assert bc isa Operators.SetDivergence
                    return lJu³[bi]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = Ju³[si]   # corresponds to idx - half
                    Ju³₊ = rJu³[bi] # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
                else
                    @assert bc isa Operators.SetDivergence
                    return rJu³[bi]
                end
            end
        end
    end
end

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    shmem_params,
    op::Operators.GradientC2F,
    args...,
)
    # allocate temp output
    RT = return_eltype(op, args...)
    u = CUDA.CuStaticSharedArray(RT, interior_size(shmem_params)) # cell centers
    lb = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params)) # left boundary
    rb = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params)) # right boundary
    return (u, lb, rb)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.GradientC2F,
    (u, lb, rb),
    bc_bds,
    arg_space,
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        is_out_of_bounds(idx, space) && return nothing
        si = FDShmemIndex()
        bi = FDShmemBoundaryIndex()
        cov3 = Geometry.Covariant3Vector(1)
        if in_domain(idx, arg_space)
            u[si] = cov3 ⊗ Operators.getidx(space, arg, idx, hidx)
        end
        if on_any_boundary(idx, space, op)
            lloc = Operators.left_boundary_window(space)
            rloc = Operators.right_boundary_window(space)
            bloc = on_left_boundary(idx, space, op) ? lloc : rloc
            @assert bloc isa typeof(lloc) && on_left_boundary(idx, space, op) ||
                    bloc isa typeof(rloc) && on_right_boundary(idx, space, op)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetGradient
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            bu = on_left_boundary(idx, space) ? lb : rb
            if bc isa Operators.SetValue
                bu[bi] = cov3 ⊗ ub
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                bu[bi] = Geometry.project(Geometry.Covariant3Axis(), ub, lg)
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        end
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.GradientC2F,
    (u, lb, rb),
    space,
    idx::PlusHalf,
    hidx,
    args...,
)
    @inbounds begin
        si = FDShmemIndex()
        bi = FDShmemBoundaryIndex()
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            u₋ = u[si - 1]   # corresponds to idx - half
            u₊ = u[si] # corresponds to idx + half
            return u₊ ⊟ u₋
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * lb[bi]   # corresponds to idx - half
                    u₊ = 2 * u[si] # corresponds to idx + half
                    return u₊ ⊟ u₋
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * u[si - 1]   # corresponds to idx - half
                    u₊ = 2 * rb[bi] # corresponds to idx + half
                    return u₊ ⊟ u₋
                end
            end
        end
    end
end

Base.@propagate_inbounds function fd_operator_shmem(
    space,
    shmem_params,
    op::Operators.InterpolateC2F,
    args...,
)
    # allocate temp output
    RT = return_eltype(op, args...)
    u = CUDA.CuStaticSharedArray(RT, interior_size(shmem_params)) # cell centers
    lb = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params)) # left boundary
    rb = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params)) # right boundary
    return (u, lb, rb)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.InterpolateC2F,
    (u, lb, rb),
    bc_bds,
    arg_space,
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        is_out_of_bounds(idx, space) && return nothing
        si = FDShmemIndex(idx)
        bi = FDShmemBoundaryIndex()
        if in_domain(idx, arg_space)
            u[si] = Operators.getidx(space, arg, idx, hidx)
        else
            lloc = Operators.left_boundary_window(space)
            rloc = Operators.right_boundary_window(space)
            bloc = on_left_boundary(idx, space, op) ? lloc : rloc
            @assert bloc isa typeof(lloc) && on_left_boundary(idx, space, op) ||
                    bloc isa typeof(rloc) && on_right_boundary(idx, space, op)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue ||
                    bc isa Operators.SetGradient ||
                    bc isa Operators.Extrapolate ||
                    bc isa Operators.NullBoundaryCondition
            if bc isa Operators.NullBoundaryCondition ||
               bc isa Operators.Extrapolate
                u[si] = Operators.getidx(space, arg, idx, hidx)
                return nothing
            end
            bu = on_left_boundary(idx, space) ? lb : rb
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            if bc isa Operators.SetValue
                bu[bi] = ub
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                bu[bi] = Geometry.covariant3(ub, lg)
            end
        end
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.InterpolateC2F,
    (u, lb, rb),
    space,
    idx::PlusHalf,
    hidx,
    args...,
)
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        ᶜidx = get_cent_idx(idx)
        si = FDShmemIndex(ᶜidx)
        bi = FDShmemBoundaryIndex()
        if !on_boundary(idx, space, op)
            u₋ = u[si - 1]   # corresponds to idx - half
            u₊ = u[si] # corresponds to idx + half
            return RecursiveApply.rdiv(u₊ ⊞ u₋, 2)
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue ||
                    bc isa Operators.SetGradient ||
                    bc isa Operators.Extrapolate
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    return lb[bi]
                elseif bc isa Operators.SetGradient
                    u₋ = lb[bi]   # corresponds to idx - half
                    u₊ = u[si] # corresponds to idx + half
                    return u₊ ⊟ RecursiveApply.rdiv(u₋, 2)
                else
                    @assert bc isa Operators.Extrapolate
                    return u[si]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    return rb[bi]
                elseif bc isa Operators.SetGradient
                    u₋ = u[si - 1] # corresponds to idx - half
                    u₊ = rb[bi]   # corresponds to idx + half
                    return u₋ ⊞ RecursiveApply.rdiv(u₊, 2)
                else
                    @assert bc isa Operators.Extrapolate
                    return u[si - 1]
                end
            end
        end
    end
end
