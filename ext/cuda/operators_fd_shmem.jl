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
    # allocate temp output and geometry cache
    RT = return_eltype(op, args...)
    FT = eltype(RT)  # Get the float type from return type
    Ju³ = CUDA.CuStaticSharedArray(RT, interior_size(shmem_params))
    lJu³ = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params))
    rJu³ = CUDA.CuStaticSharedArray(RT, boundary_size(shmem_params))
    # Cache invJ to avoid repeated global memory reads
    invJ_shmem = CUDA.CuStaticSharedArray(FT, interior_size(shmem_params))
    return (Ju³, lJu³, rJu³, invJ_shmem)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.DivergenceF2C,
    (Ju³, lJu³, rJu³, invJ_shmem),
    bc_bds,
    arg_space,
    space,
    idx::Utilities.PlusHalf,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        lg = Geometry.LocalGeometry(space, idx, hidx)
        
        # Cache invJ for use in evaluate - each face level stores invJ for the center below it
        if !on_boundary(idx, space, op)
            u³ = Operators.getidx(space, arg, idx, hidx)
            Ju³[vt, ti, tj] = Geometry.Jcontravariant3(u³, lg)
            # Cache invJ for the center at index vt (center below this face)
            invJ_shmem[vt, ti, tj] = lg.invJ
        elseif on_left_boundary(idx, space, op)
            bloc = Operators.left_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            if bc isa Operators.SetValue
                lJu³[ti, tj] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                lJu³[ti, tj] = ub
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        elseif on_right_boundary(idx, space, op)
            bloc = Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            if bc isa Operators.SetValue
                rJu³[ti, tj] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                rJu³[ti, tj] = ub
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        end
    end
    return nothing
end

Base.@propagate_inbounds function fd_operator_evaluate(
    op::Operators.DivergenceF2C,
    (Ju³, lJu³, rJu³, invJ_shmem),
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        # Use cached invJ instead of reading LocalGeometry from global memory
        invJ = invJ_shmem[vt, ti, tj]
        if !on_boundary(idx, space, op)
            Ju³₋ = Ju³[vt, ti, tj]     # corresponds to idx - half
            Ju³₊ = Ju³[vt + 1, ti, tj] # corresponds to idx + half
            return (Ju³₊ ⊟ Ju³₋) ⊠ invJ
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetDivergence
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = lJu³[ti, tj]       # corresponds to idx - half
                    Ju³₊ = Ju³[vt + 1, ti, tj] # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ invJ
                else
                    return lJu³[ti, tj]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = Ju³[vt, ti, tj]    # corresponds to idx - half
                    Ju³₊ = rJu³[ti, tj]       # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ invJ
                else
                    @assert bc isa Operators.SetDivergence
                    return rJu³[ti, tj]
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
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        cov3 = Geometry.Covariant3Vector(1)
        if in_domain(idx, arg_space)
            u[vt, ti, tj] = cov3 ⊗ Operators.getidx(space, arg, idx, hidx)
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
            if bc isa Operators.SetValue
                if on_left_boundary(idx, space)
                    lb[ti, tj] = cov3 ⊗ ub
                else
                    rb[ti, tj] = cov3 ⊗ ub
                end
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                if on_left_boundary(idx, space)
                    lb[ti, tj] = Geometry.project(Geometry.Covariant3Axis(), ub, lg)
                else
                    rb[ti, tj] = Geometry.project(Geometry.Covariant3Axis(), ub, lg)
                end
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
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            u₋ = u[vt - 1, ti, tj]   # corresponds to idx - half
            u₊ = u[vt, ti, tj]       # corresponds to idx + half
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
                    u₋ = 2 * lb[ti, tj]       # corresponds to idx - half
                    u₊ = 2 * u[vt, ti, tj]    # corresponds to idx + half
                    return u₊ ⊟ u₋
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * u[vt - 1, ti, tj]   # corresponds to idx - half
                    u₊ = 2 * rb[ti, tj]          # corresponds to idx + half
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
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        ᶜidx = get_cent_idx(idx)
        if in_domain(idx, arg_space)
            u[vt, ti, tj] = Operators.getidx(space, arg, idx, hidx)
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
                u[vt, ti, tj] = Operators.getidx(space, arg, idx, hidx)
                return nothing
            end
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            if bc isa Operators.SetValue
                if on_left_boundary(idx, space)
                    lb[ti, tj] = ub
                else
                    rb[ti, tj] = ub
                end
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                if on_left_boundary(idx, space)
                    lb[ti, tj] = Geometry.covariant3(ub, lg)
                else
                    rb[ti, tj] = Geometry.covariant3(ub, lg)
                end
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
        vt = threadIdx().x  # vertical level
        ti = threadIdx().y  # horizontal i
        tj = threadIdx().z  # horizontal j
        lg = Geometry.LocalGeometry(space, idx, hidx)
        ᶜidx = get_cent_idx(idx)
        if !on_boundary(idx, space, op)
            u₋ = u[vt - 1, ti, tj]   # corresponds to idx - half
            u₊ = u[vt, ti, tj]       # corresponds to idx + half
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
                    return lb[ti, tj]
                elseif bc isa Operators.SetGradient
                    u₋ = lb[ti, tj]            # corresponds to idx - half
                    u₊ = u[vt, ti, tj]         # corresponds to idx + half
                    return u₊ ⊟ RecursiveApply.rdiv(u₋, 2)
                else
                    @assert bc isa Operators.Extrapolate
                    return u[vt, ti, tj]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    return rb[ti, tj]
                elseif bc isa Operators.SetGradient
                    u₋ = u[vt - 1, ti, tj]     # corresponds to idx - half
                    u₊ = rb[ti, tj]            # corresponds to idx + half
                    return u₋ ⊞ RecursiveApply.rdiv(u₊, 2)
                else
                    @assert bc isa Operators.Extrapolate
                    return u[vt - 1, ti, tj]
                end
            end
        end
    end
end
