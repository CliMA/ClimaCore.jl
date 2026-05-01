import ClimaCore: DataLayouts, Spaces, Geometry, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Geometry: ⊗

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
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            u³ = Operators.getidx(space, arg, idx, hidx)
            Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
        elseif on_left_boundary(idx, space, op)
            bloc = Operators.left_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            bJu³ = on_left_boundary(idx, space) ? lJu³ : rJu³
            if bc isa Operators.SetValue
                bJu³[1] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                bJu³[1] = ub
            elseif bc isa Operators.Extrapolate # no shmem needed
            end
        elseif on_right_boundary(idx, space, op)
            bloc = Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            bJu³ = on_left_boundary(idx, space) ? lJu³ : rJu³
            if bc isa Operators.SetValue
                bJu³[1] = Geometry.Jcontravariant3(ub, lg)
            elseif bc isa Operators.SetDivergence
                bJu³[1] = ub
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
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            Ju³₋ = Ju³[vt]   # corresponds to idx - half
            Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
            return (Ju³₊ - Ju³₋) * lg.invJ
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetDivergence
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = lJu³[1]   # corresponds to idx - half
                    Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
                    return (Ju³₊ - Ju³₋) * lg.invJ
                else
                    # @assert bc isa Operators.SetDivergence
                    return lJu³[1]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = Ju³[vt]   # corresponds to idx - half
                    Ju³₊ = rJu³[1] # corresponds to idx + half
                    return (Ju³₊ - Ju³₋) * lg.invJ
                else
                    @assert bc isa Operators.SetDivergence
                    return rJu³[1]
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
        vt = threadIdx().x
        cov3 = Geometry.Covariant3Vector(1)
        if in_domain(idx, arg_space)
            u[vt] = cov3 ⊗ Operators.getidx(space, arg, idx, hidx)
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
                bu[1] = cov3 ⊗ ub
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                bu[1] = Geometry.project(Geometry.Covariant3Axis(), ub, lg)
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
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(idx, space, op)
            u₋ = u[vt - 1]   # corresponds to idx - half
            u₊ = u[vt] # corresponds to idx + half
            return u₊ - u₋
        else
            bloc =
                on_left_boundary(idx, space, op) ?
                Operators.left_boundary_window(space) :
                Operators.right_boundary_window(space)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * lb[1]   # corresponds to idx - half
                    u₊ = 2 * u[vt] # corresponds to idx + half
                    return u₊ - u₋
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * u[vt - 1]   # corresponds to idx - half
                    u₊ = 2 * rb[1] # corresponds to idx + half
                    return u₊ - u₋
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
        ᶜidx = get_cent_idx(idx)
        if in_domain(idx, arg_space)
            u[idx] = Operators.getidx(space, arg, idx, hidx)
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
                u[idx] = Operators.getidx(space, arg, idx, hidx)
                return nothing
            end
            bu = on_left_boundary(idx, space) ? lb : rb
            ub = Operators.getidx(space, bc.val, nothing, hidx)
            if bc isa Operators.SetValue
                bu[1] = ub
            elseif bc isa Operators.SetGradient
                lg = Geometry.LocalGeometry(space, idx, hidx)
                bu[1] = Geometry.covariant3(ub, lg)
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
        if !on_boundary(idx, space, op)
            u₋ = u[ᶜidx - 1]   # corresponds to idx - half
            u₊ = u[ᶜidx] # corresponds to idx + half
            return (u₊ + u₋) / 2
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
                    return lb[1]
                elseif bc isa Operators.SetGradient
                    u₋ = lb[1]   # corresponds to idx - half
                    u₊ = u[ᶜidx] # corresponds to idx + half
                    return u₊ - u₋ / 2
                else
                    @assert bc isa Operators.Extrapolate
                    return u[ᶜidx]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    return rb[1]
                elseif bc isa Operators.SetGradient
                    u₋ = u[ᶜidx - 1] # corresponds to idx - half
                    u₊ = rb[1]   # corresponds to idx + half
                    return u₋ + u₊ / 2
                else
                    @assert bc isa Operators.Extrapolate
                    return u[ᶜidx - 1]
                end
            end
        end
    end
end
