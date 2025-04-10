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
    lJu³ = CUDA.CuStaticSharedArray(RT, (1,))
    rJu³ = CUDA.CuStaticSharedArray(RT, (1,))
    return (Ju³, lJu³, rJu³)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.DivergenceF2C,
    (Ju³, lJu³, rJu³),
    loc,
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
        if !on_boundary(space, op, loc, idx)
            u³ = Operators.getidx(space, arg, loc, idx, hidx)
            Ju³[vt] = Geometry.Jcontravariant3(u³, lg)
        else
            bc = Operators.get_boundary(op, loc)
            ub = Operators.getidx(space, bc.val, loc, nothing, hidx)
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
    loc,
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(space, op, loc, idx)
            Ju³₋ = Ju³[vt]   # corresponds to idx - half
            Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
            return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
        else
            bc = Operators.get_boundary(op, loc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetDivergence
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = lJu³[1]   # corresponds to idx - half
                    Ju³₊ = Ju³[vt + 1] # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
                else
                    # @assert bc isa Operators.SetDivergence
                    return lJu³[1]
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    Ju³₋ = Ju³[vt]   # corresponds to idx - half
                    Ju³₊ = rJu³[1] # corresponds to idx + half
                    return (Ju³₊ ⊟ Ju³₋) ⊠ lg.invJ
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
    ::Val{Nvt},
    op::Operators.GradientC2F,
    args...,
) where {Nvt}
    # allocate temp output
    RT = return_eltype(op, args...)
    u = CUDA.CuStaticSharedArray(RT, (Nvt,)) # cell centers
    lb = CUDA.CuStaticSharedArray(RT, (1,)) # left boundary
    rb = CUDA.CuStaticSharedArray(RT, (1,)) # right boundary
    return (u, lb, rb)
end

Base.@propagate_inbounds function fd_operator_fill_shmem!(
    op::Operators.GradientC2F,
    (u, lb, rb),
    loc, # can be any location
    bc_bds,
    arg_space,
    space,
    idx::Integer,
    hidx,
    arg,
)
    @inbounds begin
        vt = threadIdx().x
        cov3 = Geometry.Covariant3Vector(1)
        if in_domain(idx, arg_space)
            u[vt] = cov3 ⊗ Operators.getidx(space, arg, loc, idx, hidx)
        else # idx can be Spaces.nlevels(ᶜspace)+1 because threads must extend to faces
            ᶜspace = Spaces.center_space(arg_space)
            @assert idx == Spaces.nlevels(ᶜspace) + 1
        end
        if on_any_boundary(idx, space, op)
            lloc =
                Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
            rloc = Operators.RightBoundaryWindow{
                Spaces.right_boundary_name(space),
            }()
            bloc = on_left_boundary(idx, space, op) ? lloc : rloc
            @assert bloc isa typeof(lloc) && on_left_boundary(idx, space, op) ||
                    bloc isa typeof(rloc) && on_right_boundary(idx, space, op)
            bc = Operators.get_boundary(op, bloc)
            @assert bc isa Operators.SetValue || bc isa Operators.SetGradient
            ub = Operators.getidx(space, bc.val, bloc, nothing, hidx)
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
    loc,
    space,
    idx::PlusHalf,
    hidx,
    args...,
)
    @inbounds begin
        vt = threadIdx().x
        lg = Geometry.LocalGeometry(space, idx, hidx)
        if !on_boundary(space, op, loc, idx)
            u₋ = u[vt - 1]   # corresponds to idx - half
            u₊ = u[vt] # corresponds to idx + half
            return u₊ ⊟ u₋
        else
            bc = Operators.get_boundary(op, loc)
            @assert bc isa Operators.SetValue
            if on_left_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * lb[1]   # corresponds to idx - half
                    u₊ = 2 * u[vt] # corresponds to idx + half
                    return u₊ ⊟ u₋
                end
            else
                @assert on_right_boundary(idx, space)
                if bc isa Operators.SetValue
                    u₋ = 2 * u[vt - 1]   # corresponds to idx - half
                    u₊ = 2 * rb[1] # corresponds to idx + half
                    return u₊ ⊟ u₋
                end
            end
        end
    end
end
