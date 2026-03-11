import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore: Operators
import ClimaCore.Geometry: ⊗
import ClimaCore.RecursiveApply: rzero, ⊞
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square, CenterToCenter, FaceToFace, TwoArgFDOperator, OneArgFDOperator, has_affine_bc, FDOperatorMatrix, MultiplyColumnwiseBandMatrixField, operator_input_space
import ClimaCore.MatrixFields
import ClimaCore.Utilities
import ClimaCore
using ClimaCore.MatrixFields
using ClimaCore.Geometry
using LinearAlgebra
import UnrolledUtilities


include("matmul.jl")
recursively_replace_fd_ops(val) = val

check_if_fits_in_shmem(bc::Union{StencilBroadcasted, Broadcasted, ClimaCore.Fields.Field}) = sizeof(eltype(bc)) <= 36
check_if_fits_in_shmem(val) = sizeof(typeof(val)) <= 36


function recursively_replace_fd_ops(bc::Base.Broadcast.Broadcasted{Style, Axes, F, ARGS}) where {Style, Axes, F, ARGS}
    new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
    return Base.Broadcast.Broadcasted{Style}(bc.f, new_args, bc.axes)
end

recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op <: MatrixFields.FDOperatorMatrix, Args, Axes, Work} = bc
function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op, Args, Axes, Work}
    new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
    return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(bc.op, new_args, bc.axes, bc.work)
end

function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op <: TwoArgFDOperator, Args, Axes, Work}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[end])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            recursively_replace_fd_ops(bc.args[1]),
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[end]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(newop, new_args, bc.axes, bc.work)
    # elseif check_if_fits_in_shmem(bc.args[end])
    #     return bc
    #     opmat = Base.Broadcast.broadcasted(
    #         FDOperatorMatrix(unionall_type(typeof(bc.op))()),
    #         recursively_replace_fd_ops(bc.args[1]),
    #     )
    #     new_args = (opmat, recursively_replace_fd_ops(bc.args[end]))
    #     newop = MultiplyColumnwiseBandMatrixField()
    #     inner_bc =  StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(newop, new_args, bc.axes, bc.work)
    #     outer_op = Operators.SetBoundaryOperator(bc.op.bcs...)
        # new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
        # return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(bc.op, new_args, bc.axes, bc.work)
    else
        @show bc.op.bcs
        println("\n")
        return bc
    end
end

function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op <: OneArgFDOperator, Args, Axes, Work}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[1])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end])))
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[1]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(newop, new_args, bc.axes, bc.work)
    elseif check_if_fits_in_shmem(bc.args[1]) && UnrolledUtilities.unrolled_all(Base.Fix2(isa, ClimaCore.Operators.SetValue), values(bc.op.bcs))
        if bc.op isa MatrixFields.OneArgFDOperatorWithCenterInput
            opmat = Base.Broadcast.broadcasted(
                FDOperatorMatrix(bc.op),
                Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end])))
            )
            inner_args = (opmat, recursively_replace_fd_ops(bc.args[1]))
            inner_op = MultiplyColumnwiseBandMatrixField()
            inner_bc = StencilBroadcasted{Style, typeof(inner_op), typeof(inner_args), Axes, Work}(inner_op, inner_args, bc.axes, bc.work)
            outer_op = Operators.SetBoundaryOperator(; bc.op.bcs...)
            return StencilBroadcasted{Style, typeof(outer_op), Tuple{typeof(inner_bc)}, Axes, Work}(outer_op, (inner_bc,), bc.axes, bc.work)
        else
            opmat = Base.Broadcast.broadcasted(
                FDOperatorMatrix(Utilities.unionall_type(typeof(bc.op))()),
                Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end])))
            )
            inner_args = (recursively_replace_fd_ops(bc.args[1]),)
            inner_op = Operators.SetBoundaryOperator(; bc.op.bcs...)
            inner_axes =axes(inner_args[1])
            inner_bc = StencilBroadcasted{Style, typeof(inner_op), typeof(inner_args), typeof(inner_axes), Work}(inner_op, inner_args, inner_axes, bc.work)
            outer_args = (opmat, inner_bc)
            outer_op = MultiplyColumnwiseBandMatrixField()
            return StencilBroadcasted{Style, typeof(outer_op), typeof(outer_args), Axes, Work}(outer_op, outer_args, bc.axes, bc.work)
            return bc
        end
    else
        @show bc.op.bcs
        println("\n")
        return bc
    end
end


Base.@propagate_inbounds function new_stencil_entry!(out, bc::BC, space,) where {BC}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # v == 1 && hidx == (1, 1, 1) && @cushow eltype(bc)
    val = @inbounds @inline calc_level_val(bc, space)
    # val = @inline @inbounds ClimaCore.RecursiveApply.rzero(eltype(bc))
    if space.staggering isa ClimaCore.Grids.CellFace
        @inline @inbounds setidx!(space, out,  v - half, hidx, val)
    else
        if v != Int32(64)
            @inline @inbounds setidx!(space, out,  v , hidx, val)
        end
    end
    return nothing
end


Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <: Base.Broadcast.Broadcasted}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    @inline @inbounds resolved_args = UnrolledUtilities.unrolled_map(Base.Fix2(reconstruct_space_and_call_calc_level_val, space), bc.args)
    if space isa ClimaCore.Spaces.AbstractSpace && space.staggering isa ClimaCore.Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds ClimaCore.RecursiveApply.rzero(eltype(bc))
    end
    
    return @inline @inbounds bc.f(resolved_args...)
end

Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(arg::A, space::S) where {A <: Union{ Base.Broadcast.Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field}, S} = calc_level_val(arg, ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space))
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(arg::A, space::S) where {A, S} = calc_level_val(arg, space)

Base.@propagate_inbounds calc_level_val(val::T, space) where {T <: Ref} = val[]
Base.@propagate_inbounds calc_level_val(val::T, space) where {V, T <: Tuple{V}} = val[Int32(1)]
Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {S, Op <: ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField, BC <: StencilBroadcasted{S, Op}}
        v = threadIdx().x
        mat1_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[Int32(1)]), space)

        @inline @inbounds mat2_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[Int32(2)]), space)

        mat2_row = @inline @inbounds calc_level_val(bc.args[Int32(2)], mat2_space)
        mat1_row = @inline @inbounds calc_level_val(bc.args[Int32(1)], mat1_space)
        mat2_row_converted = @inline @inbounds project_row2_for_mul(mat1_row, mat2_row, mat2_space)
        CUDA.sync_threads()
        mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), Int32(64))
        @inbounds mat2[v] = mat2_row_converted
        CUDA.sync_threads()
        mat1_space.staggering isa ClimaCore.Spaces.CellCenter && v == Int32(64) && return rzero(eltype(bc))
        if mat1_space.staggering isa ClimaCore.Spaces.CellCenter
            mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
        else
            mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
        end

        if mat2_row_converted isa ClimaCore.MatrixFields.BandMatrixRow
            if mat2_space.staggering isa ClimaCore.Spaces.CellCenter
                mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
            else
                mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
            end

            return @inline @inbounds row_mul_mat!(eltype(bc), mat1_row, mat2, mat1_shape, mat2_shape)
        else
             return @inline @inbounds row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        end
end

Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {S, Op <: Operators.SetBoundaryOperator, BC <: StencilBroadcasted{S, Op}}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # we know it is on face staggered space
    v_half =  v - half
    inner_val = @inline @inbounds calc_level_val(bc.args[Int32(1)], space)
    if ClimaCore.Operators.should_call_left_boundary(v_half, space, bc.op, nothing)
        lloc = ClimaCore.Operators.left_boundary_window(space)
        left_bndry = ClimaCore.Operators.get_boundary(bc.op, lloc)
        return @inbounds @inline calc_level_val(left_bndry.val, space)
    elseif ClimaCore.Operators.should_call_right_boundary(v_half, space, bc.op, nothing)
        rroc = ClimaCore.Operators.right_boundary_window(space)
        right_bndry = ClimaCore.Operators.get_boundary(bc.op, rroc)
        return @inbounds @inline calc_level_val(right_bndry.val, space)
    else
        return @inbounds @inline inner_val
    end
end


Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <: StencilBroadcasted}

    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    if space.staggering isa ClimaCore.Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds rzero(eltype(bc))
    end
    li = space.staggering isa ClimaCore.Spaces.CellCenter ? Int32(1) : half
    idx = v - Int32(1) + li
    return @inline @inbounds Operators.getidx(space, bc, idx, hidx)
end

Base.@propagate_inbounds function calc_level_val(arg::F, space) where {F <: ClimaCore.Fields.Field}
    data = ClimaCore.Fields.field_values(arg)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
     if space.staggering isa ClimaCore.Spaces.CellCenter
        # if eltype(data) <: ClimaCore.Geometry.LocalGeometry
        #     v == Int32(64) && return @inline @inbounds data[CartesianIndex(i, j, Int32(1), Int32(63), h)]
        # end
            v == Int32(64) && return @inline @inbounds rzero(eltype(data))
    end
    return @inline @inbounds data[CartesianIndex(i, j, Int32(1), v, h)]
end

calc_level_val(arg::S, space) where {S} = arg



Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {S, BC <:ClimaCore.Operators.StencilBroadcasted{S, <: ClimaCore.MatrixFields.FDOperatorMatrix}}
    op = bc.op.op
    args = bc.args
    val = @inline @inbounds get_op_row(op, args, space)
    CUDA.sync_warp()
    return val
end


Base.@propagate_inbounds function get_op_row(op, args, space)
    FT = ClimaCore.Spaces.undertype(space)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)

    outputs_to_face = space.staggering isa ClimaCore.Grids.CellFace
    row_type = ClimaCore.MatrixFields.op_matrix_row_type(op, FT, args[1:end-1]...)
    if !outputs_to_face && v==Int32(64)
        return rzero(row_type)
    end
    v_half = outputs_to_face ? v - half : v
    in_left_bnd =  ClimaCore.Operators.should_call_left_boundary(v_half, space, op, nothing)
    in_right_bnd = ClimaCore.Operators.should_call_right_boundary(v_half, space, op, nothing)
    op_matrix = ClimaCore.MatrixFields.FDOperatorMatrix(op)

    if in_left_bnd
        lloc = ClimaCore.Operators.left_boundary_window(space)
        left_bndry = ClimaCore.Operators.get_boundary(op, lloc)
        raw_val =  ClimaCore.Operators.stencil_left_boundary(op_matrix, left_bndry, space, v_half, hidx, args...)
        val =  convert(row_type, raw_val)
    elseif in_right_bnd
        rroc = ClimaCore.Operators.right_boundary_window(space)
        right_bndry = ClimaCore.Operators.get_boundary(op, rroc)
        raw_val =  ClimaCore.Operators.stencil_right_boundary(op_matrix, right_bndry, space, v_half, hidx, args...)
        val =  convert(row_type, raw_val)
    else
        raw_val =  ClimaCore.Operators.stencil_interior(op_matrix, space, v_half, hidx, args...)
        val =  convert(row_type, raw_val)
    end
    return val
end



Base.@propagate_inbounds function project_row2_for_mul(mat1_row, mat2_row, space)

    if !ClimaCore.Geometry.needs_projection(typeof(mat1_row), typeof(mat2_row))
         return mat2_row
    end
    v = threadIdx().x
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    project_onto = ClimaCore.Geometry.recursively_find_dual_axes_for_projection(typeof(mat1_row))
    # # this is a hack to get the correct type for the 64th lvl on a cc grid
    if space.staggering isa ClimaCore.Spaces.CellCenter && v==Int32(64)
        @inbounds lg = Geometry.LocalGeometry(space, Int32(63), hidx)
    else
        v_maybe_half = space.staggering isa ClimaCore.Spaces.CellFace  ? v - half : v
        @inbounds lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
    end
    # put needed info into tuple so we can use Base.Fix2
    projection_tuple = (project_onto, lg)
    return @inline @inbounds ClimaCore.MatrixFields.recursively_project(projection_tuple, mat2_row)
end

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(calc_level_val)
        m.recursion_relation = dont_limit
    end
end
