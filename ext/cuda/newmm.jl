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
import ClimaCore
using ClimaCore.MatrixFields
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


function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op, Args, Axes, Work}
    new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
    return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(bc.op, new_args, bc.axes, bc.work)
end

function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op <: TwoArgFDOperator, Args, Axes, Work}
    # check_if_fits_in_shmem(bc.args[end]) || @show eltype(bc.args[end])
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[end])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            recursively_replace_fd_ops(bc.args[1]),
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[end]))
        newop = MultiplyColumnwiseBandMatrixField()
         return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(newop, new_args, bc.axes, bc.work)
    else
        @info "fallback"
        @show bc.op
        return bc
        # new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
        # return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(bc.op, new_args, bc.axes, bc.work)
    end
end

function recursively_replace_fd_ops(bc::StencilBroadcasted{Style, Op, Args, Axes, Work}) where {Style, Op <: OneArgFDOperator, Args, Axes, Work}
    check_if_fits_in_shmem(bc.args[1]) || @show bc.args[1]
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[1])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end])))
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[1]))
        newop = MultiplyColumnwiseBandMatrixField()
         return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(newop, new_args, bc.axes, bc.work)
    else
        @info "fallback"
        @show bc.op
        return bc
        # new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
        # return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(bc.op, new_args, bc.axes, bc.work)
    end
end


Base.@propagate_inbounds function new_stencil_entry!(out, bc::BC, space,) where {BC}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
   
    val = @inbounds @inline calc_level_val(bc, space)
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
    # v == 1 && h == 1 && i == 1 && j == 1 && @cushow eltype(bc)
    # v == 1 && h == 1 && i == 1 && j == 1 && @cushow sizeof(typeof(resolved_args))
    if space isa ClimaCore.Spaces.AbstractSpace && space.staggering isa ClimaCore.Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds ClimaCore.RecursiveApply.rzero(eltype(bc))
    end
    
    return @inline @inbounds bc.f(resolved_args...)
end

Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(arg::A, space::S) where {A, S} = calc_level_val(arg, ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space))


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
        # @cuassert sizeof(typeof(mat2_row_converted)) <= Int32(32) "Projected row is too large for shared memory. Size: $(sizeof(typeof(mat2_row_converted))) bytes"
        mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), Int32(64))
        @inbounds mat2[v] = mat2_row_converted
        CUDA.sync_threads()
        mat1_space.staggering isa ClimaCore.Spaces.CellCenter && v == Int32(64) && return rzero(eltype(bc))
        # return rzero(eltype(bc))
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
            # return rzero(eltype(bc))
             return @inline @inbounds row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        end
end
Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <: StencilBroadcasted}

    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    
    # if bc.op isa Union{ClimaCore.MatrixFields.TwoArgFDOperator, ClimaCore.MatrixFields.OneArgFDOperator} && !ClimaCore.MatrixFields.has_affine_bc(bc.op) 
    #     # v == 1 && hidx == (1, 1, 1) && @cushow sizeof(eltype(mat2_arg))
    #     mat2_arg = bc.op isa ClimaCore.MatrixFields.OneArgFDOperator ? bc.args[Int32(1)] : bc.args[Int32(2)]
    #     if sizeof(eltype(mat2_arg)) <= Int32(36)
    #         @inline @inbounds arg2_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(mat2_arg), space)
    #         @inline @inbounds mat2_row = calc_level_val(mat2_arg, arg2_space)
    #         @inline @inbounds mat1_row =  bc.op isa ClimaCore.MatrixFields.OneArgFDOperator ? get_op_row(bc.op, (space,), space) : get_op_row(bc.op, (bc.args[1], space), space)
    #         @inline @inbounds mat2_row_converted = project_row2_for_mul(mat1_row, mat2_row, arg2_space)
    #         CUDA.sync_threads()
    #         mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), Int32(64))
    #         @inline @inbounds mat2[v] = mat2_row_converted
    #         CUDA.sync_threads()

    #         if space.staggering isa ClimaCore.Spaces.CellCenter
    #             mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
    #         else
    #             mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
    #         end
    #         if mat2_row_converted isa ClimaCore.MatrixFields.BandMatrixRow
    #             if arg2_space.staggering isa ClimaCore.Spaces.CellCenter
    #                 mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
    #             else
    #                 mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
    #             end
    #             return @inline @inbounds row_mul_mat!(eltype(bc), mat1_row, mat2, mat1_shape, mat2_shape)
    #         else
    #             return @inline @inbounds row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
    #         end
    #     end
    # end
    # return rzero(eltype(bc))
    if space.staggering isa ClimaCore.Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds rzero(eltype(bc))
    end
    # v == 1 && h == 1 && i == 1 && j == 1 && @cushow sizeof(eltype(bc))
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



Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, <: ClimaCore.MatrixFields.FDOperatorMatrix}}
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
    if outputs_to_face
        # @cuassert v_half <= Int32(63) + half
    else
        # @cuassert v_half <= Int32(63)
    end
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
   
    # v == 1 && hidx == (1, 1, 1) && @cushow typeof(mat2_row)
    # v == 1 && hidx == (1, 1, 1) && @cushow typeof(mat1_row)
    if !ClimaCore.Geometry.needs_projection(typeof(mat1_row), typeof(mat2_row))
         return mat2_row
    end
    v = threadIdx().x
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # v == 1 && hidx == (1, 1, 1) && @cushow typeof(mat1_row)
    # v == 1 && hidx == (1, 1, 1) && @cushow typeof(mat2_row)
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
    # return @inline ClimaCore.MatrixFields.project_for_mul(mat1_row, mat2_row, lg)
end

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(calc_level_val)
        m.recursion_relation = dont_limit
    end
end
