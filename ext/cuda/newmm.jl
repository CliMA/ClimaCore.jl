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
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square, CenterToCenter, FaceToFace
import ClimaCore.MatrixFields
import ClimaCore
using ClimaCore.MatrixFields
using LinearAlgebra
import UnrolledUtilities


include("matmul.jl")


Base.@propagate_inbounds function new_stencil_entry!(out, bc::BC, space,) where {BC}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    val = calc_level_val(bc, space)
    if space.staggering isa ClimaCore.Grids.CellFace
        # @inbounds parent(out)[v, i, j, 1, h] = val
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
    # return ClimaCore.RecursiveApply.rzero(eltype(bc))
    # args_size = UnrolledUtilities.unrolled_sum(bc.args) do arg
    #     if typeof(arg) <: Union{Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field}
    #         sizeof(eltype(arg))
    #     else
    #         sizeof(arg)
    #     end
    # end
    # if args_size > Int32(32)
    #     if space.staggering isa ClimaCore.Spaces.CellCenter
    #         v == Int32(64) && return rzero(eltype(bc))
    #     end
    #     # v == 1 && h == 1 && i == 1 && j == 1 && @cushow args_size
    #     li = space.staggering isa ClimaCore.Spaces.CellCenter ? Int32(1) : half
    #     idx = v - Int32(1) + li
    #     hidx = (i, j, h)
    #     return @inline Operators.getidx(space, bc, idx, hidx)
    # end
    if bc.f == ClimaCore.RecursiveApply.rmul || bc.f == ClimaCore.RecursiveApply.radd
        # arg1_space = typeof(bc.args[1]) <: Union{Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field} ? ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[1]), space) : space
        # arg1_val = calc_level_val(bc.args[1], arg1_space)
        return UnrolledUtilities.unrolled_mapreduce(bc.f, bc.args) do arg
            if typeof(arg) <: Union{Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field}
            if space isa ClimaCore.Spaces.AbstractSpace
                arg_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space)
                @inline calc_level_val(arg, arg_space)
            else
                @inline calc_level_val(arg, space)
            end
        elseif (arg isa Tuple && length(arg) == Int32(1))
            @inbounds arg[Int32(1)]
        elseif arg isa Ref
            arg[]
        else
             arg
        end
        end

    end
    resolved_args = @inbounds UnrolledUtilities.unrolled_map(bc.args) do arg
        if typeof(arg) <: Union{Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field}
            if space isa ClimaCore.Spaces.AbstractSpace
                arg_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space)
                @inline calc_level_val(arg, arg_space)
            else
                @inline calc_level_val(arg, space)
            end
        elseif (arg isa Tuple && length(arg) == Int32(1))
            @inbounds arg[Int32(1)]
        elseif arg isa Ref
            arg[]
        else
             arg
        end
    end
    if space isa ClimaCore.Spaces.AbstractSpace && space.staggering isa ClimaCore.Spaces.CellCenter
        v == Int32(64) && return ClimaCore.RecursiveApply.rzero(eltype(bc))
    end
    # if v == 1 && h == 1 && i == 1 && j == 1 && bc.f == ClimaCore.RecursiveApply.rsub
    #     @cushow typeof(resolved_args)
    #     @cushow sizeof(resolved_args)
    #     @cushow eltype(bc)
    #     @cushow typeof(bc.f)
    # end
    return @inline bc.f(resolved_args...)
end

Base.@propagate_inbounds not_twoarg(arg) = !(arg isa ClimaCore.Operators.StencilBroadcasted && typeof(arg.op) <: ClimaCore.MatrixFields.FDOperatorMatrix && arg.op.op isa ClimaCore.MatrixFields.TwoArgFDOperator)
Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <: StencilBroadcasted}
    
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    if bc.op isa ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField && not_twoarg(bc.args[Int32(1)]) && not_twoarg(bc.args[Int32(2)])
        mat1_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[Int32(1)]), space)

        mat2_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[Int32(2)]), space)
        
        mat2_row = calc_level_val(bc.args[Int32(2)], mat2_space)
        mat1_row = calc_level_val(bc.args[Int32(1)], mat1_space)
        mat2_row_converted = project_row2_for_mul(mat1_row, mat2_row, mat2_space)
        CUDA.sync_threads()
        # @cuassert sizeof(typeof(mat2_row_converted)) <= Int32(32) "Projected row is too large for shared memory. Size: $(sizeof(typeof(mat2_row_converted))) bytes"
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
            return row_mul_mat!(eltype(bc), mat1_row, mat2, mat1_shape, mat2_shape)
        else
             return row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        end
    else
        if space.staggering isa ClimaCore.Spaces.CellCenter
            v == Int32(64) && return rzero(eltype(bc))
        end
        # v == 1 && h == 1 && i == 1 && j == 1 && @cushow sizeof(eltype(bc))
        li = space.staggering isa ClimaCore.Spaces.CellCenter ? Int32(1) : half
        idx = v - Int32(1) + li
        return Operators.getidx(space, bc, idx, hidx)
    end
end

Base.@propagate_inbounds function calc_level_val(arg::F, space) where {F <: ClimaCore.Fields.Field}
    data = ClimaCore.Fields.field_values(arg)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
     if space.staggering isa ClimaCore.Spaces.CellCenter
        if eltype(data) <: ClimaCore.Geometry.LocalGeometry
            v == Int32(64) && return @inbounds data[CartesianIndex(i, j, Int32(1), Int32(63), h)]
        end
            v == Int32(64) && return rzero(eltype(data))
    end
    return @inbounds data[CartesianIndex(i, j, Int32(1), v, h)]
end

Base.@propagate_inbounds calc_level_val(arg::S, space) where {S} = arg



Base.@propagate_inbounds function calc_level_val(bc::BC, space) where {BC <:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, <: ClimaCore.MatrixFields.FDOperatorMatrix}}
    op = bc.op.op
    args = bc.args
    val = get_op_row(op, args, space)
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
    row_type = ClimaCore.MatrixFields.op_matrix_row_type(op, FT)
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

# TODO:
function get_two_arg_op_row(op, args, space)
    FT = ClimaCore.Spaces.undertype(space)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)

    outputs_to_face = space.staggering isa ClimaCore.Grids.CellFace
    row_type = ClimaCore.MatrixFields.op_matrix_row_type(op, FT)
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
    # TODO: dont always need to project
    v = threadIdx().x
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # this is a hack to get the correct type for the 64th lvl on a cc grid
    if space.staggering isa ClimaCore.Spaces.CellCenter && v==Int32(64)
        @inbounds lg = Geometry.LocalGeometry(space, Int32(63), hidx)
    else
        v_maybe_half = space.staggering isa ClimaCore.Spaces.CellFace  ? v - half : v
        @inbounds lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
    end
    return @inlineClimaCore.MatrixFields.project_for_mul(mat1_row, mat2_row, lg)
end
