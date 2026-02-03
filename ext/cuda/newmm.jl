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


function new_stencil_entry!(out, bc::BC, space,) where {BC}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    val = calc_level_val(bc, space)
    if space.staggering isa ClimaCore.Grids.CellFace
        setidx!(space, out,  v - half, hidx, val)
    else
        v != 64 && setidx!(space, out,  v , hidx, val)
    end
    return nothing
end

@inline function calc_level_val(bc::BC, space) where {BC <: Base.Broadcast.Broadcasted}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z

    if space.staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return ClimaCore.RecursiveApply.rzero(eltype(bc))
    end
    resolved_args = @inbounds UnrolledUtilities.unrolled_map(bc.args) do arg
        if typeof(arg) <: Union{Broadcasted, StencilBroadcasted, ClimaCore.Fields.Field}
            # return calc_level_val(arg, space)
            arg_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space)
            @inbounds calc_level_val(arg, arg_space)
        elseif (arg isa Tuple && length(arg) == 1)
            arg[1]
            elseif arg isa Ref
            arg[]
        else
             arg
        end
    end
    # v ==1 && h == 1 && i ==1 && j ==1 && @cushow typeof(resolved_args)
    # return rzero(eltype(bc))
    return @inbounds bc.f(resolved_args...)
end


@inline function calc_level_val(bc::BC, space) where {BC <: StencilBroadcasted}
    
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    if bc.op isa ClimaCore.MatrixFields.OneArgFDOperator && !(ClimaCore.MatrixFields.has_affine_bc(bc.op))
        mat1_row = get_op_row(bc.op, (), space)
        arg = bc.args[1]
        arg_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(arg), space)
        mat2_row = calc_level_val(arg, arg_space)
        mat1_shape = if bc.op isa ClimaCore.MatrixFields.OneArgFDOperatorWithCenterInput
            CenterToFace()
        else
            if bc.op isa ClimaCore.Operators.SetBoundaryOperator
                FaceToFace()
            else
                FaceToCenter()
            end
        end
        
        mat2_row_converted = project_row2_for_mul(mat1_row, mat2_row, arg_space)
        CUDA.sync_threads()
        mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), 64)
        @inbounds mat2[v] = mat2_row_converted
        CUDA.sync_threads()
        space isa ClimaCore.Spaces.CellCenter && v == 64 && return rzero(eltype(bc))
        return row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        # convert mat2_row to appropriate type
        # then dump in shmem
        # then matbec mul
    elseif bc.op isa ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField
        mat1_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[1]), space)
        mat2_space =  ClimaCore.Operators.reconstruct_placeholder_space(axes(bc.args[2]), space)
        mat1_row = calc_level_val(bc.args[1], mat1_space)
        mat2_row = calc_level_val(bc.args[2], mat2_space)
        # v == 1 && hidx == (1,1,1) && @cushow typeof(mat2_space)
        mat2_row_converted = project_row2_for_mul(mat1_row, mat2_row, mat2_space)
        CUDA.sync_threads()
        mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), 64)
        @inbounds mat2[v] = mat2_row_converted
        CUDA.sync_threads()
        mat1_space isa ClimaCore.Spaces.CellCenter && v == 64 && return rzero(eltype(bc))

        if mat1_space isa ClimaCore.Spaces.CellCenter
            mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
        else
            mat1_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : CenterToCenter()
        end

        if mat2_row_converted isa ClimaCore.MatrixFields.BandMatrixRow
            if mat2_space isa ClimaCore.Spaces.CellCenter
                mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
            else
                mat2_shape = eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : CenterToCenter()
            end
            return row_mul_mat!(eltype(bc), mat1_row, mat2, mat1_shape, mat2_shape)
        else
             return row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        end
    else
        if space.staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(eltype(bc))
        end
        op_space = Operators.reconstruct_placeholder_space(axes(bc), space)
        bds = Operators.window_bounds(op_space, bc)
        (li, lw, rw, ri) = bds
        idx = v - 1 + li
        # if idx isa ClimaCore.Utilities.PlusHalf && (idx < 2 + half || idx > 60 + half)
        #     return rzero(eltype(bc))
        # end
        # if !(idx isa ClimaCore.Utilities.PlusHalf) && (idx < 2 || idx > 62)
        #     return rzero(eltype(bc))
        # end
        # v == 1 && hidx == (1,1,1) && @cushow typeof(space)
        # v == 1 && hidx == (1,1,1) && @cushow typeof(Operators.reconstruct_placeholder_space(axes(bc), space))
        # return rzero(eltype(bc))
        return Operators.getidx(space, bc, idx, hidx)
    end
end

@inline function calc_level_val(arg::F, space) where {F <: ClimaCore.Fields.Field}
    data = ClimaCore.Fields.field_values(arg)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
     if space.staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(eltype(data))
    end
    return @inbounds data[CartesianIndex(i, j, 1, v, h)]
end

calc_level_val(arg::S, space) where {S} = arg# <: Union{ClimaCore.MatrixFields.BandMatrixRow, ClimaCore.Geometry.SingleValue}} = arg



@inline function calc_level_val(bc::BC, space) where {BC <:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, <: ClimaCore.MatrixFields.FDOperatorMatrix}}
    op = bc.op.op
    args = bc.args
    return get_op_row(op, args, space)
end

function get_op_row(op, args, space)
    FT = ClimaCore.Spaces.undertype(space)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)

    outputs_to_face = space.staggering isa ClimaCore.Grids.CellFace
    row_type = ClimaCore.MatrixFields.op_matrix_row_type(op, FT)
    if !outputs_to_face && v==64
        return rzero(row_type)
    end
    v_half = outputs_to_face ? v - half : v
    if outputs_to_face
        @cuassert v_half <= 63 + half
    else
        @cuassert v_half <= 63
    end
    in_left_bnd =  ClimaCore.Operators.should_call_left_boundary(v_half, space, op, nothing)
    in_right_bnd = ClimaCore.Operators.should_call_right_boundary(v_half, space, op, nothing)
    if in_left_bnd
        # return rzero(row_type)
        lloc = ClimaCore.Operators.left_boundary_window(space)
        left_bndry = ClimaCore.Operators.get_boundary(op, lloc)
        op_matrix = ClimaCore.MatrixFields.FDOperatorMatrix(op)
        return ClimaCore.Operators.stencil_left_boundary(op_matrix, left_bndry, space, v_half, hidx, args..., space)
        # return convert(row_type, ClimaCore.MatrixFields.op_matrix_first_row(op, left_bndry, space, v_half, hidx,))
    elseif in_right_bnd
        # return rzero(row_type)
        rroc = ClimaCore.Operators.right_boundary_window(space)
        right_bndry = ClimaCore.Operators.get_boundary(op, rroc)
        op_matrix = ClimaCore.MatrixFields.FDOperatorMatrix(op)
        return ClimaCore.Operators.stencil_right_boundary(op_matrix, right_bndry, space, v_half, hidx, args..., space)
        # return convert(row_type, ClimaCore.MatrixFields.op_matrix_last_row(op, right_bndry, space, v_half, hidx,))
    else
        op_matrix = ClimaCore.MatrixFields.FDOperatorMatrix(op)
        return ClimaCore.Operators.stencil_interior(op_matrix, space, v_half, hidx, args..., space)
        # return convert(row_type, ClimaCore.MatrixFields.op_matrix_interior_row(op, space, v_half, hidx,))
    end
end


@inline function project_row2_for_mul(mat1_row, mat2_row, space)
    # TODO: dont always need to project
    v = threadIdx().x
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # this is a hack to get the correct type for the 64th lvl on a cc grid
    if space.staggering isa ClimaCore.Spaces.CellCenter && v==64
        lg = Geometry.LocalGeometry(space, 63, hidx)
    else
        v_maybe_half = space.staggering isa ClimaCore.Spaces.CellFace  ? v - half : v
        lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
    end
    return ClimaCore.MatrixFields.project_for_mul(mat1_row, mat2_row, lg)
end
