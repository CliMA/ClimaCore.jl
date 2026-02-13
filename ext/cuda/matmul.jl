import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore: Operators
import ClimaCore.Geometry: ⊗
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
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


@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2,  ::FaceToCenter, ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 63

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d + half <= 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToFace, ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d - half < 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToCenter, ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 63

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  <= 63)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToFace, ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  <= 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToCenter, ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d + half  <= 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToFace, ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d - half  < 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToFace, ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  <= 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2,  ::CenterToCenter, ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = 1
        ri = 64

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  < 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

@inline function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = 1
        ri = 63

        zero_entry = rzero(prod_eltype)
        #   v == 1 && blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow eltype(matrix2)
        # v == 1 &&  blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow eltype(mat1_row)
        #   v == 1 &&  blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow prod_eltype
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (0 <  v + mat1_row_d + half <= 64)
                bb = mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d + half]
            else
                zero_entry
            end
        end
    end
end

@inline function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = 1
        ri = 64

        zero_entry = rzero(prod_eltype)
        val = zero_entry
        # for mat1_row_d in ld1:ud1
        #     if (0 <  v + mat1_row_d - half < 64)
        #         val = ClimaCore.RecursiveApply.radd(val, mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half])
        #     end
        # end
        # return val
        # v == 1 && blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow eltype(matrix2)
        # v == 1 &&  blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow typeof(mat1_row)
        #   v == 1 &&  blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1 && @cushow prod_eltype
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (0 <  v + mat1_row_d - half < 64)
                # zero_entry
                
                # if prod_eltype <: Union{Tuple, NamedTuple}
                    mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d - half]
                # else
                #     #  zero_entry
                #     mat1_row[mat1_row_d] ⊠ matrix2[v + mat1_row_d - half]
                # end
                # zero_entry
            else
                zero_entry
            end
        end
    end
end

@inline function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = 1
        ri = 63

        zero_entry = rzero(prod_eltype)
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (0 <  v + mat1_row_d  <= 63)
                mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d]
            else
                zero_entry
            end
        end
    end
end

@inline function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = 1
        ri = 64

        zero_entry = rzero(prod_eltype)
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (0 <  v + mat1_row_d  <= 64)
                mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d ]
            else
                zero_entry
            end
        end
    end
end


const COLMATMUL = ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}

function entry_mat!(out, bc::BC, space,) where {BC}#<:COLMATMUL
    mat1 = bc.args[1]
    # mat2 = bc.args[2]
    (Ni, Nj, _, Nv, Nh) = ClimaCore.DataLayouts.universal_size(ClimaCore.Fields.field_values(out))


    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    v==1 && i ==1 && j==1 && h==1 && @cushow 111111111
    mat1_staggering =  Nv == 64 ?  ClimaCore.Spaces.CellFace() : ClimaCore.Spaces.CellCenter()
    val = calc_row(mat1_staggering, bc, space)
    # i == 1 && j == 1 && h ==1 && v ==1 && @cushow ClimaCore.Geometry.components(val)[1]
    # i == 1 && j == 1 && h ==1 && v ==1 && 
    if Nv == 64
        setidx!(space, out,  v - half, hidx, val)
    else
        v != 64 && setidx!(space, out,  v , hidx, val)
    end
    return nothing
end

@inline function calc_row(mat1_staggering, bc::BC, space) where {M1<:ClimaCore.Fields.Field, M2<:ClimaCore.Fields.Field, BC <: ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField, Tuple{M1,M2}}}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)

    mat1 = bc.args[1]
    mat2 = bc.args[2]
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    mat1_data = ClimaCore.Fields.field_values(mat1)
    
    if eltype(mat2) <: ClimaCore.MatrixFields.BandMatrixRow
        (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(eltype(mat1_data))))
        # CUDA.sync_threads()
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
        # v ==1 && h==1 && @cushow typeof(matrix2)
        val = row_mul_mat!(prod_eltype, mat1_row, matrix2, mat1_shape, mat2_shape)
        return val
    else
         mat1_shape = get_mul_shape(mat1_staggering, eltype(mat1))
        mat2_shape = mat1_shape isa Union{FaceToFace, FaceToCenter} ? FaceToFace() : CenterToCenter()
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(eltype(mat1_data))))
        prod_eltype = eltype(bc)
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
        val = row_mul_vec!(prod_eltype, mat1_row, matrix2, mat1_shape)
        return val
    end
    # return zero(prod_eltype)
end





function calc_row(mat1_staggering, bc::BC, space) where {BC <: ClimaCore.Operators.StencilBroadcasted}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)

    if true && bc.op isa ClimaCore.MatrixFields.OneArgFDOperator && !(ClimaCore.MatrixFields.has_affine_bc(bc.op))
        space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
        mat1_row = get_row_from_op(space, bc.op, bc.args)
        prod_eltype = eltype(bc)
        mat1_staggering = bc.op isa ClimaCore.MatrixFields.FDOperatorWithFaceInput ? ClimaCore.Spaces.CellFace() : ClimaCore.Spaces.CellCenter()
        mat1_shape = get_mul_shape(mat1_staggering, typeof(mat1_row))
        mat2_shape = bc.op isa ClimaCore.MatrixFields.FDOperatorWithFaceInput ? FaceToFace() : CenterToCenter()
        v==1 && i ==1 && j==1 && h==1 && @cushow typeof(bc.op)
        matrix2 = shmem_mat2!(bc.args[1], mat2_shape, space, rzero(eltype(mat1_row)))
        # v==1 && i ==1 && j==1 && h==1 && @cushow eltype(matrix2)
        if mat1_staggering isa ClimaCore.Grids.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        return rzero(prod_eltype)
        # val = row_mul_vec!(prod_eltype, mat1_row, matrix2, mat1_shape)
        # return val
    else
        new_space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
        
        if new_space.staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(eltype(bc))
        end
        # correct_space = Operators.reconstruct_placeholder_broadcasted(space, bc)
        li = new_space.staggering isa ClimaCore.Spaces.CellCenter ? 1 : ClimaCore.Utilities.half
        
        idx = v - 1 + li
        # if hidx == (1, 1, 1) && v == 1
        #     @cushow typeof(bc.args[2])
        #     # @cushow typeof(bc)
        # end
        val = Operators.getidx(space, bc, idx, hidx)
        # return rzero(eltype(bc))
        return val
    end
end
function calc_row(mat1_staggering, bc::BC, space) where {BC <: Base.Broadcast.Broadcasted}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    # return zero(eltype(bc))
     if space isa ClimaCore.Spaces.AbstractSpace && space.staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return ClimaCore.RecursiveApply.rzero(eltype(bc))
    end
    resolved_args = UnrolledUtilities.unrolled_map(bc.args) do arg
        if !(arg isa Base.Broadcast.Broadcasted || arg isa ClimaCore.Operators.StencilBroadcasted)
            if arg isa ClimaCore.Fields.Field
                arg_data = ClimaCore.Fields.field_values(arg)
                @inbounds arg_data[CartesianIndex(i, j, 1, v, h)]
                # arg
             elseif arg isa ClimaCore.MatrixFields.BandMatrixRow || arg isa ClimaCore.Geometry.SingleValue
                arg
            else
                v_half = space.staggering isa ClimaCore.Spaces.CellCenter ? v : v - half
                ClimaCore.Operators.getidx(space, arg, v_half, (i, j, h))
            end
        else
            calc_row(mat1_staggering, arg, space)
        end
    end
    # space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), parent_space)
    # op = bc.op
    # return zero(eltype(bc))
    return bc.f(resolved_args...)
end




@inline function calc_row(mat1_staggering, bc::BC, space) where {M1, M2<:ClimaCore.Fields.Field, BC <: ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField, Tuple{M1,M2}}}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)

    mat1 = bc.args[1]
    mat2 = bc.args[2]
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    if eltype(mat2) <: ClimaCore.MatrixFields.BandMatrixRow
        # space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
        (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
        mat1_row =  calc_row(mat1_staggering, mat1, space)
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(mat1_row)))
        # if (i, j, v, h) == (1, 1, 1, 1)
        #     @cushow matrix2
        #     @cushow eltype(matrix2)
        #     # @cushow typeof(mat1_row)
        # end
        # return rzero(prod_eltype)
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        val = row_mul_mat!(prod_eltype, mat1_row, matrix2, mat1_shape, mat2_shape)
        return val
    else
        mat1_row =  calc_row(mat1_staggering, mat1, space)
        mat1_shape = get_mul_shape(mat1_staggering, eltype(mat1))
        mat2_shape = mat1_shape isa Union{FaceToFace, FaceToCenter} ? FaceToFace() : CenterToCenter()
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(mat1_row)))
        prod_eltype = eltype(bc)
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        val = row_mul_vec!(prod_eltype, mat1_row, matrix2, mat1_shape)
        return val
    end
end

@inline function calc_row(mat1_staggering, bc::BC, space) where {M1<:ClimaCore.Fields.Field, M2, BC <: ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField, Tuple{M1,M2}}}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
    mat1 = bc.args[1]
    mat2 = bc.args[2]
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    mat1_data = ClimaCore.Fields.field_values(mat1)
    if eltype(mat2) <: ClimaCore.MatrixFields.BandMatrixRow
        (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(eltype(mat1_data))))
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
        val = row_mul_mat!(prod_eltype, mat1_row, matrix2, mat1_shape, mat2_shape)
        return val
    else
        prod_eltype = eltype(bc)
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(eltype(mat1_data))))
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
        mat1_shape = get_mul_shape(mat1_staggering, eltype(mat1))
        mat2_shape = mat1_shape isa Union{FaceToFace, FaceToCenter} ? FaceToFace() : CenterToCenter()
         val = row_mul_vec!(prod_eltype, mat1_row, matrix2, mat1_shape)
        return val
    end
end

@inline function calc_row(mat1_staggering, bc::BC, space) where {M1, M2, BC <: ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField, Tuple{M1,M2}}}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), space)
    mat1 = bc.args[1]
    mat2 = bc.args[2]
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    if eltype(mat2) <: ClimaCore.MatrixFields.BandMatrixRow
        (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
        mat1_row = calc_row(mat1_staggering, mat1, space)
        # CUDA.sync_warp()
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(mat1_row)))
        # return RecursiveApply.rzero(eltype(bc))
        val = row_mul_mat!(prod_eltype, mat1_row, matrix2,  mat1_shape, mat2_shape)
        return val
    else
        prod_eltype = eltype(bc)
        if mat1_staggering isa ClimaCore.Spaces.CellCenter
            v == 64 && return rzero(prod_eltype)
        end
          mat1_row = calc_row(mat1_staggering, mat1, space)
        #   CUDA.sync_warp()
           mat1_shape = get_mul_shape(mat1_staggering, eltype(mat1))
        mat2_shape = mat1_shape isa Union{FaceToFace, FaceToCenter} ? FaceToFace() : CenterToCenter()
        matrix2 = shmem_mat2!(mat2, mat2_shape, space, rzero(eltype(mat1_row)))
         val = row_mul_vec!(prod_eltype, mat1_row, matrix2, mat1_shape)
        return val
    end
end

@inline function get_row_from_op(space, op, args)
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
    in_left_bnd =  ClimaCore.Operators.should_call_left_boundary(v_half, space, op, args...)
    in_right_bnd = ClimaCore.Operators.should_call_right_boundary(v_half, space, op, args...)
    if in_left_bnd
        # return rzero(row_type)
        lloc = ClimaCore.Operators.left_boundary_window(space)
        left_bndry = ClimaCore.Operators.get_boundary(op, lloc)
        return convert(row_type, ClimaCore.MatrixFields.op_matrix_first_row(op, left_bndry, space, v_half, hidx,))
    elseif in_right_bnd
        # return rzero(row_type)
        rroc = ClimaCore.Operators.right_boundary_window(space)
        right_bndry = ClimaCore.Operators.get_boundary(op, rroc)
        return convert(row_type, ClimaCore.MatrixFields.op_matrix_last_row(op, right_bndry, space, v_half, hidx,))
    else
        return convert(row_type, ClimaCore.MatrixFields.op_matrix_interior_row(op, space, v_half, hidx,))
    end
end

@inline function calc_row(mat1_staggering, bc::BC, stripped_space) where {BC <:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, <: ClimaCore.MatrixFields.FDOperatorMatrix{ <: ClimaCore.MatrixFields.OneArgFDOperator}}}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(bc), stripped_space)
    return get_row_from_op(space, bc.op.op, bc.args)
end


@inline function shmem_mat2!(mat2, ::S, space, mat1_row_entry) where {S}
    mat2_eltype = eltype(mat2)
    staggering =  S <: Union{FaceToFace, CenterToFace} ? ClimaCore.Spaces.CellFace() : ClimaCore.Spaces.CellCenter()
    mat2_row =  calc_row(staggering, mat2, space)
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(mat2), space)
    v = threadIdx().x
     i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # if v == 1 && j == 1 && h == 1 && blockIdx().x == 1
    #         @cushow mat2_eltype
    # end
    CUDA.sync_threads()
    if mat1_row_entry isa Union{ClimaCore.Geometry.AdjointAxisVector, ClimaCore.Geometry.Axis2TensorOrAdj} && (eltype(mat2_eltype) <: ClimaCore.Geometry.AxisTensor || mat2_eltype <: ClimaCore.Geometry.AxisTensor)
         targ = ClimaCore.Geometry.dual(axes(mat1_row_entry)[2])
         t_targ = typeof(ClimaCore.Geometry.dual(axes(mat1_row_entry)[2]))


         if !(mat2_row isa ClimaCore.MatrixFields.BandMatrixRow)
            if mat2_row isa ClimaCore.Geometry.AxisVector
                mat2_projected_type = ClimaCore.Geometry.axis_tensor_type(eltype(mat2_row), Tuple{typeof(targ)})
            else
                mat2_projected_type = ClimaCore.Geometry.axis_tensor_type(eltype(mat2_row), Tuple{ClimaCore.Geometry.axis2(mat2_row), typeof(targ)})
                mat2_projected_type = mat2_eltype <: ClimaCore.Geometry.AdjointAxisTensor ? Adjoint(mat2_projected_el) : mat2_projected_el
            end
        else
            (ld, ud) = ClimaCore.MatrixFields.outer_diagonals(mat2_eltype)
            if eltype(mat2_row) <: ClimaCore.Geometry.AxisVector
                mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_row)), Tuple{t_targ})
            else
                mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_row)), Tuple{t_targ, ClimaCore.Geometry.axis2(eltype(mat2_eltype)
                )})
                mat2_projected_el = eltype(mat2_eltype) <: ClimaCore.Geometry.AdjointAxisTensor ? Adjoint{mat2_projected_el} : mat2_projected_el
            end
            # mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_row)), Tuple{typeof(targ)})
            mat2_projected_type = ClimaCore.MatrixFields.band_matrix_row_type(ld, ud, mat2_projected_el)
        end
        # hidx == (1, 1, 1) && v == 1 && @cushow mat2_projected_el
        # return
        matrix2 = CUDA.CuDynamicSharedArray(mat2_projected_type, 64)
        if space.staggering isa ClimaCore.Spaces.CellFace || v != 64
            v_maybe_half = space.staggering isa ClimaCore.Spaces.CellFace  ? v - half : v
            if v_maybe_half isa ClimaCore.Utilities.PlusHalf
                @cuassert v_maybe_half <= 63 + half
            else
                @cuassert v_maybe_half <= 63
            end
            value2_lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
            if mat2_row isa ClimaCore.MatrixFields.BandMatrixRow
                projected_row = map(mat2_row) do entry
                    ClimaCore.Geometry.project(targ, entry, value2_lg)
                end
                # hidx == (1, 1, 1) && v == 1 && @cushow typeof(projected_row)
                # matrix2 = CUDA.CuDynamicSharedArray(typeof(projected_row), 64)
                matrix2[v] = projected_row
            else
                prod_val = ClimaCore.Geometry.project(targ, mat2_row, value2_lg)
                 matrix2[v] = prod_val
            end
        end
    else
         matrix2 = CUDA.CuDynamicSharedArray(mat2_eltype, 64)
        v = threadIdx().x
        # if S <: Union{FaceToFace, CenterToFace} || v != 64
            matrix2[v] = mat2_row
        # end
    end
    # h == 1 && j == 1 && (v==1 || v==64) && @cushow space.staggering
    # h == 1 && j == 1 && (v==1 || v==64) && @cushow S
    CUDA.sync_threads()
    return matrix2
end


@inline function shmem_mat2!(mat2::M, ::S, space, mat1_row_entry) where {M <: ClimaCore.Fields.Field, S}
    space = ClimaCore.Operators.reconstruct_placeholder_space(axes(mat2), space)
    mat2_data = ClimaCore.Fields.field_values(mat2)
    mat2_eltype = eltype(mat2)
    v = threadIdx().x
     i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
   
     if mat1_row_entry isa Union{ClimaCore.Geometry.AdjointAxisVector, ClimaCore.Geometry.Axis2TensorOrAdj} && (eltype(mat2_eltype) <: ClimaCore.Geometry.AxisTensor || mat2_eltype <: ClimaCore.Geometry.AxisTensor)
        # if v == 1 && j == 1 && h == 1 && blockIdx().x == 1
        #     @cushow mat2_eltype
        # end
        targ = ClimaCore.Geometry.dual(axes(mat1_row_entry)[2])
        t_targ = typeof(ClimaCore.Geometry.dual(axes(mat1_row_entry)[2]))
         if !(mat2_eltype <: ClimaCore.MatrixFields.BandMatrixRow)
            if mat2_eltype <: ClimaCore.Geometry.AxisVector
                mat2_projected_type = ClimaCore.Geometry.axis_tensor_type(eltype(mat2_eltype), Tuple{typeof(targ)})
            else
                mat2_projected_type = ClimaCore.Geometry.axis_tensor_type(eltype(mat2_eltype), Tuple{typeof(targ), ClimaCore.Geometry.axis2(mat2_eltype)})
            end
        else
            (ld, ud) = ClimaCore.MatrixFields.outer_diagonals(mat2_eltype)
            if eltype(mat2_eltype) <: ClimaCore.Geometry.AxisVector
                mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_eltype)), Tuple{t_targ})
            else
                mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_eltype)), Tuple{t_targ, ClimaCore.Geometry.axis2(eltype(mat2_eltype)
                )})
                #  mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_row)), Tuple{t_targ})
            end
            # mat2_projected_el = ClimaCore.Geometry.axis_tensor_type(eltype(eltype(mat2_row)), Tuple{typeof(targ)})
            mat2_projected_type = ClimaCore.MatrixFields.band_matrix_row_type(ld, ud, mat2_projected_el)
        end


        matrix2 = CUDA.CuDynamicSharedArray(mat2_projected_type, 64)
        if space.staggering isa ClimaCore.Spaces.CellFace  || v != 64
            mat2_row =  @inbounds mat2_data[CartesianIndex(i, j, 1, v , h)]
            v_maybe_half = space.staggering isa ClimaCore.Spaces.CellFace  ? v - half : v
            if v_maybe_half isa ClimaCore.Utilities.PlusHalf
                @cuassert v_maybe_half <= 63 + half
            else
                @cuassert v_maybe_half <= 63
            end
            value2_lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
            if mat2_row isa ClimaCore.MatrixFields.BandMatrixRow
                projected_row = map(mat2_row) do entry
                    ClimaCore.Geometry.project(targ, entry, value2_lg)
                end
                # matrix2 = CUDA.CuDynamicSharedArray(typeof(projected_row), 64)
                matrix2[v] = projected_row
             else
                prod_val = ClimaCore.Geometry.project(targ, mat2_row, value2_lg)
                #  matrix2 = CUDA.CuDynamicSharedArray(typeof(prod_val), 64)
                matrix2[v] = prod_val
            end
        end
    else
        mat2_eltype = eltype(mat2)
        matrix2 = CUDA.CuDynamicSharedArray(mat2_eltype, 64)
         if space.staggering isa ClimaCore.Spaces.CellFace  || v != 64
         matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v , h)]
        end
    end

    CUDA.sync_threads()
    return matrix2
end

@inline function get_mul_shapes(::ClimaCore.Spaces.CellCenter, ::Type{M1}, ::Type{M2}) where {M1, M2}
    if eltype(ClimaCore.MatrixFields.outer_diagonals(M1)) <: ClimaCore.Utilities.PlusHalf
        shape1 = FaceToCenter()
    else
        shape1 = CenterToCenter()
    end
    if eltype(ClimaCore.MatrixFields.outer_diagonals(M2)) <: ClimaCore.Utilities.PlusHalf
        if shape1 isa CenterToCenter
            shape2 = FaceToCenter()
        else
            shape2 = CenterToFace()
        end
    else
        shape2 = shape1 isa CenterToCenter ? CenterToCenter() : FaceToFace()
    end
    # shape2 = eltype(ClimaCore.MatrixFields.outer_diagonals(M2)) <: ClimaCore.Utilities.PlusHalf ? FaceToCenter() : (shape1 isa CenterToCenter ? CenterToCenter() : FaceToFace())
    ld1, ud1 = ClimaCore.MatrixFields.outer_diagonals(M1)
    ld2, ud2 = ClimaCore.MatrixFields.outer_diagonals(M2)
    prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
    prod_value_type = ClimaCore.Geometry.rmul_return_type(eltype(M1), eltype(M2))
    prod_eltype = ClimaCore.MatrixFields.band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    prod_shape = shape2 isa FaceToFace ? shape1 : CenterToCenter()
    return (prod_eltype, prod_shape, shape1, shape2)
end

@inline function get_mul_shapes(::ClimaCore.Spaces.CellFace, ::Type{M1}, ::Type{M2}) where {M1, M2}
    if eltype(ClimaCore.MatrixFields.outer_diagonals(M1)) <: ClimaCore.Utilities.PlusHalf
        shape1 = CenterToFace()
    else
        shape1 = FaceToFace() #f2f
    end
    if eltype(ClimaCore.MatrixFields.outer_diagonals(M2)) <: ClimaCore.Utilities.PlusHalf
        if shape1 isa FaceToFace
            shape2 = CenterToFace()
        else
            shape2 = FaceToCenter()
        end
    else
        shape2 = shape1 isa FaceToFace ? FaceToFace() : CenterToCenter()
    end
    # shape2 = eltype(ClimaCore.MatrixFields.outer_diagonals(M2)) <: ClimaCore.Utilities.PlusHalf ? CenterToFace() : (shape1 isa FaceToFace ? FaceToFace() : CenterToCenter())
    ld1, ud1 = ClimaCore.MatrixFields.outer_diagonals(M1)
    ld2, ud2 = ClimaCore.MatrixFields.outer_diagonals(M2)
    prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
    prod_value_type = ClimaCore.Geometry.rmul_return_type(eltype(M1), eltype(M2))
    prod_eltype = ClimaCore.MatrixFields.band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    prod_shape = shape2 isa CenterToCenter ? shape1 : FaceToFace()
    return (prod_eltype, prod_shape, shape1, shape2,)
end


@inline function get_mul_shape(::ClimaCore.Spaces.CellFace, ::Type{M1},) where {M1}
     if eltype(ClimaCore.MatrixFields.outer_diagonals(M1)) <: ClimaCore.Utilities.PlusHalf
        return CenterToFace()
    else
        return FaceToFace() #f2f
    end
end

@inline function get_mul_shape(::ClimaCore.Spaces.CellCenter, ::Type{M1},) where {M1}
    if eltype(ClimaCore.MatrixFields.outer_diagonals(M1)) <: ClimaCore.Utilities.PlusHalf
        return FaceToCenter()
    else
        return CenterToCenter()
    end
    
end
