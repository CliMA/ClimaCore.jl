import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square, CenterToCenter, FaceToFace
import ClimaCore.MatrixFields
import ClimaCore
using ClimaCore.MatrixFields
import UnrolledUtilities


@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::FaceToCenter, ::CenterToFace) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::CenterToFace, ::FaceToCenter) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            zero_entry
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::CenterToCenter, ::CenterToCenter) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::FaceToFace, ::FaceToFace) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::FaceToCenter, ::FaceToFace) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::CenterToFace, ::CenterToCenter) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::FaceToFace, ::CenterToFace) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function row_mul_mat!(::Type{P}, mat1_row, matrix2, space, ::CenterToCenter, ::FaceToCenter) where {P}
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

        zero_entry = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
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

@inline function cf_mul_fc!(::Type{P}, mat1_row::M, matrix2, space,) where {P, M}
    @inbounds begin
        prod_eltype = P
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        # prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        # (li, lw, rw, ri) = bds
        li = 1
        ri = 63
        # this is incorrect when l1 != 1 or ri != 63
        # prod_idx = v - 1 + li
        # TODO:
        zero_val = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero(eltype(prod_eltype))
            else
                s = UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d + half <= 64)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    else
                        zero_val
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
        
    end
    # return nothing
end
@inline function fc_mul_cf!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        v_half =  v - 1 + li
        # v ==1 && @cushow li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v_half + pd < li || v_half + pd > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                # v == 64 && @cushow pd
                for mat1_row_d in (ld1:ud1)
                    if ld2 <=pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d - half < 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end
@inline function cc_mul_cc!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        # this is incorrect when l1 != 1 or ri != 63
        # prod_idx = v - 1 + li
        # TODO:
        # matrix1 = CUDA.CuStaticSharedArray(mat1_eltype, 64)
        zero_val = zero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero(eltype(prod_eltype))
            else
                s = UnrolledUtilities.unrolled_sum(ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d <= 63)
                        mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_val
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end
@inline function ff_mul_ff!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        v_half =  v - 1 + li
        # v ==1 && @cushow li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v_half + pd < li || v_half + pd > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                for mat1_row_d in (ld1:ud1)
                    if ld2 <=pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  <= 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end
@inline function cf_mul_ff!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
       
        v_half =  v - 1 + li
        # v == 63 && @cushow ri
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd - half > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                for mat1_row_d in (ld1:ud1)
                    if ld2 <= pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d + half  <= 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
        
    end
    return nothing
end
@inline function fc_mul_cc!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        v_half =  v - 1 + li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                for mat1_row_d in (ld1:ud1)
                    if ld2 <=pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d - half < 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end
@inline function ff_mul_fc!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds

        
        v_half =  v - 1 + li
        # v ==1 && @cushow li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                # v == 64 && @cushow pd
                for mat1_row_d in (ld1:ud1)
                    if ld2 <=pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  <= 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end
@inline function cc_mul_cf!(out, mat1_row, matrix2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds

        v_half =  v - 1 + li
        # v ==1 && @cushow li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v_half + pd + half < li || v_half + pd - half > ri
                zero(eltype(prod_eltype))
            else
                s = zero(eltype(prod_eltype))
                # v == 64 && @cushow pd
                for mat1_row_d in (ld1:ud1)
                    if ld2 <=pd - mat1_row_d <= ud2 && (0 <  v + mat1_row_d  < 64)
                        s += mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    end
                end
                s
            end
        end
        val = ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
        return val
    end
    return nothing
end



function entry_matmul!(out, mat1, mat2, space,)# bc)
    shared_backing_array = CUDA.CuStaticSharedArray(Float32, 64*16)
    (Ni, Nj, _, Nv, Nh) = ClimaCore.DataLayouts.universal_size(ClimaCore.Fields.field_values(out))


    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    mat1_staggering =  Nv == 64 ?  ClimaCore.Spaces.CellFace() : ClimaCore.Spaces.CellCenter()
    val = handle_matmul!(shared_backing_array, mat1_staggering, mat1, mat2, space)
    # if i==1 &&  h==1 && v==1
    #     @cushow val.entries[2]
    # end
    if Nv == 64
        setidx!(space, out,  v - half, hidx, val)
    else
        v != 64 && setidx!(space, out,  v , hidx, val)
    end
    return nothing
end

@inline function handle_matmul!(shared_backing_array, mat1_staggering, mat1::M1, mat2::M2, space) where {M1<:ClimaCore.Fields.Field, M2<:ClimaCore.Fields.Field}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
    matrix2 = shmem_mat2!(shared_backing_array, mat2, mat2_shape, space)
    if mat1_staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return zero(prod_eltype)
    end
    mat1_data = ClimaCore.Fields.field_values(mat1)
    mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
    val = row_mul_mat!(prod_eltype, mat1_row, matrix2, space, mat1_shape, mat2_shape)
    return val
end

@inline function handle_matmul!(shared_backing_array, mat1_staggering, mat1::M1, mat2::M2, space) where {M1<:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}, M2<:ClimaCore.Fields.Field}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
    mat1_row =  handle_matmul!(shared_backing_array, mat1_staggering, mat1.args[1], mat1.args[2], space)
    matrix2 = shmem_mat2!(shared_backing_array, mat2, mat2_shape, space)
    if mat1_staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return zero(prod_eltype)
    end
    val = row_mul_mat!(prod_eltype, mat1_row, matrix2, space, mat1_shape, mat2_shape)
    return val
end

@inline function handle_matmul!(shared_backing_array, mat1_staggering, mat1::M1, mat2::M2, space) where {M1<:ClimaCore.Fields.Field, M2<:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
    if mat1_staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return zero(prod_eltype)
    end
    mat1_data = ClimaCore.Fields.field_values(mat1)
    mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
    matrix2 = shmem_mat2!(shared_backing_array, mat2, mat2_shape, space)
    val = row_mul_mat!(prod_eltype, mat1_row, matrix2, space, mat1_shape, mat2_shape)
    return val
end

@inline function handle_matmul!(shared_backing_array, mat1_staggering, mat1::M1, mat2::M2, space) where {M1<:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}, M2<:ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}}
    i = blockIdx().x
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    (prod_eltype, prod_shape, mat1_shape, mat2_shape) = get_mul_shapes(mat1_staggering, eltype(mat1), eltype(mat2))
    if mat1_staggering isa ClimaCore.Spaces.CellCenter
        v == 64 && return zero(prod_eltype)
    end
    mat1_row = handle_matmul!(shared_backing_array, mat1_staggering, mat1.args[1], mat1.args[2], space)
    matrix2 = shmem_mat2!(shared_backing_array, mat2, mat2_shape, space)
    val = row_mul_mat!(prod_eltype, mat1_row, matrix2, space, mat1_shape, mat2_shape)
    return val
end




@inline function shmem_mat2!(shared_backing_array, mat2, ::S, space) where {S <: Union{FaceToFace, CenterToFace}}
    mat2_eltype = eltype(mat2)
    mat2_row =  handle_matmul!(shared_backing_array, ClimaCore.Spaces.CellFace(), mat2.args[1], mat2.args[2], space)
    mat2_eltype_size_b = sizeof(mat2_eltype)
    matrix2 = reinterpret(mat2_eltype, @view shared_backing_array[1:64 * (mat2_eltype_size_b ÷ sizeof(Float32))])
    i = blockIdx().x
    j = blockIdx().y
    h = blockIdx().z
    v = threadIdx().x
    matrix2[v] = mat2_row
    CUDA.sync_threads()
    return matrix2
end

@inline function shmem_mat2!(shared_backing_array, mat2, ::S, space) where {S <: Union{CenterToCenter, FaceToCenter}}
    mat2_eltype = eltype(mat2)
    mat2_row =  handle_matmul!(shared_backing_array, ClimaCore.Spaces.CellCenter(), mat2.args[1], mat2.args[2], space)
    mat2_eltype_size_b = sizeof(mat2_eltype)
    matrix2 = reinterpret(mat2_eltype, @view shared_backing_array[1:64 * (mat2_eltype_size_b ÷ sizeof(Float32))])
    i = blockIdx().x
    j = blockIdx().y
    h = blockIdx().z
    v = threadIdx().x
    if v != 64
        matrix2[v] = mat2_row
    end
    CUDA.sync_threads()
    return matrix2
end




@inline function shmem_mat2!(shared_backing_array, mat2::M, ::S, space) where {M <: ClimaCore.Fields.Field, S <: Union{FaceToFace, CenterToFace}}
    mat2_eltype = eltype(mat2)
    mat2_data = ClimaCore.Fields.field_values(mat2)
    mat2_eltype_size_b = sizeof(mat2_eltype)
    matrix2 = reinterpret(mat2_eltype, @view shared_backing_array[1:64 * (mat2_eltype_size_b ÷ sizeof(Float32))])
    i = blockIdx().x
    j = blockIdx().y
    h = blockIdx().z
    v = threadIdx().x
    matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v , h)]
    CUDA.sync_threads()
    return matrix2
end

@inline function shmem_mat2!(shared_backing_array, mat2::M, ::S, space) where {M <: ClimaCore.Fields.Field, S <: Union{CenterToCenter, FaceToCenter}}
    mat2_eltype = eltype(mat2)
    mat2_data = ClimaCore.Fields.field_values(mat2)
    mat2_eltype_size_b = sizeof(mat2_eltype)
    matrix2 = reinterpret(mat2_eltype, @view shared_backing_array[1:64 * (mat2_eltype_size_b ÷ sizeof(Float32))])
    i = blockIdx().x
    j = blockIdx().y
    h = blockIdx().z
    v = threadIdx().x
    if v != 64
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v , h)]
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
