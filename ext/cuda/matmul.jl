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
import ClimaCore.MatrixFields
using ClimaCore.MatrixFields
import UnrolledUtilities

function cf_mul_fc!(out, mat1::M, mat2, space, bds) where {M}
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        # this is incorrect when l1 != 1 or ri != 63
        # prod_idx = v - 1 + li
        # TODO:
        # matrix1 = CUDA.CuStaticSharedArray(mat1_eltype, 64) # padded with one extra level
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)

        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        CUDA.sync_threads()
        v == 64 && return nothing
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v, hidx, val)
    end
    return nothing
end
function fc_mul_cf!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        if v != 64
            matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        end
        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v_half, hidx, val)
    end
    return nothing
end
function cc_mul_cc!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        # this is incorrect when l1 != 1 or ri != 63
        # prod_idx = v - 1 + li
        # TODO:
        # matrix1 = CUDA.CuStaticSharedArray(mat1_eltype, 64)
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64) # padded with one extra level
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        v == 64 && return nothing
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v, hidx, val)
    end
    return nothing
end
function ff_mul_ff!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v_half, hidx, val)
    end
    return nothing
end
function cf_mul_ff!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        CUDA.sync_threads()
        v == 64 && return nothing
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v, hidx, val)
    end
    return nothing
end
function fc_mul_cc!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        if v != 64
            matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        end
        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
        v_half =  v - 1 + li
        # v ==1 && @cushow li
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
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
        setidx!(space, out,  v_half, hidx, val)
    end
    return nothing
end
function ff_mul_fc!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]

        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v_half, hidx, val)
    end
    return nothing
end
function cc_mul_cf!(out, mat1, mat2, space, bds)
    @inbounds begin
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        prod_eltype = eltype(out)
        mat1_eltype = eltype(mat1)
        mat2_eltype = eltype(mat2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
        (li, lw, rw, ri) = bds
        matrix2 = CUDA.CuStaticSharedArray(mat2_eltype, 64)
        mat1_data = Fields.field_values(mat1)
        mat2_data = Fields.field_values(mat2)
        v==64 && return nothing
        # if v != 64
        matrix2[v] = @inbounds mat2_data[CartesianIndex(i, j, 1, v, h)]
        # end
        CUDA.sync_threads()
        mat1_row = @inbounds mat1_data[CartesianIndex(i, j, 1, v, h)]
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
        setidx!(space, out,  v_half, hidx, val)
    end
    return nothing
end
