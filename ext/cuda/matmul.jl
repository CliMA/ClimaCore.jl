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


Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2,  ::FaceToCenter, ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(63)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d + half <= Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToFace, ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d - half < Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToCenter, ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(63)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d  <= Int32(63))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToFace, ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd < li || v + pd > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d  <= Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToCenter, ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d + half  <= Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d + half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::CenterToFace, ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd + half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d - half  < Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d - half][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2, ::FaceToFace, ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d  <= Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_mat!(::Type{P}, mat1_row, matrix2,  ::CenterToCenter, ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
        pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(eltype(prod_eltype))
        prod_entries = UnrolledUtilities.unrolled_map(pd1:pd2) do pd
            if v + pd + half < li || v + pd +half > ri
                zero_entry
            else
                UnrolledUtilities.unrolled_sum( ld1:ud1) do mat1_row_d
                    if ld2 <= pd - mat1_row_d <= ud2 && (Int32(0) <  v + mat1_row_d  < Int32(64))
                        @inbounds mat1_row[mat1_row_d] * matrix2[v + mat1_row_d][pd - mat1_row_d]
                    else
                        zero_entry
                    end
                end
            end
        end
        return ClimaCore.MatrixFields.BandMatrixRow{pd1}(prod_entries...)
    end
end

Base.@propagate_inbounds function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::FaceToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = Int32(1)
        ri = Int32(63)

        zero_entry = rzero(prod_eltype)
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (Int32(0) <  v + mat1_row_d + half <= Int32(64))
                @inbounds bb = mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d + half]
            else
                zero_entry
            end
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::CenterToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(prod_eltype)
        val = zero_entry
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (Int32(0) <  v + mat1_row_d - half < Int32(64))
                    @inbounds mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d - half]
            else
                zero_entry
            end
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::CenterToCenter) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = Int32(1)
        ri = Int32(63)

        zero_entry = rzero(prod_eltype)
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (Int32(0) <  v + mat1_row_d  <= Int32(63))
                @inbounds mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d]
            else
                zero_entry
            end
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(::Type{P}, mat1_row, matrix2,  ::FaceToFace) where {P}
    @inbounds begin
        prod_eltype = P
        v = threadIdx().x
        mat1_eltype = typeof(mat1_row)
        mat2_eltype = eltype(matrix2)
        ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
        li = Int32(1)
        ri = Int32(64)

        zero_entry = rzero(prod_eltype)
        return UnrolledUtilities.unrolled_mapreduce(⊞, ld1:ud1; init=zero_entry) do mat1_row_d
            if (Int32(0) <  v + mat1_row_d  <= Int32(64))
                @inbounds mat1_row[mat1_row_d] ⊗ matrix2[v + mat1_row_d ]
            else
                zero_entry
            end
        end
    end
end
