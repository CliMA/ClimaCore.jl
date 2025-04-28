using Test
using JET

import ClimaCore: Geometry, Domains, Meshes, Spaces, Fields, MatrixFields
import ClimaCore.Utilities: half
import ClimaComms
import ClimaCore.MatrixFields: @name
ClimaComms.@import_required_backends
include("matrix_field_test_utils.jl")

@testset "get_field_first_index_offset" begin
    struct Singleton{T}
        x::T
    end
    struct Two_fields{T1, T2}
        x::T1
        y::T2
    end
    function test_get_field_first_index_offset(
        name,
        ::Type{T},
        ::Type{S},
        expected_offset,
    ) where {T, S}
        @test_all MatrixFields.get_field_first_index_offset(name, T, S) ==
                  expected_offset
    end
    test_get_field_first_index_offset(
        @name(x),
        Float64,
        Singleton{Singleton{Singleton{Singleton{Float64}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(x.x.x.x),
        Float64,
        Singleton{Singleton{Singleton{Singleton{Float64}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(y.x),
        Float64,
        Two_fields{Two_fields{Float64, Float64}, Two_fields{Float64, Float64}},
        2,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        Float64,
        Two_fields{
            Two_fields{Float64, Float64},
            Two_fields{Float64, Two_fields{Float64, Singleton{Float64}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        Float32,
        Two_fields{Two_fields{Float64, Float64}, Two_fields{Float64, Float64}},
        6,
    )
    test_get_field_first_index_offset(
        @name(y.y.x),
        Float64,
        Two_fields{
            Two_fields{Float64, Float64},
            Two_fields{Float64, Two_fields{Float64, Singleton{Float64}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y.y.x),
        Float64,
        Two_fields{
            Two_fields{Float64, Float64},
            Two_fields{Float64, Two_fields{Float64, Singleton{Float64}}},
        },
        4,
    )
end

@testset "broadcasted_get_field_type" begin
    struct Singleton{T}
        x::T
    end
    struct Two_fields{T1, T2}
        x::T1
        y::T2
    end
    function test_broadcasted_get_field_type(
        name,
        ::Type{T},
        expected_type,
    ) where {T}
        @test_all MatrixFields.broadcasted_get_field_type(T, name) ==
                  expected_type
    end
    test_broadcasted_get_field_type(
        @name(x),
        Singleton{Singleton{Singleton{Singleton{Float64}}}},
        Singleton{Singleton{Singleton{Float64}}},
    )
    test_broadcasted_get_field_type(
        @name(x.x.x),
        Singleton{Singleton{Singleton{Singleton{Float64}}}},
        Singleton{Float64},
    )
    test_broadcasted_get_field_type(
        @name(y.x),
        Two_fields{
            Two_fields{Float64, Float64},
            Two_fields{Float64, Two_fields{Float64, Singleton{Float64}}},
        },
        Float64,
    )
    test_broadcasted_get_field_type(
        @name(y.y.y),
        Two_fields{
            Two_fields{Float64, Float64},
            Two_fields{Float64, Two_fields{Float64, Singleton{Float64}}},
        },
        Singleton{Float64},
    )
end

@testset "fieldmatrix to scalar fieldmatrix unit tests" begin
    FT = Float64
    A, _ = dycore_prognostic_EDMF_FieldMatrix(FT)
    for key in MatrixFields.get_scalar_keys(A)
        @test_all A[key] isa MatrixFields.ColumnwiseBandMatrixField ?
                  eltype(eltype(A[key])) == eltype(parent(A[key])) :
                  eltype(eltype(A[key])) == eltype(A[key])
    end
    @test all(
        entry ->
            entry isa MatrixFields.UniformScaling ||
                eltype(eltype(entry)) <: FT,
        MatrixFields.scalar_fieldmatrix(A).entries,
    )
    test_get(A, entry, key) = A[key] == entry
    for (key, entry) in MatrixFields.scalar_fieldmatrix(A)
        @test test_get(A, entry, key)
        @test (@allocated test_get(A, entry, key)) == 0
        @test_opt test_get(A, entry, key)
    end
    function scalar_fieldmatrix_wrapper(field_matrix_of_tensors)
        A_scalar = MatrixFields.scalar_fieldmatrix(field_matrix_of_tensors)
        return true
    end
    scalar_fieldmatrix_wrapper(A) # compile the wrapper function
    @test (@allocated scalar_fieldmatrix_wrapper(A)) == 0
    @test_opt MatrixFields.scalar_fieldmatrix(A)
end
