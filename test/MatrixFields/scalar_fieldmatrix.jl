using Test
using JET

import ClimaCore:
    Geometry, Domains, Meshes, Spaces, Fields, MatrixFields, CommonSpaces
import ClimaCore.Utilities: half
import ClimaComms
import ClimaCore.MatrixFields: @name
ClimaComms.@import_required_backends
include("matrix_field_test_utils.jl")

@testset "get_field_first_index_offset" begin
    FT = Float64
    struct Singleton{T}
        x::T
    end
    struct TwoFields{T1, T2}
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
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(x.x.x.x),
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(y.x),
        FT,
        TwoFields{TwoFields{FT, FT}, TwoFields{FT, FT}},
        2,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        Float32,
        TwoFields{TwoFields{FT, FT}, TwoFields{FT, FT}},
        6,
    )
    test_get_field_first_index_offset(
        @name(y.y.x),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y.y.x),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        4,
    )
end

@testset "broadcasted_get_field_type" begin
    FT = Float64
    struct Singleton{T}
        x::T
    end
    struct TwoFields{T1, T2}
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
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        Singleton{Singleton{Singleton{FT}}},
    )
    test_broadcasted_get_field_type(
        @name(x.x.x),
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        Singleton{FT},
    )
    test_broadcasted_get_field_type(
        @name(y.x),
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        FT,
    )
    test_broadcasted_get_field_type(
        @name(y.y.y),
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        Singleton{FT},
    )
end

@testset "fieldmatrix to scalar fieldmatrix unit tests" begin
    FT = Float64
    A, b = dycore_prognostic_EDMF_FieldMatrix(FT)
    for (A, b) in (
        dycore_prognostic_EDMF_FieldMatrix(FT),
        scaling_only_dycore_prognostic_EDMF_FieldMatrix(FT),
    )
        @test all(
            entry ->
                entry isa MatrixFields.UniformScaling ||
                    eltype(eltype(entry)) <: FT,
            MatrixFields.scalar_fieldmatrix(A, b).entries,
        )
        test_get(A, entry, key) = A[key] === entry
        for (key, entry) in MatrixFields.scalar_fieldmatrix(A, b)
            @test test_get(A, entry, key)
            @test (@allocated test_get(A, entry, key)) == 0
            @test_opt test_get(A, entry, key)
        end

        function scalar_fieldmatrix_wrapper(field_matrix_of_tensors, b)
            A_scalar =
                MatrixFields.scalar_fieldmatrix(field_matrix_of_tensors, b)
            return nothing
        end
        scalar_fieldmatrix_wrapper(A, b)
        @test (@allocated scalar_fieldmatrix_wrapper(A, b)) == 0
        @test_opt MatrixFields.scalar_fieldmatrix(A, b)
    end
end
