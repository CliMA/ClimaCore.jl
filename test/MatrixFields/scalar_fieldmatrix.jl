using Test
using JET

import ClimaCore:
    Geometry, Domains, Meshes, Spaces, Fields, MatrixFields, CommonSpaces
import ClimaCore.Utilities: half
import ClimaComms
import ClimaCore.MatrixFields: @name
ClimaComms.@import_required_backends
include("matrix_field_test_utils.jl")

@testset "field_offset_and_type" begin
    FT = Float64
    struct Singleton{T}
        x::T
    end
    struct TwoFields{T1, T2}
        x::T1
        y::T2
    end
    function test_field_offset_and_type(
        name,
        ::Type{T},
        ::Type{S},
        expected_offset,
        ::Type{E},
        key_error,
    ) where {T, S, E}
        @test_all MatrixFields.field_offset_and_type(name, T, S, key_error) ==
                  (expected_offset, E)
    end
    test_field_offset_and_type(
        @name(x),
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
        Singleton{Singleton{Singleton{FT}}},
        KeyError(@name(x.x.x.x)),
    )
    test_field_offset_and_type(
        @name(x.x.x.x),
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
        FT,
        KeyError(@name(x.x.x.x)),
    )
    test_field_offset_and_type(
        @name(y.x),
        FT,
        TwoFields{TwoFields{FT, FT}, TwoFields{FT, FT}},
        2,
        FT,
        KeyError(@name(y.x)),
    )
    test_field_offset_and_type(
        @name(y.y),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
        TwoFields{FT, Singleton{FT}},
        KeyError(@name(y.y.x)),
    )
    test_field_offset_and_type(
        @name(y.y),
        Float32,
        TwoFields{TwoFields{FT, FT}, TwoFields{FT, FT}},
        6,
        FT,
        KeyError(@name(y.y.x)),
    )
    test_field_offset_and_type(
        (@name(y.y), @name(x)),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
        FT,
        KeyError(@name(y.y.x.x)),
    )
    test_field_offset_and_type(
        (@name(y.y), @name(y.x)),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        4,
        FT,
        KeyError(@name(y.y.y.x)),
    )
end

@testset "fieldmatrix to scalar fieldmatrix unit tests" begin
    FT = Float64
    for (A, _) in (
        dycore_prognostic_EDMF_FieldMatrix(FT),
        scaling_only_dycore_prognostic_EDMF_FieldMatrix(FT),
    )
        @test all(
            entry ->
                entry isa MatrixFields.UniformScaling ||
                    eltype(eltype(entry)) <: FT,
            MatrixFields.scalar_fieldmatrix(A, FT).entries,
        )
        test_get(A, entry, key) = A[key] === entry
        for (key, entry) in MatrixFields.scalar_fieldmatrix(A, FT)
            @test test_get(A, entry, key)
            @test (@allocated test_get(A, entry, key)) == 0
            @test_opt test_get(A, entry, key)
        end

        function scalar_fieldmatrix_wrapper(
            field_matrix_of_tensors,
            ::Type{T},
        ) where {T}
            A_scalar =
                MatrixFields.scalar_fieldmatrix(field_matrix_of_tensors, T)
            return nothing
        end

        scalar_fieldmatrix_wrapper(A, FT)
        @test (@allocated scalar_fieldmatrix_wrapper(A, FT)) == 0
        @test_opt MatrixFields.scalar_fieldmatrix(A, FT)
    end
end
