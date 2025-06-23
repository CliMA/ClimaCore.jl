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
        ::Type{E};
        apply_zero = false,
    ) where {T, S, E}
        @test_all MatrixFields.field_offset_and_type(name, T, S, name) ==
                  (expected_offset, E, apply_zero)
    end
    test_field_offset_and_type(
        (@name(x), @name()),
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
        Singleton{Singleton{Singleton{FT}}},
    )
    test_field_offset_and_type(
        (@name(), @name(x.x.x.x)),
        FT,
        Singleton{Singleton{Singleton{Singleton{FT}}}},
        0,
        FT,
    )
    test_field_offset_and_type(
        (@name(), @name(y.x)),
        FT,
        TwoFields{TwoFields{FT, FT}, TwoFields{FT, FT}},
        2,
        FT,
    )
    test_field_offset_and_type(
        (@name(y), @name(y)),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
        TwoFields{FT, Singleton{FT}},
    )
    test_field_offset_and_type(
        (@name(y.k), @name(y.k)),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
        TwoFields{FT, Singleton{FT}},
    )
    test_field_offset_and_type(
        (@name(y.k.g), @name(y.k.l)),
        FT,
        TwoFields{
            TwoFields{FT, FT},
            TwoFields{FT, TwoFields{FT, Singleton{FT}}},
        },
        3,
        TwoFields{FT, Singleton{FT}},
        apply_zero = true,
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
    )
end

@testset "fieldmatrix to scalar fieldmatrix unit tests" begin
    FT = Float64
    for (A, b) in (
        dycore_prognostic_EDMF_FieldMatrix(FT),
        scaling_only_dycore_prognostic_EDMF_FieldMatrix(FT),
    )
        @test all(
            entry ->
                entry isa MatrixFields.UniformScaling ||
                    eltype(eltype(entry)) <: FT,
            MatrixFields.scalar_field_matrix(A).entries,
        )
        test_get(A, entry, key) = A[key] === entry
        for (key, entry) in MatrixFields.scalar_field_matrix(A)
            @test test_get(A, entry, key)
            @test (@allocated test_get(A, entry, key)) == 0
            @test_opt test_get(A, entry, key)
        end

        function scalar_field_matrix_wrapper(field_matrix_of_tensors)
            A_scalar = MatrixFields.scalar_field_matrix(field_matrix_of_tensors)
            return nothing
        end

        scalar_field_matrix_wrapper(A)
        @test (@allocated scalar_field_matrix_wrapper(A)) == 0
        @test_opt MatrixFields.scalar_field_matrix(A)

        A_with_tree =
            MatrixFields.replace_name_tree(A, MatrixFields.FieldNameTree(b))
        @test MatrixFields.scalar_field_matrix(A_with_tree).keys.name_tree ==
              A_with_tree.keys.name_tree
    end
end

@testset "implicit tensor structure optimization indexing" begin
    FT = Float64
    center_space = test_spaces(FT)[1]
    for (maybe_copy, maybe_to_field) in
        ((identity, identity), (copy, x -> fill(x, center_space)))
        A = MatrixFields.FieldMatrix(
            (@name(c.uₕ), @name(c.uₕ)) =>
                maybe_to_field(DiagonalMatrixRow(FT(2))),
            (@name(foo), @name(bar)) => maybe_to_field(
                DiagonalMatrixRow(
                    Geometry.Covariant12Vector(FT(1), FT(2)) *
                    Geometry.Contravariant12Vector(FT(1), FT(2))',
                ),
            ),
        )
        @test A[(
            @name(c.uₕ.components.data.:1),
            @name(c.uₕ.components.data.:1)
        )] == A[(@name(c.uₕ), @name(c.uₕ))]
        @test maybe_copy(
            A[(@name(c.uₕ.components.data.:2), @name(c.uₕ.components.data.:1))],
        ) == maybe_to_field(DiagonalMatrixRow(FT(0)))
        @test maybe_copy(A[(@name(foo.dog), @name(bar.dog))]) ==
              A[(@name(foo), @name(bar))]
        @test maybe_copy(A[(@name(foo.cat), @name(bar.dog))]) ==
              zero(A[(@name(foo), @name(bar))])
        @test A[(
            @name(foo.dog.components.data.:1),
            @name(bar.dog.components.data.:2)
        )] == maybe_to_field(DiagonalMatrixRow(FT(2)))
        @test maybe_copy(
            A[(
                @name(foo.dog.components.data.:1),
                @name(bar.cat.components.data.:2)
            )],
        ) == maybe_to_field(DiagonalMatrixRow(FT(0)))
    end
end
