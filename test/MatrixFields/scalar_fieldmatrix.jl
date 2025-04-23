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
    center_space, face_space = test_spaces(FT)
    surface_space = Spaces.level(face_space, half)
    seed!(1)
    λ = 10
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)

    e¹² = Geometry.Covariant12Vector(1, 2)
    e³ = Geometry.Covariant3Vector(1)
    e₃ = Geometry.Contravariant3Vector(1)

    ρχ_unit = (; ρq_tot = 1, ρq_liq = 1, ρq_ice = 1, ρq_rai = 1, ρq_sno = 1)
    ρaχ_unit =
        (; ρaq_tot = 1, ρaq_liq = 1, ρaq_ice = 1, ρaq_rai = 1, ρaq_sno = 1)


    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit ⊠ e₃')), ᶜᶠmat2)
    A = dycore_prognostic_EDMF_FieldMatrix(;
        ᶜᶜmat1,
        ᶜᶠmat2,
        ᶠᶜmat2,
        ᶜᶜmat3,
        ᶠᶠmat3,
        e¹²,
        e³,
        e₃,
        ρχ_unit,
        ρaχ_unit,
        ᶜᶠmat2_ρχ_u₃,
        ᶠᶠmat3_u₃_u₃,
        ᶜᶠmat2_scalar_u₃,
        ᶠᶜmat2_u₃_scalar,
    )

    scalar_keys = MatrixFields.get_scalar_keys(A)
    @test length(scalar_keys) > length(keys(A))
    @test all(
        f -> f isa MatrixFields.UniformScaling || eltype(eltype(f)) <: FT,
        getindex.(Ref(A), scalar_keys),
    )
    @test (@allocated MatrixFields.get_scalar_keys(A)) == 0
    @test_opt MatrixFields.get_scalar_keys(A)
    foreach(scalar_keys) do key
        @test_opt A[key]
        @test (@allocated A[key]) == 0
    end

    A_scalar = MatrixFields.scalar_fieldmatrix(A)
    @test all(
        f -> f isa MatrixFields.UniformScaling || eltype(eltype(f)) <: FT,
        getindex.(Ref(A_scalar), scalar_keys),
    )
    @test all(scalar_keys) do key
        entry = A_scalar[key]
        entry isa MatrixFields.UniformScaling && return true
        key′, entry′ =
            filter(pair -> MatrixFields.is_child_value(key, pair[1]), pairs(A))[1]
        if key != key′
            parent(entry) isa SubArray ||
                parent(entry) === parent(entry′) ||
                return false
        end
        parent(parent(entry)) === parent(entry′) || return false
        return true
    end
    function scalar_fieldmatrix_wrapper(field_matrix_of_tensors)
        MatrixFields.scalar_fieldmatrix(field_matrix_of_tensors)
        return
    end
    scalar_fieldmatrix_wrapper(A)
    @test (@allocated scalar_fieldmatrix_wrapper(A)) == 0
    @test_opt MatrixFields.scalar_fieldmatrix(A)

    A_copy = deepcopy(A)
    @test all(scalar_keys) do key
        entry_copy = A_copy[key]
        entry_original = A[key]
        entry_flattened = A_scalar[key]
        entry_flattened isa MatrixFields.UniformScaling && return true
        entry_original == entry_copy || return false
        entry_original === entry_flattened || return false
        entry_flattened .*= 2
        entry_original == 2 .* entry_copy || return false
        return true
    end
end
