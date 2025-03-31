using Test
using JET

import ClimaCore: Geometry, Domains, Meshes, Spaces, Fields, MatrixFields
import ClimaCore.Utilities: half
import ClimaCore.RecursiveApply: ⊠
import ClimaComms
import LinearAlgebra: I, norm, ldiv!, mul!
import ClimaCore.MatrixFields: @name
ClimaComms.@import_required_backends
include("matrix_field_test_utils.jl")

@testset "get_field_first_index_offset" begin
    struct singleton{T}
        x::T
    end
    struct two_fields{T1, T2}
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
        singleton{singleton{singleton{singleton{Float64}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(x.x.x.x),
        Float64,
        singleton{singleton{singleton{singleton{Float64}}}},
        0,
    )
    test_get_field_first_index_offset(
        @name(y.x),
        Float64,
        two_fields{two_fields{Float64, Float64}, two_fields{Float64, Float64}},
        2,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        Float64,
        two_fields{
            two_fields{Float64, Float64},
            two_fields{Float64, two_fields{Float64, singleton{Float64}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y),
        Float32,
        two_fields{two_fields{Float64, Float64}, two_fields{Float64, Float64}},
        6,
    )
    test_get_field_first_index_offset(
        @name(y.y.x),
        Float64,
        two_fields{
            two_fields{Float64, Float64},
            two_fields{Float64, two_fields{Float64, singleton{Float64}}},
        },
        3,
    )
    test_get_field_first_index_offset(
        @name(y.y.y.x),
        Float64,
        two_fields{
            two_fields{Float64, Float64},
            two_fields{Float64, two_fields{Float64, singleton{Float64}}},
        },
        4,
    )
end

@testset "broadcasted_get_field_type" begin
    struct singleton{T}
        x::T
    end
    struct two_fields{T1, T2}
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
        singleton{singleton{singleton{singleton{Float64}}}},
        singleton{singleton{singleton{Float64}}},
    )
    test_broadcasted_get_field_type(
        @name(x.x.x),
        singleton{singleton{singleton{singleton{Float64}}}},
        singleton{Float64},
    )
    test_broadcasted_get_field_type(
        @name(y.x),
        two_fields{
            two_fields{Float64, Float64},
            two_fields{Float64, two_fields{Float64, singleton{Float64}}},
        },
        Float64,
    )
    test_broadcasted_get_field_type(
        @name(y.y.y),
        two_fields{
            two_fields{Float64, Float64},
            two_fields{Float64, two_fields{Float64, singleton{Float64}}},
        },
        singleton{Float64},
    )
end

@testset "fieldmatrix to scalar fieldmatrix unit tests" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)
    surface_space = Spaces.level(face_space, half)
    seed!(1)
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    sfc_vec = random_field(FT, surface_space)
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

    dry_center_gs_unit = (; ρ = 1, ρe_tot = 1, uₕ = e¹²)
    center_gs_unit = (; dry_center_gs_unit..., ρatke = 1, ρχ = ρχ_unit)
    center_sgsʲ_unit = (; ρa = 1, ρae_tot = 1, ρaχ = ρaχ_unit)

    ᶜᶜmat3_uₕ_scalar = ᶜᶜmat3 .* (e¹²,)
    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶜᶜmat3_scalar_uₕ = ᶜᶜmat3 .* (e¹²,)
    ᶜᶜmat1_scalar_uₕ = ᶜᶜmat1 .* (e¹²,)
    ᶜᶠmat2_uₕ_u₃ = ᶜᶠmat2 .* (e¹² * e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶜmat3_ρχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit)), ᶜᶜmat3)
    ᶜᶜmat3_ρaχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit)), ᶜᶜmat3)
    ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit ⊠ e₃')), ᶜᶠmat2)
    ᶜᶠmat2_ρaχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit ⊠ e₃')), ᶜᶠmat2)

    A = MatrixFields.FieldMatrix(
        # GS-GS blocks:
        (@name(sfc), @name(sfc)) => I,
        (@name(c.ρ), @name(c.ρ)) => I,
        (@name(c.ρe_tot), @name(c.ρe_tot)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρatke), @name(c.ρatke)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρχ), @name(c.ρχ)) => deepcopy(ᶜᶜmat3),
        (@name(c.uₕ), @name(c.uₕ)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρ), @name(f.u₃)) => deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.ρe_tot), @name(f.u₃)) => deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.ρatke), @name(f.u₃)) => deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.ρχ), @name(f.u₃)) => deepcopy(ᶜᶠmat2_ρχ_u₃),
        (@name(f.u₃), @name(c.ρ)) => deepcopy(ᶠᶜmat2_u₃_scalar),
        (@name(f.u₃), @name(c.ρe_tot)) => deepcopy(ᶠᶜmat2_u₃_scalar),
        (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        # GS-SGS blocks:
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) =>
            deepcopy(ᶜᶜmat3),
        (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) =>
            deepcopy(ᶜᶜmat3),
        (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) =>
            deepcopy(ᶜᶜmat3),
        (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) =>
            deepcopy(ᶜᶜmat3),
        (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) =>
            deepcopy(ᶜᶜmat3),
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) => deepcopy(ᶜᶜmat3),
        (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_ρχ_scalar,
        (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_scalar_uₕ,
        (@name(c.sgsʲs.:(1).ρa), @name(c.uₕ)) => ᶜᶜmat3_uₕ_scalar,
        (@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)) =>
            deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.ρatke), @name(f.sgsʲs.:(1).u₃)) =>
            deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.ρχ), @name(f.sgsʲs.:(1).u₃)) => deepcopy(ᶜᶠmat2_ρχ_u₃),
        (@name(c.uₕ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_uₕ_u₃,
        (@name(f.u₃), @name(c.sgsʲs.:(1).ρa)) => deepcopy(ᶠᶜmat2_u₃_scalar),
        (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => deepcopy(ᶠᶠmat3_u₃_u₃),
        # SGS-SGS blocks:
        (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)) => I,
        (@name(c.sgsʲs.:(1).ρae_tot), @name(c.sgsʲs.:(1).ρae_tot)) => I,
        (@name(c.sgsʲs.:(1).ρaχ), @name(c.sgsʲs.:(1).ρaχ)) => I,
        (@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)) =>
            deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.sgsʲs.:(1).ρae_tot), @name(f.sgsʲs.:(1).u₃)) =>
            deepcopy(ᶜᶠmat2_scalar_u₃),
        (@name(c.sgsʲs.:(1).ρaχ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_ρaχ_u₃,
        (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρa)) =>
            deepcopy(ᶠᶜmat2_u₃_scalar),
        (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρae_tot)) =>
            deepcopy(ᶠᶜmat2_u₃_scalar),
        (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
            deepcopy(ᶠᶠmat3_u₃_u₃),
    )

    scalar_keys = MatrixFields.get_scalar_keys(A)
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
