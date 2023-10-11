import LinearAlgebra: I, norm
import ClimaCore.Utilities: half
import ClimaCore.RecursiveApply: ⊠
import ClimaCore.MatrixFields: @name

include("matrix_field_test_utils.jl")

# This broadcast must be wrapped in a function to be tested with @test_opt.
field_matrix_mul!(b, A, x) = @. b = A * x

function test_field_matrix_solver(;
    test_name,
    alg,
    A,
    b,
    ignore_approximation_error = false,
    skip_correctness_test = false,
)
    @testset "$test_name" begin
        x = similar(b)
        b_test = similar(b)
        solver = FieldMatrixSolver(alg, A, b)
        args = (solver, x, A, b)

        solve_time = @benchmark field_matrix_solve!(args...)
        mul_time = @benchmark field_matrix_mul!(b_test, A, x)

        solve_time_rounded = round(solve_time; sigdigits = 2)
        mul_time_rounded = round(mul_time; sigdigits = 2)
        time_ratio = solve_time_rounded / mul_time_rounded
        time_ratio_rounded = round(time_ratio; sigdigits = 2)

        # If possible, test that A * (inv(A) * b) == b.
        if skip_correctness_test
            relative_error =
                norm(abs.(parent(b_test) .- parent(b))) / norm(parent(b))
            relative_error_rounded = round(relative_error; sigdigits = 2)
            error_string = "Relative Error = $(relative_error_rounded * 100) %"
        else
            if ignore_approximation_error
                @assert alg isa MatrixFields.ApproximateFactorizationSolve
                b_view = MatrixFields.field_vector_view(b)
                A₁, A₂ =
                    MatrixFields.approximate_factors(alg.name_pairs₁, A, b_view)
                @. b_test = A₁ * A₂ * x
            end
            max_error = maximum(abs.(parent(b_test) .- parent(b)))
            max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))
            error_string = "Maximum Error = $max_eps_error eps"
        end

        @info "$test_name:\n\tSolve Time = $solve_time_rounded s, \
               Multiplication Time = $mul_time_rounded s (Ratio = \
               $time_ratio_rounded)\n\t$error_string"

        skip_correctness_test || @test max_eps_error <= 3

        @test_opt ignored_modules = ignore_cuda FieldMatrixSolver(alg, A, b)
        @test_opt ignored_modules = ignore_cuda field_matrix_solve!(args...)
        @test_opt ignored_modules = ignore_cuda field_matrix_mul!(b, A, x)

        using_cuda || @test @allocated(field_matrix_solve!(args...)) == 0
        using_cuda || @test @allocated(field_matrix_mul!(b, A, x)) == 0
    end
end

@testset "FieldMatrixSolver Unit Tests" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)
    surface_space = Spaces.level(face_space, half)

    seed!(1) # ensures reproducibility

    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    sfc_vec = random_field(FT, surface_space)

    # Make each random square matrix diagonally dominant in order to avoid large
    # large roundoff errors when computing its inverse. Scale the non-square
    # matrices by the same amount as the square matrices.
    λ = 10 # scale factor
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat1 = random_field(DiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)
    ᶜᶠmat4 = random_field(QuaddiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat4 = random_field(QuaddiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat5 = random_field(PentadiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat5 = random_field(PentadiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)

    for (vector, matrix, string1, string2) in (
        (sfc_vec, I, "UniformScaling", "a single level"),
        (ᶜvec, I, "UniformScaling", "cell centers"),
        (ᶠvec, I, "UniformScaling", "cell faces"),
        (ᶜvec, ᶜᶜmat1, "diagonal matrix", "cell centers"),
        (ᶠvec, ᶠᶠmat1, "diagonal matrix", "cell faces"),
        (ᶜvec, ᶜᶜmat3, "tri-diagonal matrix", "cell centers"),
        (ᶠvec, ᶠᶠmat3, "tri-diagonal matrix", "cell faces"),
        (ᶜvec, ᶜᶜmat5, "penta-diagonal matrix", "cell centers"),
        (ᶠvec, ᶠᶠmat5, "penta-diagonal matrix", "cell faces"),
    )
        test_field_matrix_solver(;
            test_name = "$string1 solve on $string2",
            alg = MatrixFields.BlockDiagonalSolve(),
            A = MatrixFields.FieldMatrix((@name(_), @name(_)) => matrix),
            b = Fields.FieldVector(; _ = vector),
        )
    end

    # TODO: Add a simple test where typeof(x) != typeof(b).

    for alg in (
        MatrixFields.BlockDiagonalSolve(),
        MatrixFields.BlockLowerTriangularSolve(@name(c)),
        MatrixFields.SchurComplementSolve(@name(f)),
        MatrixFields.ApproximateFactorizationSolve((@name(c), @name(c))),
    )
        test_field_matrix_solver(;
            test_name = "$(typeof(alg).name.name) for a block diagonal matrix \
                         with diagonal and penta-diagonal blocks",
            alg,
            A = MatrixFields.FieldMatrix(
                (@name(c), @name(c)) => ᶜᶜmat1,
                (@name(f), @name(f)) => ᶠᶠmat5,
            ),
            b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
        )
    end

    test_field_matrix_solver(;
        test_name = "BlockDiagonalSolve for a block diagonal matrix with \
                     tri-diagonal and penta-diagonal blocks",
        alg = MatrixFields.BlockDiagonalSolve(),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat3,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
    )

    test_field_matrix_solver(;
        test_name = "BlockLowerTriangularSolve for a block lower triangular \
                     matrix with tri-diagonal, bi-diagonal, and penta-diagonal \
                     blocks",
        alg = MatrixFields.BlockLowerTriangularSolve(@name(c)),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat3,
            (@name(f), @name(c)) => ᶠᶜmat2,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
    )

    test_field_matrix_solver(;
        test_name = "SchurComplementSolve for a block matrix with diagonal, \
                     quad-diagonal, bi-diagonal, and penta-diagonal blocks",
        alg = MatrixFields.SchurComplementSolve(@name(f)),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat1,
            (@name(c), @name(f)) => ᶜᶠmat4,
            (@name(f), @name(c)) => ᶠᶜmat2,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
    )

    test_field_matrix_solver(;
        test_name = "ApproximateFactorizationSolve for a block matrix with \
                     tri-diagonal, quad-diagonal, bi-diagonal, and \
                     penta-diagonal blocks",
        alg = MatrixFields.ApproximateFactorizationSolve(
            (@name(c), @name(c));
            alg₂ = MatrixFields.SchurComplementSolve(@name(f)),
        ),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat3,
            (@name(c), @name(f)) => ᶜᶠmat4,
            (@name(f), @name(c)) => ᶠᶜmat2,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
        ignore_approximation_error = true,
    )
end

@testset "FieldMatrixSolver ClimaAtmos-Based Tests" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)
    surface_space = Spaces.level(face_space, half)

    seed!(1) # ensures reproducibility

    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    sfc_vec = random_field(FT, surface_space)

    # Make each random square matrix diagonally dominant in order to avoid large
    # large roundoff errors when computing its inverse. Scale the non-square
    # matrices by the same amount as the square matrices.
    λ = 10 # scale factor
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)

    e¹² = Geometry.Covariant12Vector(1, 1)
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
    ᶜᶠmat2_uₕ_u₃ = ᶜᶠmat2 .* (e¹² * e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶜmat3_ρχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit)), ᶜᶜmat3)
    ᶜᶜmat3_ρaχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit)), ᶜᶜmat3)
    ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit ⊠ e₃')), ᶜᶠmat2)
    ᶜᶠmat2_ρaχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit ⊠ e₃')), ᶜᶠmat2)
    # We need to use Fix1 and Fix2 instead of defining anonymous functions in
    # order for the result of map to be inferrable.

    b_dry_dycore = Fields.FieldVector(;
        c = ᶜvec .* (dry_center_gs_unit,),
        f = ᶠvec .* ((; u₃ = e³),),
    )

    b_moist_dycore_diagnostic_edmf = Fields.FieldVector(;
        c = ᶜvec .* (center_gs_unit,),
        f = ᶠvec .* ((; u₃ = e³),),
    )

    b_moist_dycore_prognostic_edmf_prognostic_surface = Fields.FieldVector(;
        sfc = sfc_vec .* ((; T = 1),),
        c = ᶜvec .* ((; center_gs_unit..., sgsʲs = (center_sgsʲ_unit,)),),
        f = ᶠvec .* ((; u₃ = e³, sgsʲs = ((; u₃ = e³),)),),
    )

    test_field_matrix_solver(;
        test_name = "similar solve to ClimaAtmos's dry dycore with implicit \
                     acoustic waves",
        alg = MatrixFields.SchurComplementSolve(@name(f)),
        A = MatrixFields.FieldMatrix(
            (@name(c.ρ), @name(c.ρ)) => I,
            (@name(c.ρe_tot), @name(c.ρe_tot)) => I,
            (@name(c.uₕ), @name(c.uₕ)) => I,
            (@name(c.ρ), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(c.ρe_tot), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(f.u₃), @name(c.ρ)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(c.ρe_tot)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        ),
        b = b_dry_dycore,
    )

    test_field_matrix_solver(;
        test_name = "similar solve to ClimaAtmos's dry dycore with implicit \
                     acoustic waves and diffusion",
        alg = MatrixFields.ApproximateFactorizationSolve(
            (@name(c), @name(f)),
            (@name(f), @name(c)),
            (@name(f), @name(f));
            alg₁ = MatrixFields.SchurComplementSolve(@name(f)),
        ),
        A = MatrixFields.FieldMatrix(
            (@name(c.ρ), @name(c.ρ)) => I,
            (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
            (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3,
            (@name(c.ρ), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(c.ρe_tot), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(f.u₃), @name(c.ρ)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(c.ρe_tot)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        ),
        b = b_dry_dycore,
        ignore_approximation_error = true,
    )

    test_field_matrix_solver(;
        test_name = "similar solve to ClimaAtmos's moist dycore + diagnostic \
                     EDMF with implicit acoustic waves and SGS fluxes",
        alg = MatrixFields.ApproximateFactorizationSolve(
            (@name(c), @name(f)),
            (@name(f), @name(c)),
            (@name(f), @name(f));
            alg₁ = MatrixFields.SchurComplementSolve(@name(f)),
        ),
        A = MatrixFields.FieldMatrix(
            (@name(c.ρ), @name(c.ρ)) => I,
            (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
            (@name(c.ρatke), @name(c.ρatke)) => ᶜᶜmat3,
            (@name(c.ρχ), @name(c.ρχ)) => ᶜᶜmat3,
            (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3,
            (@name(c.ρ), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(c.ρe_tot), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(c.ρatke), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
            (@name(c.ρχ), @name(f.u₃)) => ᶜᶠmat2_ρχ_u₃,
            (@name(f.u₃), @name(c.ρ)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(c.ρe_tot)) => ᶠᶜmat2_u₃_scalar,
            (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        ),
        b = b_moist_dycore_diagnostic_edmf,
        ignore_approximation_error = true,
    )

    # TODO: This unit test is currently broken.
    # test_field_matrix_solver(;
    #     test_name = "similar solve to ClimaAtmos's moist dycore + prognostic \
    #                  EDMF + prognostic surface temperature with implicit \
    #                  acoustic waves and SGS fluxes",
    #     alg = MatrixFields.BlockLowerTriangularSolve(
    #         @name(c.sgsʲs),
    #         @name(f.sgsʲs);
    #         alg₁ = MatrixFields.SchurComplementSolve(@name(f)),
    #         alg₂ = MatrixFields.ApproximateFactorizationSolve(
    #             (@name(c), @name(f)),
    #             (@name(f), @name(c)),
    #             (@name(f), @name(f));
    #             alg₁ = MatrixFields.SchurComplementSolve(@name(f)),
    #         ),
    #     ),
    #     A = MatrixFields.FieldMatrix(
    #         # GS-GS blocks:
    #         (@name(sfc), @name(sfc)) => I,
    #         (@name(c.ρ), @name(c.ρ)) => I,
    #         (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
    #         (@name(c.ρatke), @name(c.ρatke)) => ᶜᶜmat3,
    #         (@name(c.ρχ), @name(c.ρχ)) => ᶜᶜmat3,
    #         (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3,
    #         (@name(c.ρ), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
    #         (@name(c.ρe_tot), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
    #         (@name(c.ρatke), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
    #         (@name(c.ρχ), @name(f.u₃)) => ᶜᶠmat2_ρχ_u₃,
    #         (@name(f.u₃), @name(c.ρ)) => ᶠᶜmat2_u₃_scalar,
    #         (@name(f.u₃), @name(c.ρe_tot)) => ᶠᶜmat2_u₃_scalar,
    #         (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
    #         # GS-SGS blocks:
    #         (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) => ᶜᶜmat3,
    #         (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) => ᶜᶜmat3,
    #         (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) => ᶜᶜmat3,
    #         (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) => ᶜᶜmat3,
    #         (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) => ᶜᶜmat3,
    #         (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) => ᶜᶜmat3,
    #         (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
    #         (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
    #         (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_ρχ_scalar,
    #         (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_uₕ_scalar,
    #         (@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_scalar_u₃,
    #         (@name(c.ρatke), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_scalar_u₃,
    #         (@name(c.ρχ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_ρχ_u₃,
    #         (@name(c.uₕ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_uₕ_u₃,
    #         (@name(f.u₃), @name(c.sgsʲs.:(1).ρa)) => ᶠᶜmat2_u₃_scalar,
    #         (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
    #         # SGS-SGS blocks:
    #         (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)) => I,
    #         (@name(c.sgsʲs.:(1).ρae_tot), @name(c.sgsʲs.:(1).ρae_tot)) => I,
    #         (@name(c.sgsʲs.:(1).ρaχ), @name(c.sgsʲs.:(1).ρaχ)) => I,
    #         (@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)) =>
    #             ᶜᶠmat2_scalar_u₃,
    #         (@name(c.sgsʲs.:(1).ρae_tot), @name(f.sgsʲs.:(1).u₃)) =>
    #             ᶜᶠmat2_scalar_u₃,
    #         (@name(c.sgsʲs.:(1).ρaχ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_ρaχ_u₃,
    #         (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρa)) =>
    #             ᶠᶜmat2_u₃_scalar,
    #         (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρae_tot)) =>
    #             ᶠᶜmat2_u₃_scalar,
    #         (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
    #     ),
    #     b = b_moist_dycore_prognostic_edmf_prognostic_surface,
    #     skip_correctness_test = true,
    # )
end
