#=
julia --project=test
using Revise; include(joinpath("test", "MatrixFields", "field_matrix_solvers.jl"))
=#
import Logging
import Logging: Debug
import LinearAlgebra: I, norm, ldiv!, mul!
import ClimaComms
import ClimaCore.Utilities: half
import ClimaCore.MatrixFields: @name
import ClimaCore:
    Spaces, MatrixFields, Fields, Domains, Meshes, Topologies, Geometry

include("matrix_field_test_utils.jl")

function test_field_matrix_solver(; test_name, alg, A, b, use_rel_error = false)
    @testset "$test_name" begin
        x = similar(b)
        A′ = FieldMatrixWithSolver(A, b, alg)
        @test zero(A′) isa typeof(A′)
        solve_time =
            @benchmark ClimaComms.@cuda_sync comms_device ldiv!(x, A′, b)

        b_test = similar(b)
        @test zero(b) isa typeof(b)
        mul_time =
            @benchmark ClimaComms.@cuda_sync comms_device mul!(b_test, A′, x)

        solve_time_rounded = round(solve_time; sigdigits = 2)
        mul_time_rounded = round(mul_time; sigdigits = 2)
        time_ratio = solve_time_rounded / mul_time_rounded
        time_ratio_rounded = round(time_ratio; sigdigits = 2)

        error_vector = abs.(parent(b_test) .- parent(b))
        if use_rel_error
            rel_error = norm(error_vector) / norm(parent(b))
            rel_error_rounded = round(rel_error; sigdigits = 2)
            error_string = "Relative Error = $rel_error_rounded"
        else
            max_error = maximum(error_vector)
            max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))
            error_string = "Maximum Error = $max_eps_error eps"
        end

        @info "$test_name:\n\tSolve Time = $solve_time_rounded s, \
               Multiplication Time = $mul_time_rounded s (Ratio = \
               $time_ratio_rounded)\n\t$error_string"

        if use_rel_error
            @test rel_error < 1e-5
        else
            @test max_eps_error <= 3
        end

        # In addition to ignoring the type instabilities from CUDA, ignore those
        # from CUBLAS (norm), KrylovKit (eigsolve), and CoreLogging (@debug).
        ignored = (
            cuda_frames...,
            cublas_frames...,
            AnyFrameModule(MatrixFields.KrylovKit),
            AnyFrameModule(Base.CoreLogging),
        )
        using_cuda ||
            @test_opt ignored_modules = ignored FieldMatrixWithSolver(A, b, alg)
        using_cuda || @test_opt ignored_modules = ignored ldiv!(x, A′, b)
        @test_opt ignored_modules = ignored mul!(b_test, A′, x)

        # TODO: fix broken test when Nv is added to the type space
        using_cuda || @test @allocated(ldiv!(x, A′, b)) ≤ 1536
        using_cuda || @test @allocated(mul!(b_test, A′, x)) == 0
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

    # Note: The round-off error of StationaryIterativeSolve can be much larger
    # on GPUs, so n_iters often has to be increased when using_cuda is true.

    for alg in (
        MatrixFields.BlockDiagonalSolve(),
        MatrixFields.BlockLowerTriangularSolve(@name(c)),
        MatrixFields.BlockArrowheadSolve(@name(c)),
        MatrixFields.ApproximateBlockArrowheadIterativeSolve(@name(c)),
        MatrixFields.StationaryIterativeSolve(; n_iters = using_cuda ? 28 : 18),
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
        test_name = "BlockArrowheadSolve for a block matrix with diagonal, \
                     quad-diagonal, bi-diagonal, and penta-diagonal blocks",
        alg = MatrixFields.BlockArrowheadSolve(@name(c)),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat1,
            (@name(c), @name(f)) => ᶜᶠmat4,
            (@name(f), @name(c)) => ᶠᶜmat2,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
    )

    # Since test_field_matrix_solver runs the solver many times with the same
    # values of x, A, and b for benchmarking, setting correlated_solves to true
    # is equivalent to setting n_iters to some very large number.
    test_field_matrix_solver(;
        test_name = "StationaryIterativeSolve with correlated_solves for a \
                     block matrix with tri-diagonal, quad-diagonal, \
                     bi-diagonal, and penta-diagonal blocks",
        alg = MatrixFields.StationaryIterativeSolve(; correlated_solves = true),
        A = MatrixFields.FieldMatrix(
            (@name(c), @name(c)) => ᶜᶜmat3,
            (@name(c), @name(f)) => ᶜᶠmat4,
            (@name(f), @name(c)) => ᶠᶜmat2,
            (@name(f), @name(f)) => ᶠᶠmat5,
        ),
        b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
    )

    # Each of the scaled identity matrices below was chosen to minimize the
    # value of ρ(I - P⁻¹ * A), which was found by setting print_radius to true.
    # Each value of n_iters below was then chosen to be the smallest value for
    # which the relative error was less than 1e-6.
    scaled_identity_matrix(scalar) =
        MatrixFields.FieldMatrix((@name(), @name()) => scalar * I)
    for (P_name, alg) in (
        (
            "no (identity matrix)",
            MatrixFields.StationaryIterativeSolve(;
                n_iters = using_cuda ? 10 : 7,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.3777
        (
            "Richardson (damped identity matrix)",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.CustomPreconditioner(
                    scaled_identity_matrix(FT(1.12)),
                ),
                n_iters = using_cuda ? 8 : 7,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.2294
        (
            "Jacobi (diagonal)",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.MainDiagonalPreconditioner(),
                n_iters = using_cuda ? 8 : 6,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.3241
        (
            "damped Jacobi (diagonal)",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.WeightedPreconditioner(
                    scaled_identity_matrix(FT(1.08)),
                    MatrixFields.MainDiagonalPreconditioner(),
                ),
                n_iters = using_cuda ? 8 : 7,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.2249
        (
            "block Jacobi (diagonal)",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.BlockDiagonalPreconditioner(),
                n_iters = 7,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.1450
        (
            "damped block Jacobi (diagonal)",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.WeightedPreconditioner(
                    scaled_identity_matrix(FT(1.002)),
                    MatrixFields.BlockDiagonalPreconditioner(),
                ),
                n_iters = 7,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.1427
        (
            "block arrowhead",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.BlockArrowheadPreconditioner(@name(c)),
                n_iters = 6,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.1356
        (
            "damped block arrowhead",
            MatrixFields.StationaryIterativeSolve(;
                P_alg = MatrixFields.BlockArrowheadPreconditioner(
                    @name(c);
                    P_alg₁ = MatrixFields.WeightedPreconditioner(
                        scaled_identity_matrix(FT(1.0001)),
                        MatrixFields.MainDiagonalPreconditioner(),
                    ),
                ),
                n_iters = 6,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.1355
        (
            "block arrowhead Schur complement",
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                @name(c);
                n_iters = 3,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.0009
        (
            "damped block arrowhead Schur complement",
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                @name(c);
                P_alg₁ = MatrixFields.WeightedPreconditioner(
                    scaled_identity_matrix(FT(1.09)),
                    MatrixFields.MainDiagonalPreconditioner(),
                ),
                n_iters = 2,
            ),
        ), # ρ(I - P⁻¹ * A) ≈ 0.000006
    )
        test_field_matrix_solver(;
            test_name = "approximate iterative solve with $P_name \
                         preconditioning for a block matrix with tri-diagonal, \
                         quad-diagonal, bi-diagonal, and penta-diagonal blocks",
            alg,
            A = MatrixFields.FieldMatrix(
                (@name(c), @name(c)) => ᶜᶜmat3,
                (@name(c), @name(f)) => ᶜᶠmat4,
                (@name(f), @name(c)) => ᶠᶜmat2,
                (@name(f), @name(f)) => ᶠᶠmat5,
            ),
            b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec),
            use_rel_error = true,
        )
    end

    @testset "approximate iterative solve with debugging" begin
        Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Debug)) do
            # Recreate the setup from the previous unit test.
            alg = MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                @name(c);
                P_alg₁ = MatrixFields.WeightedPreconditioner(
                    scaled_identity_matrix(FT(1.09)),
                    MatrixFields.MainDiagonalPreconditioner(),
                ),
                n_iters = 2,
            )
            A = MatrixFields.FieldMatrix(
                (@name(c), @name(c)) => ᶜᶜmat3,
                (@name(c), @name(f)) => ᶜᶠmat4,
                (@name(f), @name(c)) => ᶠᶜmat2,
                (@name(f), @name(f)) => ᶠᶠmat5,
            )
            b = Fields.FieldVector(; c = ᶜvec, f = ᶠvec)

            x = similar(b)
            A′ = FieldMatrixWithSolver(A, b, alg)

            # Compare the debugging logs to RegEx strings. Note that debugging the
            # spectral radius is currently not possible on GPUs.
            spectral_radius_logs =
                using_cuda ? () : ((:debug, r"ρ\(I \- inv\(P\) \* A\) ≈"),)
            error_norm_logs = (
                (:debug, r"||x[0] - x'||₂ ≈"),
                (:debug, r"||x[1] - x'||₂ ≈"),
                (:debug, r"||x[2] - x'||₂ ≈"),
            )
            logs = (spectral_radius_logs..., error_norm_logs...)
            @test_logs logs... min_level = Logging.Debug ldiv!(x, A′, b)
        end
    end
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

    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶠmat2_ρχ_u₃ = ᶜᶠmat2 .* (ρχ_unit,) .* (e₃',)

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
        alg = MatrixFields.BlockArrowheadSolve(@name(c)),
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
        alg = MatrixFields.ApproximateBlockArrowheadIterativeSolve(
            @name(c);
            n_iters = 6,
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
    )

    test_field_matrix_solver(;
        test_name = "similar solve to ClimaAtmos's moist dycore + diagnostic \
                     EDMF with implicit acoustic waves and SGS fluxes",
        alg = MatrixFields.ApproximateBlockArrowheadIterativeSolve(
            @name(c);
            n_iters = 6,
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
    )
    (
        A_moist_dycore_prognostic_edmf_prognostic_surface,
        b_moist_dycore_prognostic_edmf_prognostic_surface,
    ) = dycore_prognostic_EDMF_FieldMatrix(FT)
    test_field_matrix_solver(;
        test_name = "similar solve to ClimaAtmos's moist dycore + prognostic \
                     EDMF + prognostic surface temperature with implicit \
                     acoustic waves and SGS fluxes",
        alg = MatrixFields.BlockLowerTriangularSolve(
            @name(c.sgsʲs),
            @name(f.sgsʲs);
            alg₁ = MatrixFields.BlockArrowheadSolve(@name(c)),
            alg₂ = MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                @name(c);
                n_iters = 6,
            ),
        ),
        A = A_moist_dycore_prognostic_edmf_prognostic_surface,
        b = b_moist_dycore_prognostic_edmf_prognostic_surface,
    )
end

@testset "FieldMatrixSolver with CenterFiniteDifferenceSpace" begin
    # Set up FiniteDifferenceSpace
    FT = Float32
    zmax = FT(0)
    zmin = FT(-0.35)
    nelems = 5

    context = ClimaComms.context()
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zmin),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = nelems)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    space = Spaces.CenterFiniteDifferenceSpace(z_topology)

    # Create a field containing a `TridiagonalMatrixRow` at each point
    tridiag_type = MatrixFields.TridiagonalMatrixRow{FT}
    tridiag_field = Fields.Field(tridiag_type, space)

    # Set up objects for matrix solve
    A = MatrixFields.FieldMatrix((@name(_), @name(_)) => tridiag_field)
    field = Fields.ones(space)
    b = Fields.FieldVector(; _ = field)
    x = similar(b)
    A′ = FieldMatrixWithSolver(A, b)

    # Run matrix solve
    ldiv!(x, A′, b)
end
