using Test
using Combinatorics

@testset "Equiangular cubed-sphere warp tests" begin
    import ClimaCore.Meshes: equiangular_sphere_warp

    # Create function alias for shorter formatting
    eacsw = equiangular_sphere_warp

    @testset "check radius" begin
        @test hypot(eacsw(3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(eacsw(-3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(eacsw(1.1, -2.2, 3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(eacsw(1.1, -2.2, -3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(eacsw(1.1, 3.0, 0.0)...) ≈ 3.0 rtol = eps()
        @test hypot(eacsw(1.1, -3.0, 0.0)...) ≈ 3.0 rtol = eps()
    end

    @testset "check sign" begin
        @test sign.(eacsw(3.0, -2.2, 1.3)) == sign.((3.0, -2.2, 1.3))
        @test sign.(eacsw(-3.0, -2.2, 1.3)) == sign.((-3.0, -2.2, 1.3))
        @test sign.(eacsw(1.1, -2.2, 3.0)) == sign.((1.1, -2.2, 3.0))
        @test sign.(eacsw(1.1, -2.2, -3.0)) == sign.((1.1, -2.2, -3.0))
        @test sign.(eacsw(1.1, 3.0, 0.0)) == sign.((1.1, 3.0, 0.0))
        @test sign.(eacsw(1.1, -3.0, 0.0)) == sign.((1.1, -3.0, 0.0))
    end

    @testset "check continuity" begin
        for (u, v) in zip(
            permutations([3.0, 2.999999999, 1.3]),
            permutations([2.999999999, 3.0, 1.3]),
        )
            @test all(eacsw(u...) .≈ eacsw(v...))
        end
        for (u, v) in zip(
            permutations([3.0, -2.999999999, 1.3]),
            permutations([2.999999999, -3.0, 1.3]),
        )
            @test all(eacsw(u...) .≈ eacsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, 2.999999999, 1.3]),
            permutations([-2.999999999, 3.0, 1.3]),
        )
            @test all(eacsw(u...) .≈ eacsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, -2.999999999, 1.3]),
            permutations([-2.999999999, -3.0, 1.3]),
        )
            @test all(eacsw(u...) .≈ eacsw(v...))
        end
    end
end

@testset "Equiangular cubed-sphere unwarp tests" begin
    import ClimaCore.Meshes: equiangular_sphere_warp, equiangular_sphere_unwarp

    # Create function aliases for shorter formatting
    eacsw = equiangular_sphere_warp
    eacsu = equiangular_sphere_unwarp

    for u in permutations([3.0, 2.999999999, 1.3])
        @test all(eacsu(eacsw(u...)...) .≈ u)
    end
    for u in permutations([3.0, -2.999999999, 1.3])
        @test all(eacsu(eacsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, 2.999999999, 1.3])
        @test all(eacsu(eacsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, -2.999999999, 1.3])
        @test all(eacsu(eacsw(u...)...) .≈ u)
    end
end

@testset "Equidistant cubed-sphere warp tests" begin
    import ClimaCore.Meshes: equidistant_sphere_warp

    # Create function alias for shorter formatting
    edcsw = equidistant_sphere_warp

    @testset "check radius" begin
        @test hypot(edcsw(3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(edcsw(-3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(edcsw(1.1, -2.2, 3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(edcsw(1.1, -2.2, -3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(edcsw(1.1, 3.0, 0.0)...) ≈ 3.0 rtol = eps()
        @test hypot(edcsw(1.1, -3.0, 0.0)...) ≈ 3.0 rtol = eps()
    end

    @testset "check sign" begin
        @test sign.(edcsw(3.0, -2.2, 1.3)) == sign.((3.0, -2.2, 1.3))
        @test sign.(edcsw(-3.0, -2.2, 1.3)) == sign.((-3.0, -2.2, 1.3))
        @test sign.(edcsw(1.1, -2.2, 3.0)) == sign.((1.1, -2.2, 3.0))
        @test sign.(edcsw(1.1, -2.2, -3.0)) == sign.((1.1, -2.2, -3.0))
        @test sign.(edcsw(1.1, 3.0, 0.0)) == sign.((1.1, 3.0, 0.0))
        @test sign.(edcsw(1.1, -3.0, 0.0)) == sign.((1.1, -3.0, 0.0))
    end

    @testset "check continuity" begin
        for (u, v) in zip(
            permutations([3.0, 2.999999999, 1.3]),
            permutations([2.999999999, 3.0, 1.3]),
        )
            @test all(edcsw(u...) .≈ edcsw(v...))
        end
        for (u, v) in zip(
            permutations([3.0, -2.999999999, 1.3]),
            permutations([2.999999999, -3.0, 1.3]),
        )
            @test all(edcsw(u...) .≈ edcsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, 2.999999999, 1.3]),
            permutations([-2.999999999, 3.0, 1.3]),
        )
            @test all(edcsw(u...) .≈ edcsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, -2.999999999, 1.3]),
            permutations([-2.999999999, -3.0, 1.3]),
        )
            @test all(edcsw(u...) .≈ edcsw(v...))
        end
    end
end

@testset "Equidistant cubed-sphere unwarp tests" begin
    import ClimaCore.Meshes: equidistant_sphere_warp, equidistant_sphere_unwarp

    # Create function aliases for shorter formatting
    edcsw = equidistant_sphere_warp
    edcsu = equidistant_sphere_unwarp

    for u in permutations([3.0, 2.999999999, 1.3])
        @test all(edcsu(edcsw(u...)...) .≈ u)
    end
    for u in permutations([3.0, -2.999999999, 1.3])
        @test all(edcsu(edcsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, 2.999999999, 1.3])
        @test all(edcsu(edcsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, -2.999999999, 1.3])
        @test all(edcsu(edcsw(u...)...) .≈ u)
    end
end

@testset "Conformal cubed-sphere warp tests" begin
    import ClimaCore.Meshes: conformal_sphere_warp

    # Create function alias for shorter formatting
    ccsw = conformal_sphere_warp

    @testset "check radius" begin
        @test hypot(ccsw(3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(ccsw(-3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(ccsw(1.1, -2.2, 3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(ccsw(1.1, -2.2, -3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(ccsw(1.1, 3.0, 0.0)...) ≈ 3.0 rtol = eps()
        @test hypot(ccsw(1.1, -3.0, 0.0)...) ≈ 3.0 rtol = eps()
    end

    @testset "check sign" begin
        @test sign.(ccsw(3.0, -2.2, 1.3)) == sign.((3.0, -2.2, 1.3))
        @test sign.(ccsw(-3.0, -2.2, 1.3)) == sign.((-3.0, -2.2, 1.3))
        @test sign.(ccsw(1.1, -2.2, 3.0)) == sign.((1.1, -2.2, 3.0))
        @test sign.(ccsw(1.1, -2.2, -3.0)) == sign.((1.1, -2.2, -3.0))
        @test sign.(ccsw(1.1, 3.0, -2.2)) == sign.((1.1, 3.0, -2.2))
        @test sign.(ccsw(1.1, -3.0, -2.2)) == sign.((1.1, -3.0, -2.2))
    end

    @testset "check continuity" begin
        for (u, v) in zip(
            permutations([3.0, 2.999999999, 1.3]),
            permutations([2.999999999, 3.0, 1.3]),
        )
            @test all(ccsw(u...) .≈ ccsw(v...))
        end
        for (u, v) in zip(
            permutations([3.0, -2.999999999, 1.3]),
            permutations([2.999999999, -3.0, 1.3]),
        )
            @test all(ccsw(u...) .≈ ccsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, 2.999999999, 1.3]),
            permutations([-2.999999999, 3.0, 1.3]),
        )
            @test all(ccsw(u...) .≈ ccsw(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, -2.999999999, 1.3]),
            permutations([-2.999999999, -3.0, 1.3]),
        )
            @test all(ccsw(u...) .≈ ccsw(v...))
        end
    end
end

@testset "Conformal cubed-sphere unwarp tests" begin
    import ClimaCore.Meshes: conformal_sphere_warp, conformal_sphere_unwarp

    # Create function aliases for shorter formatting
    ccsw = conformal_sphere_warp
    ccsu = conformal_sphere_unwarp

    for u in permutations([3.0, 2.999999999, 1.3])
        @test all(ccsu(ccsw(u...)...) .≈ u)
    end
    for u in permutations([3.0, -2.999999999, 1.3])
        @test all(ccsu(ccsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, 2.999999999, 1.3])
        @test all(ccsu(ccsw(u...)...) .≈ u)
    end
    for u in permutations([-3.0, -2.999999999, 1.3])
        @test all(ccsu(ccsw(u...)...) .≈ u)
    end
end
