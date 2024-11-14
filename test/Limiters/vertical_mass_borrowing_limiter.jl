#=
julia --project
using Revise; include(joinpath("test", "Limiters", "vertical_mass_borrowing_limiter.jl"))
=#
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Fields, Spaces, Limiters
using ClimaCore.RecursiveApply
using ClimaCore.Geometry
using ClimaCore.Grids
using ClimaCore.CommonGrids
using Test
using Random

function perturb_field!(f::Fields.Field; perturb_radius)
    device = ClimaComms.device(f)
    ArrayType = ClimaComms.array_type(device)
    rand_data = ArrayType(rand(size(parent(f))...)) # [0-1]
    rand_data = rand_data .- sum(rand_data) / length(rand_data) # make centered about 0 ([-0.5:0.5])
    rand_data = (rand_data ./ maximum(rand_data)) .* perturb_radius # rand_data now in [-perturb_radius:perturb_radius]
    parent(f) .= parent(f) .+ rand_data # f in [f ± perturb_radius]
    return nothing
end

@testset "Vertical mass borrowing limiter - column" begin
    Random.seed!(1234)
    z_elem = 10
    z_min = 0
    z_max = 1
    device = ClimaComms.device()
    grid = ColumnGrid(; z_elem, z_min, z_max, device)
    cspace = Spaces.FiniteDifferenceSpace(grid, Grids.CellCenter())
    tol = 0.01
    perturb_q = 0.3
    perturb_ρ = 0.2

    # Initialize fields
    coords = Fields.coordinate_field(cspace)
    ρ = map(coord -> 1.0, coords)
    q = map(coord -> 0.1, coords)
    (; z) = coords
    perturb_field!(q; perturb_radius = perturb_q)
    perturb_field!(ρ; perturb_radius = perturb_ρ)
    ρq_init = ρ .⊠ q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == z_elem == 10
    @test 0.3 ≤ count(x -> x < 0, parent(q)) / 10 ≤ 0.5 # ensure reasonable percentage of points are negative

    @test -2 * perturb_q ≤ minimum(q) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter(q, (0.0,))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(q)
    ρq = ρ .⊠ q
    @test isapprox(sum(ρq), sum_ρq_init; atol = 1e-15)
    @test isapprox(sum(ρq), sum_ρq_init; rtol = 1e-10)
    # @show sum(ρq)     # 0.07388931313511024
    # @show sum_ρq_init # 0.07388931313511025
end

@testset "Vertical mass borrowing limiter - sphere" begin
    z_elem = 10
    z_min = 0
    z_max = 1
    radius = 10
    h_elem = 10
    n_quad_points = 4
    grid = ExtrudedCubedSphereGrid(;
        z_elem,
        z_min,
        z_max,
        radius,
        h_elem,
        n_quad_points,
    )
    cspace = Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
    tol = 0.01
    perturb_q = 0.2
    perturb_ρ = 0.1

    # Initialize fields
    coords = Fields.coordinate_field(cspace)
    ρ = map(coord -> 1.0, coords)
    q = map(coord -> 0.1, coords)

    perturb_field!(q; perturb_radius = perturb_q)
    perturb_field!(ρ; perturb_radius = perturb_ρ)
    ρq_init = ρ .⊠ q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == z_elem * h_elem^2 * 6 * n_quad_points^2 == 96000
    @test 0.10 ≤ count(x -> x < 0.0001, parent(q)) / 96000 ≤ 1 # ensure 10% of points are negative

    @test -2 * perturb_q ≤ minimum(q) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter(q, (0.0,))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(q)
    ρq = ρ .⊠ q
    @test isapprox(sum(ρq), sum_ρq_init; atol = 0.029)
    @test isapprox(sum(ρq), sum_ρq_init; rtol = 0.00023)
    # @show sum(ρq)     # 125.5483524237572
    # @show sum_ρq_init # 125.52052986152977
end

@testset "Vertical mass borrowing limiter - deep atmosphere" begin
    z_elem = 10
    z_min = 0
    z_max = 1
    radius = 10
    h_elem = 10
    n_quad_points = 4
    grid = ExtrudedCubedSphereGrid(;
        z_elem,
        z_min,
        z_max,
        radius,
        global_geometry = Geometry.DeepSphericalGlobalGeometry(radius),
        h_elem,
        n_quad_points,
    )
    cspace = Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
    tol = 0.01
    perturb_q = 0.2
    perturb_ρ = 0.1

    # Initialize fields
    coords = Fields.coordinate_field(cspace)
    ρ = map(coord -> 1.0, coords)
    q = map(coord -> 0.1, coords)

    perturb_field!(q; perturb_radius = perturb_q)
    perturb_field!(ρ; perturb_radius = perturb_ρ)
    ρq_init = ρ .⊠ q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == z_elem * h_elem^2 * 6 * n_quad_points^2 == 96000
    @test 0.10 ≤ count(x -> x < 0.0001, parent(q)) / 96000 ≤ 1 # ensure 10% of points are negative

    @test -2 * perturb_q ≤ minimum(q) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter(q, (0.0,))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(q)
    ρq = ρ .⊠ q
    @test isapprox(sum(ρq), sum_ρq_init; atol = 0.269)
    @test isapprox(sum(ρq), sum_ρq_init; rtol = 0.00199)
    # @show sum(ρq)     # 138.90494931641584
    # @show sum_ρq_init # 139.1735731377394
end
