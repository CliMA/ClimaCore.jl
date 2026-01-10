#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Limiters", "vertical_mass_borrowing_limiter.jl"))
=#
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Fields, Spaces, Limiters
using ClimaCore.Geometry
using ClimaCore.Grids
using ClimaCore.CommonGrids
import ClimaCore
using Test
using Random


import Plots
import ClimaCorePlots
dir = "vert_mass_borrow"
device_name = ClimaComms.device() isa ClimaComms.CUDADevice ? "GPU" : "CPU"
path = joinpath(@__DIR__, "output", dir, device_name)
mkpath(path)
function plot_results(f, f₀)
    col = Fields.ColumnIndex((1, 1), 1)
    fcol = f[col]
    f₀col = f₀[col]
    p = Plots.plot()
    Plots.plot(fcol; label = "field")
    Plots.plot!(f₀col; label = "initial")
    Plots.savefig(joinpath(path, "lim.png"))
end
# usage:
# plot_results(ρq, ρq_init)


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
    FT = Float64
    Random.seed!(1134)
    z_elem = 10
    z_min = 0
    z_max = 1
    device = ClimaComms.device()
    grid = ColumnGrid(FT; z_elem, z_min, z_max, device)
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
    ρq_init = ρ .* q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == z_elem == 10
    @test 0.3 ≤ count(x -> x < 0, parent(q)) / 10 ≤ 0.5 # ensure reasonable percentage of points are negative

    @test -2 * perturb_q ≤ minimum(q) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter((0.0,))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(q)
    ρq = ρ .* q
    @test isapprox(sum(ρq), sum_ρq_init; atol = 1e-15)
    @test isapprox(sum(ρq), sum_ρq_init; rtol = 1e-10)
    plot_results(ClimaCore.to_cpu(ρq), ClimaCore.to_cpu(ρq_init))
end

@testset "Vertical mass borrowing limiter - sphere" begin
    FT = Float64
    z_elem = 10
    z_min = 0
    z_max = 1
    radius = 10
    h_elem = 10
    n_quad_points = 4
    grid = ExtrudedCubedSphereGrid(FT;
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
    q = map(coord -> (a = 0.1, b = 0.1), coords)
    scalar_field = map(coord -> 0.1, coords)
    (; z) = coords
    perturb_field!(scalar_field; perturb_radius = perturb_q)
    q.a .= scalar_field
    scalar_field = map(coord -> 0.1, coords)
    perturb_field!(scalar_field; perturb_radius = perturb_q)
    q.b .= scalar_field
    perturb_field!(ρ; perturb_radius = perturb_ρ)
    ρq_init = ρ .* q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == 2 * z_elem * h_elem^2 * 6 * n_quad_points^2 == 192000
    @test 0.10 ≤ count(x -> x < 0.0001, parent(q)) / 192000 ≤ 1 # ensure 10% of points are negative

    @test -2 * perturb_q ≤ minimum(parent(q)) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter((0.0, 0.0))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(parent(q))
    ρq = ρ .* q
    @test isapprox(sum(ρq.a), sum_ρq_init.a; atol = 0.07)
    @test isapprox(sum(ρq.a), sum_ρq_init.a; rtol = 0.07)
    @test isapprox(sum(ρq.b), sum_ρq_init.b; atol = 0.07)
    @test isapprox(sum(ρq.b), sum_ρq_init.b; rtol = 0.07)
end

@testset "Vertical mass borrowing limiter - deep atmosphere" begin
    FT = Float64
    Random.seed!(12214)
    z_elem = 10
    z_min = 0
    z_max = 1
    radius = 10
    h_elem = 10
    n_quad_points = 4
    grid = ExtrudedCubedSphereGrid(FT;
        z_elem,
        z_min,
        z_max,
        radius,
        global_geometry = Geometry.DeepSphericalGlobalGeometry{FT}(radius),
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
    ρq_init = ρ .* q
    sum_ρq_init = sum(ρq_init)

    # Test that the minimum is below 0
    @test length(parent(q)) == z_elem * h_elem^2 * 6 * n_quad_points^2 == 96000
    @test 0.10 ≤ count(x -> x < 0.0001, parent(q)) / 96000 ≤ 1 # ensure 10% of points are negative

    @test -2 * perturb_q ≤ minimum(q) ≤ -tol
    limiter = Limiters.VerticalMassBorrowingLimiter((0.0,))
    Limiters.apply_limiter!(q, ρ, limiter)
    @test 0 ≤ minimum(q)
    ρq = ρ .* q
    @test isapprox(sum(ρq), sum_ρq_init; atol = 0.1)
    @test isapprox(sum(ρq), sum_ρq_init; rtol = 0.001)
end
