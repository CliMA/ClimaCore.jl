using Test
using StaticArrays
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets


domain = Domains.RectangleDomain(
    -pi..pi,
    -pi..pi;
    x1periodic = true,
    x2periodic = true,
)
mesh = Meshes.EquispacedRectangleMesh(domain, 8, 8)
grid_topology = Topologies.GridTopology(mesh)

Nq = 5
quad = Spaces.Quadratures.GLL{Nq}()

space = Spaces.SpectralElementSpace2D(grid_topology, quad)

coords = Fields.coordinate_field(space)

@testset "gradient" begin
    f = sin.(coords.x1 .+ 2 .* coords.x2)

    grad = Operators.Gradient()
    gradf = grad.(f)
    Spaces.weighted_dss!(gradf)

    @test gradf.u1 ≈ cos.(coords.x1 .+ 2 .* coords.x2) rtol = 1e-2
    @test gradf.u2 ≈ 2 .* cos.(coords.x1 .+ 2 .* coords.x2) rtol = 1e-2
end


@testset "weak gradient" begin
    f = sin.(coords.x1 .+ 2 .* coords.x2)

    wgrad = Operators.WeakGradient()
    gradf = wgrad.(f)
    Spaces.weighted_dss!(gradf)

    @test gradf.u1 ≈ cos.(coords.x1 .+ 2 .* coords.x2) rtol = 1e-2
    @test gradf.u2 ≈ 2 .* cos.(coords.x1 .+ 2 .* coords.x2) rtol = 1e-2
end

@testset "curl" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 4 .* coords.x2),
        )

    curl = Operators.StrongCurl()
    curlv = curl.(v)
    Spaces.weighted_dss!(curlv)
    curlv_ref =
        .-3 .* sin.(3 .* coords.x1 .+ 4 .* coords.x2) .-
        2 .* cos.(coords.x1 .+ 2 .* coords.x2)

    @test curlv.u³ ≈ curlv_ref rtol = 1e-2
end



@testset "curl-curl" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 2 .* coords.x2),
        )
    curlv_ref =
        .-3 .* sin.(3 .* coords.x1 .+ 2 .* coords.x2) .-
        2 .* cos.(coords.x1 .+ 2 .* coords.x2)
    curlcurlv_ref1 =
        .-6 .* cos.(3 .* coords.x1 .+ 2 .* coords.x2) .+
        4 .* sin.(coords.x1 .+ 2 .* coords.x2)
    curlcurlv_ref2 =
        9 .* cos.(3 .* coords.x1 .+ 2 .* coords.x2) .-
        2 .* sin.(coords.x1 .+ 2 .* coords.x2)

    curl = Operators.StrongCurl()
    curlcurlv =
        Geometry.Cartesian12Vector.(
            curl.(Geometry.Covariant3Vector.(curl.(v).u³)),
        )
    Spaces.weighted_dss!(curlcurlv)

    @test curlcurlv.u1 ≈ curlcurlv_ref1 rtol = 1e-2
    @test curlcurlv.u2 ≈ curlcurlv_ref2 rtol = 4e-2
end

@testset "weak curl-strong curl" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 2 .* coords.x2),
        )
    curlv_ref =
        .-3 .* sin.(3 .* coords.x1 .+ 2 .* coords.x2) .-
        2 .* cos.(coords.x1 .+ 2 .* coords.x2)
    curlcurlv_ref1 =
        .-6 .* cos.(3 .* coords.x1 .+ 2 .* coords.x2) .+
        4 .* sin.(coords.x1 .+ 2 .* coords.x2)
    curlcurlv_ref2 =
        9 .* cos.(3 .* coords.x1 .+ 2 .* coords.x2) .-
        2 .* sin.(coords.x1 .+ 2 .* coords.x2)

    curl = Operators.StrongCurl()
    wcurl = Operators.WeakCurl()
    curlcurlv =
        Geometry.Cartesian12Vector.(
            wcurl.(Geometry.Covariant3Vector.(curl.(v).u³)),
        )
    Spaces.weighted_dss!(curlcurlv)

    @test curlcurlv.u1 ≈ curlcurlv_ref1 rtol = 1e-2
    @test curlcurlv.u2 ≈ curlcurlv_ref2 rtol = 4e-2
end



@testset "weak curl" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 4 .* coords.x2),
        )

    wcurl = Operators.WeakCurl()
    curlv = wcurl.(v)
    Spaces.weighted_dss!(curlv)
    curlv_ref =
        .-3 .* sin.(3 .* coords.x1 .+ 4 .* coords.x2) .-
        2 .* cos.(coords.x1 .+ 2 .* coords.x2)

    @test curlv.u³ ≈ curlv_ref rtol = 1e-2
end

@testset "div" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 2 .* coords.x2),
        )

    div = Operators.StrongDivergence()
    divv = div.(v)
    Spaces.weighted_dss!(divv)
    divv_ref =
        cos.(coords.x1 .+ 2 .* coords.x2) .-
        2 .* sin.(3 .* coords.x1 .+ 2 .* coords.x2)

    @test divv ≈ divv_ref rtol = 1e-2
end


@testset "weak div" begin
    v =
        Geometry.Cartesian12Vector.(
            sin.(coords.x1 .+ 2 .* coords.x2),
            cos.(3 .* coords.x1 .+ 2 .* coords.x2),
        )

    wdiv = Operators.WeakDivergence()
    divv = wdiv.(v)
    Spaces.weighted_dss!(divv)
    divv_ref =
        cos.(coords.x1 .+ 2 .* coords.x2) .-
        2 .* sin.(3 .* coords.x1 .+ 2 .* coords.x2)

    @test divv ≈ divv_ref rtol = 1e-2
end


@testset "annhilator property: curl-grad" begin
    f = sin.(coords.x1 .+ 2 .* coords.x2)

    grad = Operators.Gradient()
    gradf = grad.(f)
    Spaces.weighted_dss!(gradf)

    curl = Operators.StrongCurl()
    curlgradf = curl.(gradf)
    Spaces.weighted_dss!(curlgradf)

    @test norm(curlgradf) < 1e-12
end


@testset "annhilator property: div-curl" begin
    v = Geometry.Covariant3Vector.(sin.(coords.x1 .+ 2 .* coords.x2))
    curl = Operators.StrongCurl()
    curlv = curl.(v)
    Spaces.weighted_dss!(curlv)

    div = Operators.StrongDivergence()
    divcurlv = div.(curlv)
    Spaces.weighted_dss!(divcurlv)

    @test norm(divcurlv) < 1e-12
end
