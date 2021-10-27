using Test
using StaticArrays
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets

FT = Float64
domain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(-pi)..Geometry.XPoint{FT}(pi),
    Geometry.YPoint{FT}(-pi)..Geometry.YPoint{FT}(pi);
    x1periodic = true,
    x2periodic = true,
)

Nq = 5
quad = Spaces.Quadratures.GLL{Nq}()

grid_mesh = Meshes.EquispacedRectangleMesh(domain, 17, 16)
grid_topology = Topologies.GridTopology(grid_mesh)
grid_space = Spaces.SpectralElementSpace2D(grid_topology, quad)
grid_coords = Geometry.Cartesian12Point.(Fields.coordinate_field(grid_space))

ts_mesh = Meshes.TensorProductMesh(domain, 17, 16)
ts_topology = Topologies.GridTopology(ts_mesh)
ts_space = Spaces.SpectralElementSpace2D(ts_topology, quad)
ts_coords = Geometry.Cartesian12Point.(Fields.coordinate_field(ts_space))

grid_test_setup = (grid_topology, grid_space, grid_coords)
ts_test_setup = (ts_topology, ts_space, ts_coords)

@testset "interpolate / restrict" begin

    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        INq = 9
        Iquad = Spaces.Quadratures.GLL{INq}()
        Ispace = Spaces.SpectralElementSpace2D(topology, Iquad)

        I = Operators.Interpolate(Ispace)
        R = Operators.Restrict(space)

        f = sin.(coords.x1 .+ 2 .* coords.x2)

        interpolated_field = I.(f)
        Spaces.weighted_dss!(interpolated_field)

        @test axes(interpolated_field).quadrature_style == Iquad
        @test axes(interpolated_field).topology == topology

        restrict_field = R.(f)
        Spaces.weighted_dss!(restrict_field)

        @test axes(restrict_field).quadrature_style == quad
        @test axes(restrict_field).topology == topology

        interp_restrict_field = R.(I.(f))
        Spaces.weighted_dss!(interp_restrict_field)

        @test axes(interp_restrict_field).quadrature_style == quad
        @test axes(interp_restrict_field).topology == topology

        @test norm(interp_restrict_field .- f) ≤ 3.0e-4
    end
end

@testset "gradient" begin

    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x1 .+ 2 .* coords.x2)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss!(gradf)

        @test gradf ≈
              Geometry.Covariant12Vector.(
            Geometry.Cartesian12Vector.(
                cos.(coords.x1 .+ 2 .* coords.x2),
                2 .* cos.(coords.x1 .+ 2 .* coords.x2),
            ),
        ) rtol = 1e-2

        fv =
            Geometry.Cartesian12Vector.(
                sin.(coords.x1 .+ 2 .* coords.x2),
                cos.(coords.x1 .+ 2 .* coords.x2),
            )
        gradfv = Geometry.transform.(Ref(Geometry.Cartesian12Axis()), grad.(fv))
        Spaces.weighted_dss!(gradfv)
        @test eltype(gradfv) <: Geometry.Axis2Tensor
    end
end


@testset "weak gradient" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x1 .+ 2 .* coords.x2)

        wgrad = Operators.WeakGradient()
        gradf = wgrad.(f)
        Spaces.weighted_dss!(gradf)

        @test Geometry.Cartesian12Vector.(gradf) ≈
              Geometry.Cartesian12Vector.(
            cos.(coords.x1 .+ 2 .* coords.x2),
            2 .* cos.(coords.x1 .+ 2 .* coords.x2),
        ) rtol = 1e-2
    end
end

@testset "curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.Cartesian12Vector.(
                sin.(coords.x1 .+ 2 .* coords.x2),
                cos.(3 .* coords.x1 .+ 4 .* coords.x2),
            )

        curl = Operators.Curl()
        curlv = curl.(v)
        Spaces.weighted_dss!(curlv)
        curlv_ref =
            Geometry.Contravariant3Vector.(
                .-3 .* sin.(3 .* coords.x1 .+ 4 .* coords.x2) .-
                2 .* cos.(coords.x1 .+ 2 .* coords.x2),
            )

        @test curlv ≈ curlv_ref rtol = 1e-2
    end
end

@testset "curl-curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
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

        curl = Operators.Curl()
        curlcurlv =
            Geometry.Cartesian12Vector.(
                curl.(Geometry.Covariant3Vector.(curl.(v))),
            )
        Spaces.weighted_dss!(curlcurlv)

        @test curlcurlv ≈
              Geometry.Cartesian12Vector.(curlcurlv_ref1, curlcurlv_ref2) rtol =
            4e-2
    end
end

@testset "weak curl-strong curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
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

        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()
        curlcurlv =
            Geometry.Cartesian12Vector.(
                wcurl.(Geometry.Covariant3Vector.(curl.(v))),
            )
        Spaces.weighted_dss!(curlcurlv)

        @test curlcurlv ≈
              Geometry.Cartesian12Vector.(curlcurlv_ref1, curlcurlv_ref2) rtol =
            4e-2
    end
end

@testset "weak curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
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

        @test curlv ≈ Geometry.Contravariant3Vector.(curlv_ref) rtol = 1e-2
    end
end

@testset "div" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.Cartesian12Vector.(
                sin.(coords.x1 .+ 2 .* coords.x2),
                cos.(3 .* coords.x1 .+ 2 .* coords.x2),
            )

        div = Operators.Divergence()
        divv = div.(v)
        Spaces.weighted_dss!(divv)
        divv_ref =
            cos.(coords.x1 .+ 2 .* coords.x2) .-
            2 .* sin.(3 .* coords.x1 .+ 2 .* coords.x2)

        @test divv ≈ divv_ref rtol = 1e-2
    end
end


@testset "weak div" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
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
end


@testset "annhilator property: curl-grad" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x1 .+ 2 .* coords.x2)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss!(gradf)

        curl = Operators.Curl()
        curlgradf = curl.(gradf)
        Spaces.weighted_dss!(curlgradf)

        @test norm(curlgradf) < 1e-12
    end
end

@testset "annhilator property: div-curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v = Geometry.Covariant3Vector.(sin.(coords.x1 .+ 2 .* coords.x2))
        curl = Operators.Curl()
        curlv = curl.(v)
        Spaces.weighted_dss!(curlv)

        div = Operators.Divergence()
        divcurlv = div.(curlv)
        Spaces.weighted_dss!(divcurlv)

        @test norm(divcurlv) < 1e-12
    end
end

@testset "scalar hyperdiffusion" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        k = 2
        l = 3
        y = @. sin(k * coords.x1 + l * coords.x2)
        ∇⁴y_ref = @. (k^2 + l^2)^2 * sin(k * coords.x1 + l * coords.x2)

        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        χ = Spaces.weighted_dss!(@. wdiv(grad(y)))
        ∇⁴y = Spaces.weighted_dss!(@. wdiv(grad(χ)))

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end
end

@testset "vector hyperdiffusion" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        k = 2
        l = 3
        y = @. Geometry.Cartesian12Vector(
            sin(k * coords.x1 + l * coords.x2),
            0.0,
        )
        ∇⁴y_ref = @. Geometry.Cartesian12Vector(
            (k^2 + l^2)^2 * sin(k * coords.x1 + l * coords.x2),
            0.0,
        )
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        sdiv = Operators.Divergence()
        wgrad = Operators.WeakGradient()

        χ = Spaces.weighted_dss!(
            @. Geometry.Cartesian12Vector(wgrad(sdiv(y))) -
               Geometry.Cartesian12Vector(
                wcurl(Geometry.Covariant3Vector(curl(y))),
            )
        )
        ∇⁴y = Spaces.weighted_dss!(
            @. Geometry.Cartesian12Vector(wgrad(sdiv(χ))) -
               Geometry.Cartesian12Vector(
                wcurl(Geometry.Covariant3Vector(curl(χ))),
            )
        )

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end
end
