using Test
using StaticArrays
using ClimaComms, ClimaCore
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets
using CUDA

FT = Float64
domain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
    x1periodic = true,
    x2periodic = true,
)

Nq = 5
quad = Spaces.Quadratures.GLL{Nq}()

grid_mesh = Meshes.RectilinearMesh(domain, 17, 16)


grid_topology_cpu =
    Topologies.Topology2D(ClimaComms.SingletonCommsContext(ClimaCore.Device.device(;disablegpu=true)), grid_mesh)
grid_space_cpu = Spaces.SpectralElementSpace2D(grid_topology_cpu, quad)
coords_cpu = Fields.coordinate_field(grid_space_cpu)

f_cpu = sin.(coords_cpu.x .+ 2 .* coords_cpu.y)

grad = Operators.Gradient()
gradf_cpu = grad.(f_cpu)


grid_topology =
    Topologies.Topology2D(ClimaComms.SingletonCommsContext(ClimaCore.Device.device()), grid_mesh)
grid_space = Spaces.SpectralElementSpace2D(grid_topology, quad)
coords = Fields.coordinate_field(grid_space)

CUDA.allowscalar(false)
f = sin.(coords.x .+ 2 .* coords.y)

grad = Operators.Gradient()
gradf = grad.(f)







#==

ts_mesh = Meshes.RectilinearMesh(domain, 17, 16)
ts_topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), ts_mesh)
ts_space = Spaces.SpectralElementSpace2D(ts_topology, quad)
ts_coords = Fields.coordinate_field(ts_space)

grid_test_setup = (grid_topology, grid_space, grid_coords)
ts_test_setup = (ts_topology, ts_space, ts_coords)

@testset "interpolate / restrict" begin

    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        INq = 9
        Iquad = Spaces.Quadratures.GLL{INq}()
        Ispace = Spaces.SpectralElementSpace2D(topology, Iquad)

        I = Operators.Interpolate(Ispace)
        R = Operators.Restrict(space)

        f = sin.(coords.x .+ 2 .* coords.y)

        interpolated_field = I.(f)
        Spaces.weighted_dss2!(interpolated_field)

        @test axes(interpolated_field).quadrature_style == Iquad
        @test axes(interpolated_field).topology == topology

        restrict_field = R.(f)
        Spaces.weighted_dss2!(restrict_field)

        @test axes(restrict_field).quadrature_style == quad
        @test axes(restrict_field).topology == topology

        interp_restrict_field = R.(I.(f))
        Spaces.weighted_dss2!(interp_restrict_field)

        @test axes(interp_restrict_field).quadrature_style == quad
        @test axes(interp_restrict_field).topology == topology

        @test norm(interp_restrict_field .- f) ≤ 3.0e-4
    end
end

@testset "gradient" begin

    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x .+ 2 .* coords.y)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss2!(gradf)

        @test gradf ≈
              Geometry.Covariant12Vector.(
            Geometry.UVVector.(
                cos.(coords.x .+ 2 .* coords.y),
                2 .* cos.(coords.x .+ 2 .* coords.y),
            ),
        ) rtol = 1e-2

        fv =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(coords.x .+ 2 .* coords.y),
            )
        gradfv = Geometry.transform.(Ref(Geometry.UVAxis()), grad.(fv))
        Spaces.weighted_dss2!(gradfv)
        @test eltype(gradfv) <: Geometry.Axis2Tensor
    end
end


@testset "weak gradient" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x .+ 2 .* coords.y)

        wgrad = Operators.WeakGradient()
        gradf = wgrad.(f)
        Spaces.weighted_dss2!(gradf)

        @test Geometry.UVVector.(gradf) ≈
              Geometry.UVVector.(
            cos.(coords.x .+ 2 .* coords.y),
            2 .* cos.(coords.x .+ 2 .* coords.y),
        ) rtol = 1e-2
    end
end

@testset "curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 4 .* coords.y),
            )

        curl = Operators.Curl()
        curlv = curl.(Geometry.Covariant12Vector.(v))
        Spaces.weighted_dss2!(curlv)
        curlv_ref =
            Geometry.Contravariant3Vector.(
                .-3 .* sin.(3 .* coords.x .+ 4 .* coords.y) .-
                2 .* cos.(coords.x .+ 2 .* coords.y),
            )

        @test curlv ≈ curlv_ref rtol = 1e-2
    end
end

@testset "curl-curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )
        curlv_ref =
            .-3 .* sin.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* cos.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref1 =
            .-6 .* cos.(3 .* coords.x .+ 2 .* coords.y) .+
            4 .* sin.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref2 =
            9 .* cos.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* sin.(coords.x .+ 2 .* coords.y)

        curl = Operators.Curl()
        curlcurlv =
            curl.(
                Geometry.Covariant3Vector.(
                    curl.(Geometry.Covariant12Vector.(v)),
                ),
            )
        Spaces.weighted_dss2!(curlcurlv)

        @test Geometry.UVVector.(curlcurlv) ≈
              Geometry.UVVector.(curlcurlv_ref1, curlcurlv_ref2) rtol = 4e-2
    end
end

@testset "weak curl-strong curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )
        curlv_ref =
            .-3 .* sin.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* cos.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref1 =
            .-6 .* cos.(3 .* coords.x .+ 2 .* coords.y) .+
            4 .* sin.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref2 =
            9 .* cos.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* sin.(coords.x .+ 2 .* coords.y)

        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()
        curlcurlv =
            Geometry.UVVector.(
                wcurl.(
                    Geometry.Covariant3Vector.(
                        curl.(Geometry.Covariant12Vector.(v)),
                    ),
                ),
            )
        Spaces.weighted_dss2!(curlcurlv)

        @test curlcurlv ≈ Geometry.UVVector.(curlcurlv_ref1, curlcurlv_ref2) rtol =
            4e-2
    end
end

@testset "weak curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 4 .* coords.y),
            )

        wcurl = Operators.WeakCurl()
        curlv = wcurl.(Geometry.Covariant12Vector.(v))
        Spaces.weighted_dss2!(curlv)
        curlv_ref =
            .-3 .* sin.(3 .* coords.x .+ 4 .* coords.y) .-
            2 .* cos.(coords.x .+ 2 .* coords.y)

        @test curlv ≈ Geometry.Contravariant3Vector.(curlv_ref) rtol = 1e-2
    end
end

@testset "div" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )

        div = Operators.Divergence()
        divv = div.(v)
        Spaces.weighted_dss2!(divv)
        divv_ref =
            cos.(coords.x .+ 2 .* coords.y) .-
            2 .* sin.(3 .* coords.x .+ 2 .* coords.y)

        @test divv ≈ divv_ref rtol = 1e-2
    end
end


@testset "weak div" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )

        wdiv = Operators.WeakDivergence()
        divv = wdiv.(v)
        Spaces.weighted_dss2!(divv)
        divv_ref =
            cos.(coords.x .+ 2 .* coords.y) .-
            2 .* sin.(3 .* coords.x .+ 2 .* coords.y)

        @test divv ≈ divv_ref rtol = 1e-2
    end
end


@testset "annhilator property: curl-grad" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        f = sin.(coords.x .+ 2 .* coords.y)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss2!(gradf)

        curl = Operators.Curl()
        curlgradf = curl.(gradf)
        Spaces.weighted_dss2!(curlgradf)

        @test norm(curlgradf) < 1e-12
    end
end

@testset "annhilator property: div-curl" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        v = Geometry.Covariant3Vector.(sin.(coords.x .+ 2 .* coords.y))
        curl = Operators.Curl()
        curlv = curl.(v)
        Spaces.weighted_dss2!(curlv)

        div = Operators.Divergence()
        divcurlv = div.(curlv)
        Spaces.weighted_dss2!(divcurlv)

        @test norm(divcurlv) < 1e-12
    end
end

@testset "scalar hyperdiffusion" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        k = 2
        l = 3
        y = @. sin(k * coords.x + l * coords.y)
        ∇⁴y_ref = @. (k^2 + l^2)^2 * sin(k * coords.x + l * coords.y)

        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        χ = Spaces.weighted_dss2!(@. wdiv(grad(y)))
        ∇⁴y = Spaces.weighted_dss2!(@. wdiv(grad(χ)))

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end
end

@testset "vector hyperdiffusion" begin
    for (topology, space, coords) in (grid_test_setup, ts_test_setup)
        k = 2
        l = 3
        y = @. Geometry.UVVector(sin(k * coords.x + l * coords.y), 0.0)
        ∇⁴y_ref = @. Geometry.UVVector(
            (k^2 + l^2)^2 * sin(k * coords.x + l * coords.y),
            0.0,
        )
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        sdiv = Operators.Divergence()
        wgrad = Operators.WeakGradient()

        χ = Spaces.weighted_dss2!(
            @. Geometry.UVVector(wgrad(sdiv(y))) - Geometry.UVVector(
                wcurl(
                    Geometry.Covariant3Vector(
                        curl(Geometry.Covariant12Vector(y)),
                    ),
                ),
            )
        )
        ∇⁴y = Spaces.weighted_dss2!(
            @. Geometry.UVVector(wgrad(sdiv(χ))) - Geometry.UVVector(
                wcurl(
                    Geometry.Covariant3Vector(
                        curl(Geometry.Covariant12Vector(χ)),
                    ),
                ),
            )
        )

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end
end
==#