using Test
using StaticArrays
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
using LinearAlgebra, IntervalSets

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

split_div = Operators.SplitDivergence()
div = Operators.Divergence()

function create_test_fields(space, FT)
    coords = Fields.coordinate_field(space)
    u = @. Geometry.UVVector(cos(coords.x), zero(FT))
    psi = @. FT(2) + sin(coords.x)
    return u, psi
end

function SpectralElem2D(FT, context, x2periodic)
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic,
        x1boundary = nothing,
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    quad = Quadratures.GLL{4}()
    mesh = Meshes.RectilinearMesh(domain, 4, 4)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, quad)
end

for FT in (Float32, Float64)
    @testset "split divergence (FT = $FT)" begin
        context = ClimaComms.context()

        @testset "fully periodic Cartesian domain" begin
            space = SpectralElem2D(FT, context, true)
            u, psi = create_test_fields(space, FT)

            u_const = Geometry.UVVector.(ones(FT, space), ones(FT, space))
            # Test psi = 1
            psi_const = FT(1)
            res = split_div.(u_const, psi_const)
            @test norm(res) < 100 * eps(FT)

            psi_const_field = ones(FT, space) .* FT(2)
            split_result_const = split_div.(u, psi_const_field)
            Spaces.weighted_dss!(split_result_const)

            div_result_const = div.(u .* psi_const_field)
            Spaces.weighted_dss!(div_result_const)

            @test norm(split_result_const .- div_result_const) < 100 * eps(FT)

            split_result = split_div.(u, psi)
            Spaces.weighted_dss!(split_result)

            div_result = div.(u .* psi)
            Spaces.weighted_dss!(div_result)

            # Test comparison - not sure what tolerance to use here
            max_val = max(norm(split_result), norm(div_result), one(FT))
            @test norm(split_result .- div_result) < FT(0.1) * max_val

            # Test conservation - integral of divergence over periodic domain should be zero
            integral = sum(split_result)
            @test abs(integral) < 100 * eps(FT)
        end

        @testset "non-periodic boundary behavior" begin
            space = SpectralElem2D(FT, context, false)
            u, psi = create_test_fields(space, FT)

            split_result = split_div.(u, psi)
            Spaces.weighted_dss!(split_result)

            div_result = div.(u .* psi)
            Spaces.weighted_dss!(div_result)

            @test_broken norm(split_result .- div_result) < 100 * eps(FT)
        end
    end
end
