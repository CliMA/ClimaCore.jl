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
wdiv = Operators.WeakDivergence()

function create_test_fields(space, FT)
    coords = Fields.coordinate_field(space)
    # Generic test fields
    if space isa Spaces.SpectralElementSpace2D
        # Trigonometric functions in Cartesian and spherical domains
        if eltype(coords) <: Geometry.LatLongPoint
            lon = coords.long
            lat = coords.lat
            u = @. Geometry.UVVector(cosd(lon), sind(lat))
            psi = @. FT(2) + sind(lon) * cosd(lat)
        else
            x = coords.x
            y = coords.y
            u = @. Geometry.UVVector(cos(x) * sin(y), sin(x) * cos(y))
            psi = @. FT(2) + sin(x) * cos(y)
        end
    else
        error("Unsupported space")
    end
    return u, psi
end

function SpectralElem2D(FT, context, Ne, Nq)
    # Doubly periodic Cartesian domain
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic = true,
        x1boundary = nothing,
        x2boundary = nothing,
    )
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.RectilinearMesh(domain, Ne, Ne)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function SphereSpace(FT, context, Ne, Nq)
    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, quad)
end

for FT in (Float32, Float64)
    @testset "split divergence (FT = $FT)" begin
        context = ClimaComms.context()
        Nq = 4   # Nq is the number of quadrature points; polynomial order is Nq-1

        # Helper to compute discretization-based tolerance
        function heuristic_tol(Ne, Nq)
            h = 1 / Ne
            order = Nq - 1
            # Safety factor 10
            return 10 * h^order
        end

        @testset "fully periodic Cartesian domain" begin
            Ne = 9
            space = SpectralElem2D(FT, context, Ne, Nq)
            u, psi = create_test_fields(space, FT)

            u_const = Geometry.UVVector.(ones(FT, space), ones(FT, space))
            # Test psi = 1
            psi_const = FT(1)
            # Test with constant u (divergence is zero)
            res = split_div.(u_const, psi_const)
            @test norm(res) < 100 * eps(FT)

            # Test with varying u 
            res_var = split_div.(u, psi_const)
            wdiv_res_var = wdiv.(u .* psi_const)

            psi_const_field = ones(FT, space) .* FT(2)
            split_result_const = split_div.(u, psi_const_field)
            Spaces.weighted_dss!(split_result_const)

            wdiv_result_const = wdiv.(u .* psi_const_field)
            Spaces.weighted_dss!(wdiv_result_const)

            @test norm(split_result_const .- wdiv_result_const) < 100 * eps(FT)

            split_result = split_div.(u, psi)
            Spaces.weighted_dss!(split_result)

            wdiv_result = wdiv.(u .* psi)
            Spaces.weighted_dss!(wdiv_result)

            # Test comparison
            tol = heuristic_tol(Ne, Nq)
            # For Cartesian grid with constant metric, error might be smaller (zero?), 
            # but using robust tolerance is safer.
            @test norm(split_result .- wdiv_result) < max(tol, sqrt(eps(FT)))

            # Test conservation - integral of divergence over periodic domain should be zero
            integral = sum(split_result)
            @test abs(integral) < 30 * eps(FT)
        end

        @testset "sphere domain" begin
            Ne = 6
            space = SphereSpace(FT, context, Ne, Nq)
            u, psi = create_test_fields(space, FT)
            coords = Fields.coordinate_field(space)

            tol = heuristic_tol(Ne, Nq)

            # 1. Test consistency with standard divergence for a generic field
            split_result = split_div.(u, psi)
            Spaces.weighted_dss!(split_result)

            wdiv_result = wdiv.(u .* psi)
            Spaces.weighted_dss!(wdiv_result)

            @test norm(split_result .- wdiv_result) < tol

            # Test conservation
            integral = sum(split_result)
            @test abs(integral) < 100 * eps(FT)

            # 2. Test "divergence of a constant is 0"
            # Solid body rotation (divergence-free flow) with constant psi should yield ~0 divergence
            # We test the convergence of this error as resolution increases.
            Nes = [6, 12]
            errs = zeros(FT, length(Nes))
            for (i, Ne) in enumerate(Nes)
                space_conv = SphereSpace(FT, context, Ne, Nq)
                coords_conv = Fields.coordinate_field(space_conv)
                u_solid_body = @. Geometry.UVVector(cosd(coords_conv.lat), FT(0))

                # Using psi = 1 (constant)
                res_solid = split_div.(u_solid_body, FT(1))
                Spaces.weighted_dss!(res_solid)
                errs[i] = norm(res_solid)
                @test errs[i] < heuristic_tol(Ne, Nq)
            end

            # Check convergence rate
            # err ~ h^(Nq-1) => err1/err2 = (h1/h2)^(Nq-1) = (Ne2/Ne1)^(Nq-1)
            # rate = log(err1/err2) / log(Ne2/Ne1)
            rate = log(errs[1] / errs[2]) / log(Nes[2] / Nes[1])
            @test rate ≈ (Nq - 1) atol = 0.5
        end
    end
end
