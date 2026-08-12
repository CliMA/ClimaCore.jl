using Test
using IntervalSets
import ClimaCore
import ClimaCore: Domains, Geometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# Direct unit tests for the Domains module.

@testset "Domains" begin
    TU.@test_precisions FT begin
        @testset "IntervalDomain [$FT]" begin
            zmin = Geometry.ZPoint{FT}(0)
            zmax = Geometry.ZPoint{FT}(1)

            walled = Domains.IntervalDomain(
                zmin,
                zmax;
                boundary_names = (:bottom, :top),
            )
            @test !Domains.isperiodic(walled)
            # boundary_names returns the unique names (as a collection)
            @test collect(Domains.boundary_names(walled)) == [:bottom, :top]
            @test walled.coord_min == zmin
            @test walled.coord_max == zmax

            periodic = Domains.IntervalDomain(zmin, zmax; periodic = true)
            @test Domains.isperiodic(periodic)
            @test isempty(Domains.boundary_names(periodic))

            # Either `periodic = true` or `boundary_names` is required.
            @test_throws ArgumentError Domains.IntervalDomain(zmin, zmax)

            # show() must not error
            @test !isempty(sprint(show, walled))
            @test !isempty(sprint(show, periodic))
        end

        @testset "RectangleDomain [$FT]" begin
            domain = Domains.RectangleDomain(
                Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(1),
                Geometry.YPoint{FT}(0) .. Geometry.YPoint{FT}(1);
                x1periodic = true,
                x2periodic = false,
                x2boundary = (:south, :north),
            )
            # Periodic x contributes no boundary names; walls in y do.
            @test Set(Domains.boundary_names(domain)) == Set([:south, :north])
            @test Domains.isperiodic(domain.interval1)
            @test !Domains.isperiodic(domain.interval2)
            @test !isempty(sprint(show, domain))
        end

        @testset "SphereDomain [$FT]" begin
            radius = FT(6.371e6)
            domain = Domains.SphereDomain(radius)
            @test domain.radius == radius
            # Spheres are boundary-free.
            @test isempty(Domains.boundary_names(domain))
            @test !isempty(sprint(show, domain))
        end
    end

    @testset "IntervalDomain coordinate promotion" begin
        # Mixed-precision endpoints promote to a common coordinate type.
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(0),
            Geometry.ZPoint(1.0);
            boundary_names = (:bottom, :top),
        )
        @test typeof(domain.coord_min) == typeof(domain.coord_max)
    end
end
