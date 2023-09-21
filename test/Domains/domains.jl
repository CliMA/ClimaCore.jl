import ClimaCore: Domains

using Test

@testset "Domain" begin
    # Check that we can build a SphereDomain with all the reasonable types
    for T in (Float64, Float32, Int)
        radius = convert(T, 1)

        domain = Domains.SphereDomain(radius)

        @test typeof(domain) <: Domains.AbstractDomain
        @test domain.radius == radius
    end
end
