import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields
using Test

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

@testset "Sum function over field" begin
    FT = Float32
    context = ClimaComms.context()

    space = TU.SpectralElementSpace2D(FT; context)
    field = Fields.zeros(space)
    f = x -> x + 1

    # Sum of a function applied to a field should be the same as the sum of the
    # function applied to the parent field, but it isn't
    @test sum(f, field) != sum(f, parent(field))
    @test sum(f.(field)) != sum(f.(parent(field)))
end
