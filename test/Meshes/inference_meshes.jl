using Test
using SparseArrays
using JET

import ClimaCore: ClimaCore, Domains, Meshes, Geometry
@isdefined(JETConfig) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "JETConfig.jl"),
)

@testset "monotonic_check" begin
    faces = range(Geometry.ZPoint(0), Geometry.ZPoint(10); length = 11)
    cfaces = collect(faces)
    JETConfig.@test_with_jet Meshes.monotonic_check(faces)
    JETConfig.@test_with_jet Meshes.monotonic_check(cfaces)
    @test 0 == @allocated Meshes.monotonic_check(cfaces)
end
