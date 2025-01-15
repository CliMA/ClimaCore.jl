using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
@testset "Construct high resolution space on GPU" begin
    result = try
        ClimaCore.CommonSpaces.ExtrudedCubedSphereSpace(
            Float32;
            radius = 1.0,
            h_elem = 105,
            z_elem = 10,
            z_min = 1.0,
            z_max = 2.0,
            n_quad_points = 4,
            staggering = ClimaCore.Grids.CellCenter(),
        )
    catch
        println("unable to create center space")
        false
    end
    @test result isa ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace
end
