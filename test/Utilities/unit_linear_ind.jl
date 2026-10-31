using Test
import ClimaCore.Utilities: linear_ind

@testset "linear_ind" begin
    n = (3, 4, 2)
    for ci in CartesianIndices(n)
        @test linear_ind(n, ci) == LinearIndices(n)[ci]
        @test linear_ind(n, Tuple(ci)) == LinearIndices(n)[ci]
    end
end
