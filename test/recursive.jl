using JET
using Test

using ClimaCore.RecursiveApply

@static if @isdefined(var"@test_opt") # v1.7 and higher
    @testset "RecursiveApply optimization test" begin
        for x in [
            1.0,
            1.0f0,
            (1.0, 2.0),
            (1.0f0, 2.0f0),
            (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
            (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
        ]
            @test_opt 2 ⊠ x
            @test_opt x ⊞ x
            @test_opt RecursiveApply.rdiv(x, 3)
        end
    end
end
