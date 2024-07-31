using Test
import ClimaCore.Utilities: half, PlusHalf


@testset "PlusHalf" begin
    @test half + 0 == half
    @test half < half + 1
    @test half <= half + 1
    @test !(half > half + 1)
    @test !(half >= half + 1)
    @test half != half + 1
    @test half + half == 1
    @test half - half == 0
    @test half + 3 == 3 + half == PlusHalf(3)
    @test min(half, half + 3) == half
    @test max(half, half + 3) == half + 3

    @test length(half:half) == 1
    @test length(half:(2 + half)) == 3
    @test collect(half:(2 + half)) == [half, 1 + half, 2 + half]

    @test_throws InexactError convert(Int, half)
    @test_throws InexactError convert(PlusHalf, 1)
    @test_throws InexactError convert(PlusHalf{Int}, 1)
end
