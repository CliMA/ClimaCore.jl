using Test
using JET

using ClimaCore.DataLayouts
using StaticArrays
using ClimaCore.DataLayouts: get_struct, set_struct!

TestFloatTypes = (Float32, Float64)

@testset "DataF" begin
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}
        array = rand(FT, 1, 3)

        data = DataF{S}(array)
        @test getfield(data, :array) == @view(array[:, :])

        # test tuple assignment
        data[] = (Complex{FT}(-1.0, -2.0), FT(-3.0))
        @test array[1, 1] == -1.0
        @test array[1, 2] == -2.0
        @test array[1, 3] == -3.0

        # sum of all the first field elements
        @test sum(data.:1) ≈ Complex{FT}(sum(array[:, 1]), sum(array[:, 2])) atol =
            10eps()

        @test sum(x -> x[2], data) ≈ sum(array[:, 3]) atol = 10eps()
    end
end

@testset "DataF boundscheck" begin
    S = Tuple{Complex{Float64}, Float64}
    array = zeros(Float64, 1, 3)
    data = DataF{S}(array)
    @test data[][2] == zero(Float64)
    @test_throws MethodError data[1]
end

@testset "DataF type safety" begin
    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    array = zeros(Float64, 1, 2)
    data = DataF{typeof(SA)}(array)

    ret = begin
        data[] = SA
    end
    @test ret === SA
    @test data[] isa typeof(SA)
    @test_throws MethodError data[] = SB
end

@testset "DataF broadcasting between 0D data objects and scalars" begin
    for FT in TestFloatTypes
        data1 = ones(FT, 1, 2)
        S = Complex{FT}
        data1 = DataF{S}(data1)
        res = data1 .+ 1
        @test res isa DataF
        @test parent(res) == FT[2.0 1.0]
        @test sum(res) == Complex{FT}(2.0, 1.0)
        @test sum(Base.Broadcast.broadcasted(+, data1, 1)) ==
              Complex{FT}(2.0, 1.0)
    end
end

@testset "DataF broadcasting 0D assignment from scalar" begin
    for FT in TestFloatTypes
        S = Complex{FT}
        data = DataF{S}(Array{FT})
        data .= Complex{FT}(1.0, 2.0)
        @test parent(data) == FT[1.0 2.0]
        data .= 1
        @test parent(data) == FT[1.0 0.0]
    end
end

@testset "DataF broadcasting between 0D data objects" begin
    for FT in TestFloatTypes
        data1 = ones(FT, 1, 2)
        data2 = ones(FT, 1, 1)
        S1 = Complex{FT}
        S2 = FT
        data1 = DataF{S1}(data1)
        data2 = DataF{S2}(data2)
        res = data1 .+ data2
        @test res isa DataF{S1}
        @test parent(res) == FT[2.0 1.0]
        @test sum(res) == Complex{FT}(2.0, 1.0)
    end
end

@testset "broadcasting DataF + VF data object => VF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_vf = VF{S}(ones(FT, 3, 2))
    data_vf2 = data_f .+ data_vf
    @test data_vf2 isa VF{S}
    @test size(data_vf2) == (1, 1, 1, 3, 1)

end

@testset "broadcasting DataF + IF data object => IF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_if = IF{S, 2}(ones(FT, 2, 2))
    data_if2 = data_f .+ data_if
    @test data_if2 isa IF{S}
    @test size(data_if2) == (2, 1, 1, 1, 1)

end

@testset "broadcasting DataF + IFH data object => IFH" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_ifh = IFH{S, 2}(ones(FT, 2, 2, 3))
    data_ifh2 = data_f .+ data_ifh
    @test data_ifh2 isa IFH{S}
    @test size(data_ifh2) == (2, 1, 1, 1, 3)

end

@testset "broadcasting DataF + IJF data object => IJF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_ijf = IJF{S, 2}(ones(FT, 2, 2, 2))
    data_ijf2 = data_f .+ data_ijf
    @test data_ijf2 isa IJF{S}
    @test size(data_ijf2) == (2, 2, 1, 1, 1)

end

@testset "broadcasting DataF + IJFH data object => IJFH" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_ijfh = IJFH{S, 2}(ones(2, 2, 2, 3))
    data_ijfh2 = data_f .+ data_ijfh
    @test data_ijfh2 isa IJFH{S}
    @test size(data_ijfh2) == (2, 2, 1, 1, 3)

end

@testset "broadcasting DataF + VIFH data object => VIFH" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_vifh = VIFH{S, 4}(ones(FT, 10, 4, 3, 10))
    data_vifh2 = data_f .+ data_vifh
    @test data_vifh2 isa VIFH{S}
    @test size(data_vifh2) == (4, 1, 1, 10, 10)

end

@testset "broadcasting DataF + VIJFH data object => VIJFH" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 1, 2))
    data_vijfh = VIJFH{S, 2}(ones(FT, 2, 2, 2, 2, 2))
    data_vijfh2 = data_f .+ data_vijfh
    @test data_vijfh2 isa VIJFH{S}
    @test size(data_vijfh2) == (2, 2, 1, 2, 2)

end

# Test that Julia ia able to optimize DataF DataLayouts v1.7+
@static if @isdefined(var"@test_opt")
    @testset "DataF analyzer optimizations" begin
        for FT in TestFloatTypes
            S1 = NamedTuple{(:a, :b), Tuple{Complex{FT}, FT}}
            data1 = ones(FT, 1, 2)
            S2 = NamedTuple{(:c,), Tuple{FT}}
            data2 = ones(FT, 1, 1)

            dl1 = DataF{S1}(data1)
            dl2 = DataF{S2}(data2)

            f(a1, a2) = a1.a.re * a2.c + a1.b

            # property access
            @test_opt getproperty(data1, :a)
            # test map as proxy for broadcast
            @test_opt map(f, data1, data2)
            @test_opt mapreduce(f, +, data1, data2)
        end
    end
end
