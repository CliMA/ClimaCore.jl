using Test
using JET

using ClimaCore.DataLayouts
using StaticArrays
using ClimaCore.DataLayouts: get_struct, set_struct!

TestFloatTypes = (Float32, Float64)

@testset "VF" begin
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}
        array = rand(FT, 4, 3)

        data = VF{S}(array)
        @test getfield(data.:1, :array) == @view(array[:, 1:2])

        # test tuple assignment
        data[1] = (Complex{FT}(-1.0, -2.0), FT(-3.0))
        @test array[1, 1] == -1.0
        @test array[1, 2] == -2.0
        @test array[1, 3] == -3.0

        # sum of all the first field elements
        @test sum(data.:1) ≈ Complex{FT}(sum(array[:, 1]), sum(array[:, 2])) atol =
            10eps()

        @test sum(x -> x[2], data) ≈ sum(array[:, 3]) atol = 10eps()
    end
end

@testset "VF boundscheck" begin
    S = Tuple{Complex{Float64}, Float64}
    array = zeros(Float64, 4, 3)
    data = VF{S}(array)
    @test data[1][2] == zero(Float64)
    @test_throws BoundsError data[-1]
    @test_throws BoundsError data[5]
end

@testset "VF type safety" begin
    Nv = 1 # number of vertical levels

    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    array = zeros(Float64, Nv, 2)
    data = VF{typeof(SA)}(array)

    ret = begin
        data[1] = SA
    end
    @test ret === SA
    @test data[1] isa typeof(SA)
    @test_throws MethodError data[1] = SB
end

@testset "VF broadcasting between 1D data objects and scalars" begin
    for FT in TestFloatTypes
        data1 = ones(FT, 2, 2)
        S = Complex{FT}
        data1 = VF{S}(data1)
        res = data1 .+ 1
        @test res isa VF
        @test parent(res) == FT[2.0 1.0; 2.0 1.0]
        @test sum(res) == Complex{FT}(4.0, 2.0)
        @test sum(Base.Broadcast.broadcasted(+, data1, 1)) ==
              Complex{FT}(4.0, 2.0)
    end
end

@testset "VF broadcasting 1D assignment from scalar" begin
    for FT in TestFloatTypes
        S = Complex{FT}
        data = VF{S}(Array{FT}, 3)
        data .= Complex{FT}(1.0, 2.0)
        @test parent(data) == FT[1.0 2.0; 1.0 2.0; 1.0 2.0]
        data .= 1
        @test parent(data) == FT[1.0 0.0; 1.0 0.0; 1.0 0.0]
    end
end

@testset "VF broadcasting between 1D data objects" begin
    for FT in TestFloatTypes
        data1 = ones(FT, 2, 2)
        data2 = ones(FT, 2, 1)
        S1 = Complex{FT}
        S2 = FT
        data1 = VF{S1}(data1)
        data2 = VF{S2}(data2)
        res = data1 .+ data2
        @test res isa VF{S1}
        @test parent(res) == FT[2.0 1.0; 2.0 1.0]
        @test sum(res) == Complex{FT}(4.0, 2.0)
    end
end

# Test that Julia ia able to optimize VF DataLayouts v1.7+
@static if @isdefined(var"@test_opt")
    @testset "VF analyzer optimizations" begin
        for FT in TestFloatTypes
            S1 = NamedTuple{(:a, :b), Tuple{Complex{FT}, FT}}
            data1 = ones(FT, 2, 2)
            S2 = NamedTuple{(:c,), Tuple{FT}}
            data2 = ones(FT, 2, 1)

            dl1 = VF{S1}(data1)
            dl2 = VF{S2}(data2)

            f(a1, a2) = a1.a.re * a2.c + a1.b

            # property access
            @test_opt getproperty(data1, :a)
            # test map as proxy for broadcast
            @test_opt broadcast(f, dl1, dl2)
        end
    end
end
