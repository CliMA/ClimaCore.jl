#=
julia --project=test
using Revise; include(joinpath("test", "DataLayouts", "data2dx.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaCore.DataLayouts: VF, IJFH, VIJFH, slab, column, slab_index, vindex

@testset "VIJFH" begin
    Nv = 10 # number of vertical levels
    Nij = 4 # Nij × Nij nodal points per element
    Nh = 10 # number of elements

    TestFloatTypes = (Float32, Float64)
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}

        # construct a data object with 10 cells in vertical and
        # 10 elements in horizontal with 4 × 4 nodal points per element in horizontal
        array = rand(FT, Nv, Nij, Nij, 3, Nh)

        data = VIJFH{S, Nv, Nij, Nh}(array)

        @test getfield(data.:1, :array) == @view(array[:, :, :, 1:2, :])
        @test getfield(data.:2, :array) == @view(array[:, :, :, 3:3, :])

        @test size(data) == (Nij, Nij, 1, Nv, Nh)

        # test tuple assignment on columns
        val = (Complex{FT}(-1.0, -2.0), FT(-3.0))

        column(data, 1, 2, 1)[vindex(1)] = val
        @test array[1, 1, 2, 1, 1] == -1.0
        @test array[1, 1, 2, 2, 1] == -2.0
        @test array[1, 1, 2, 3, 1] == -3.0

        # test value of assing tuple on slab
        sdata = slab(data, 1, 1)
        @test sdata[slab_index(1, 2)] == val

        # sum of all the first field elements
        @test sum(data.:1) ≈
              Complex{FT}(sum(array[:, :, :, 1, :]), sum(array[:, :, :, 2, :]))
        @test sum(x -> x[2], data) ≈ sum(array[:, :, :, 3, :])
    end
end

@testset "VIJFH boundscheck" begin
    Nv = 1 # number of vertical levels
    Nij = 1  # number of nodal points
    Nh = 2 # number of elements

    S = Tuple{Complex{Float64}, Float64}
    array = zeros(Float64, Nv, Nij, Nij, 3, Nh)
    data = VIJFH{S, Nv, Nij, Nh}(array)

    @test_throws BoundsError slab(data, -1, 1)
    @test_throws BoundsError slab(data, 1, -1)
    @test_throws BoundsError slab(data, 3, 1)
    @test_throws BoundsError slab(data, 1, 3)

    @test_throws BoundsError column(data, -1, 1, 1)
    @test_throws BoundsError column(data, 1, -1, 1)
    @test_throws BoundsError column(data, 1, 1, -1)
    @test_throws BoundsError column(data, 3, 1, 1)
    @test_throws BoundsError column(data, 1, 3, 1)
    @test_throws BoundsError column(data, 1, 1, 3)
end

@testset "VIJFH type safety" begin
    Nv = 1 # number of vertical levels
    Nij = 2 # Nij × Nij nodal points per element
    Nh = 1 # number of elements

    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    array = zeros(Float64, Nv, Nij, Nij, 2, Nh)
    data = VIJFH{typeof(SA), Nv, Nij, Nh}(array)

    cdata = column(data, 1, 2, 1)
    cdata[vindex(1)] = SA
    @test cdata[vindex(1)] isa typeof(SA)
    @test_throws MethodError cdata[vindex(1)] = SB

    sdata = slab(data, 1, 1)
    @test sdata[slab_index(1, 2)] isa typeof(SA)
    @test_throws MethodError sdata[slab_index(1)] = SB
end

@testset "broadcasting between VIJFH data object + scalars" begin
    FT = Float64
    array = ones(FT, 2, 2, 2, 2, 2)
    Nv = size(array, 1)
    Nh = size(array, 5)
    S = Complex{Float64}
    data1 = VIJFH{S, Nv, 2, Nh}(array)
    res = data1 .+ 1
    @test res isa VIJFH{S, Nv}
    @test parent(res) == FT[
        f == 1 ? 2 : 1 for v in 1:2, i in 1:2, j in 1:2, f in 1:2, h in 1:2
    ]
    @test sum(res) == Complex(FT(32.0), FT(16.0))
    @test sum(Base.Broadcast.broadcasted(+, data1, 1)) ==
          Complex(FT(32.0), FT(16.0))
end

@testset "broadcasting between VF + IJFH data object => VIJFH" begin
    FT = Float64
    S = Complex{FT}
    Nv = 3
    Nh = 2
    data_vf = VF{S, Nv}(ones(FT, Nv, 2))
    data_ijfh = IJFH{FT, 2, Nh}(ones(FT, 2, 2, 1, Nh))
    data_vijfh = data_vf .+ data_ijfh
    @test data_vijfh isa VIJFH{S, Nv}
    @test size(data_vijfh) == (2, 2, 1, 3, 2)

    @test parent(data_vijfh) == FT[
        f == 1 ? 2 : 1 for v in 1:3, i in 1:2, j in 1:2, f in 1:2, h in 1:2
    ]
    @test parent(data_vijfh .+ data_vf) == FT[
        f == 1 ? 3 : 2 for v in 1:3, i in 1:2, j in 1:2, f in 1:2, h in 1:2
    ]
    @test parent(data_vijfh .+ data_ijfh) == FT[
        f == 1 ? 3 : 1 for v in 1:3, i in 1:2, j in 1:2, f in 1:2, h in 1:2
    ]
end
