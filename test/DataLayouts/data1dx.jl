#=
julia --project=test
using Revise; include(joinpath("test", "DataLayouts", "data1dx.jl"))
=#
using Test
using ClimaComms
using ClimaCore.DataLayouts
import ClimaCore.DataLayouts: VIFH, slab, column, VF, IFH, vindex, slab_index

device = ClimaComms.device()
ArrayType = ClimaComms.array_type(device)
@testset "VIFH" begin
    TestFloatTypes = (Float32, Float64)
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}
        Nv = 10 # number of vertical levels
        Ni = 4  # number of nodal points
        Nh = 10 # number of elements

        # construct a data object with 10 cells in vertical and
        # 10 elements in horizontal with 4 nodal points per element in horizontal

        data = VIFH{S}(ArrayType{FT}, rand; Nv, Ni, Nh)
        array = parent(data)
        sum(x -> x[2], data)

        @test getfield(data.:1, :array) == @view(array[:, :, 1:2, :])
        @test getfield(data.:2, :array) == @view(array[:, :, 3:3, :])

        @test size(data) == (Ni, 1, 1, Nv, Nh)

        # test tuple assignment on columns
        val = (Complex{FT}(-1.0, -2.0), FT(-3.0))
        column(data, 1, 1)[vindex(1)] = val
        @test array[1, 1, 1, 1] == -1.0
        @test array[1, 1, 2, 1] == -2.0
        @test array[1, 1, 3, 1] == -3.0

        # test value of assing tuple on slab
        sdata = slab(data, 1, 1)
        @test sdata[slab_index(1)] == val

        # sum of all the first field elements
        @test sum(data.:1) ≈
              Complex{FT}(sum(array[:, :, 1, :]), sum(array[:, :, 2, :]))
        @test sum(x -> x[2], data) ≈ sum(array[:, :, 3, :])
    end
end

@testset "VIFH boundscheck" begin
    Nv = 1 # number of vertical levels
    Ni = 1  # number of nodal points
    Nh = 2 # number of elements

    S = Tuple{Complex{Float64}, Float64}
    data = VIFH{S}(ArrayType{Float64}, zeros; Nv, Ni, Nh)

    @test_throws BoundsError slab(data, -1, -1)
    @test_throws BoundsError slab(data, 1, 3)

    sdata = slab(data, 1, 1)
    @test_throws BoundsError sdata[slab_index(-1)]
    @test_throws BoundsError sdata[slab_index(2)]

    @test_throws BoundsError column(data, -1, 1)
    @test_throws BoundsError column(data, -1, 1, 1)
    @test_throws BoundsError column(data, 2, 1)
    @test_throws BoundsError column(data, 1, 3)
end


@testset "VIFH type safety" begin
    Nv = 1 # number of vertical levels
    Ni = 1  # number of nodal points per element
    Nh = 1 # number of elements

    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    data = VIFH{typeof(SA)}(ArrayType{Float64}, zeros; Nv, Ni, Nh)

    cdata = column(data, 1, 1)
    cdata[slab_index(1)] = SA
    @test cdata[slab_index(1)] isa typeof(SA)
    @test_throws MethodError cdata[slab_index(1)] = SB

    sdata = slab(data, 1, 1)
    @test sdata[slab_index(1)] isa typeof(SA)
    @test_throws MethodError sdata[slab_index(1)] = SB
end

@testset "broadcasting between VIFH data object + scalars" begin
    FT = Float64
    Nv = 2
    Nh = 2
    S = Complex{Float64}
    data1 = VIFH{S}(ArrayType{FT}, ones; Nv, Ni = 2, Nh = 2)
    res = data1 .+ 1
    @test res isa VIFH{S, Nv}
    @test parent(res) ==
          FT[f == 1 ? 2 : 1 for i in 1:2, j in 1:2, f in 1:2, h in 1:2]
    @test sum(res) == Complex(16.0, 8.0)
    @test sum(Base.Broadcast.broadcasted(+, data1, 1)) == Complex(16.0, 8.0)
end

@testset "broadcasting between VF + IFH data object => VIFH" begin
    FT = Float64
    S = Complex{FT}
    Nv = 3
    Nh = 2
    data_vf = VF{S}(ArrayType{FT}, ones; Nv)
    data_ifh = IFH{FT}(ArrayType{FT}, ones; Ni = 2, Nh = 2)
    data_vifh = data_vf .+ data_ifh
    @test data_vifh isa VIFH{S, Nv}
    @test size(data_vifh) == (2, 1, 1, 3, 2)
    @test parent(data_vifh) ==
          FT[f == 1 ? 2 : 1 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]

    @test parent(data_vifh .+ data_vf) ==
          FT[f == 1 ? 3 : 2 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]
    @test parent(data_vifh .+ data_ifh) ==
          FT[f == 1 ? 3 : 1 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]

end

@testset "fill" begin
    data = IFH{Float64}(ArrayType{Float64}, ones; Ni = 3, Nh = 3)
    data .= 2.0
    @test all(==(2.0), parent(data))
end
