using Test
using StaticArrays

import ClimaCore.DataLayouts: VIFH, slab, column

@testset "VIFH" begin
    TestFloatTypes = (Float32, Float64)
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}
        Nv = 10 # number of vertical levels
        Ni = 4  # number of nodal points
        Ne = 10 # number of elements

        # construct a data object with 10 cells in vertical and
        # 10 elements in horizontal with 4 nodal points per element in horizontal
        array = rand(FT, Nv, Ni, 3, Ne)

        data = VIFH{S, Ni}(array)
        sum(x -> x[2], data)

        @test getfield(data.:1, :array) == @view(array[:, :, 1:2, :])
        @test getfield(data.:2, :array) == @view(array[:, :, 3:3, :])

        @test size(data) == (Ni, 1, 1, Nv, Ne)

        # test tuple assignment on columns
        val = (Complex{FT}(-1.0, -2.0), FT(-3.0))
        column(data, 1, 1)[1] = val
        @test array[1, 1, 1, 1] == -1.0
        @test array[1, 1, 2, 1] == -2.0
        @test array[1, 1, 3, 1] == -3.0

        # test value of assing tuple on slab
        sdata = slab(data, 1, 1)
        @test sdata[1] == val

        # sum of all the first field elements
        @test sum(data.:1) ≈
              Complex{FT}(sum(array[:, :, 1, :]), sum(array[:, :, 2, :]))
        @test sum(x -> x[2], data) ≈ sum(array[:, :, 3, :])
    end
end

@testset "broadcasting between VIFH data object + scalars" begin
    FT = Float64
    data1 = ones(FT, 2, 2, 2, 2)
    S = Complex{Float64}
    data1 = VIFH{S, 2}(data1)
    res = data1 .+ 1
    @test res isa VIFH{S}
    @test parent(res) ==
          FT[f == 1 ? 2 : 1 for i in 1:2, j in 1:2, f in 1:2, h in 1:2]
    @test sum(res) == Complex(16.0, 8.0)
    @test sum(Base.Broadcast.broadcasted(+, data1, 1)) == Complex(16.0, 8.0)
end

@testset "broadcasting between VF + IFH data object => VIFH" begin
    FT = Float64
    S = Complex{FT}
    data_vf = VF{S}(ones(FT, 3, 2))
    data_ifh = IFH{FT, 2}(ones(FT, 2, 1, 2))
    data_vifh = data_vf .+ data_ifh
    @test data_vifh isa VIFH{S}
    @test size(data_vifh) == (2, 1, 1, 3, 2)
    @test parent(data_vifh) ==
          FT[f == 1 ? 2 : 1 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]

    @test parent(data_vifh .+ data_vf) ==
          FT[f == 1 ? 3 : 2 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]
    @test parent(data_vifh .+ data_ifh) ==
          FT[f == 1 ? 3 : 1 for v in 1:3, i in 1:2, f in 1:2, h in 1:2]

end
