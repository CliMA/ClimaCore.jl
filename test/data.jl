using Test
using ClimateMachineCore.DataLayouts

using ClimateMachineCore.DataLayouts: get_struct, set_struct!

@testset "get_struct / set_struct!" begin
    array = [1.0, 2.0, 3.0]
    S = Tuple{Complex{Float64}, Float64}
    @test get_struct(array, S) == (1.0 + 2.0im, 3.0)
    set_struct!(array, (4.0 + 2.0im, 6.0))
    @test array == [4.0, 2.0, 6.0]
    @test get_struct(array, S) == (4.0 + 2.0im, 6.0)
end


@testset "IJFH" begin
    array = rand(2, 2, 3, 2)
    S = Tuple{Complex{Float64}, Float64}
    data = IJFH{S}(array)
    @test getfield(data.:1, :array) == @view(array[:, :, 1:2, :])
    data_slab = slab(data, 1, 1, 1)
    @test data_slab[2, 1] ==
          (Complex(array[2, 1, 1, 1], array[2, 1, 2, 1]), array[2, 1, 3, 1])
    data_slab[2, 1] = (Complex(-1.0, -2.0), -3.0)
    @test array[2, 1, 1, 1] == -1.0
    @test array[2, 1, 2, 1] == -2.0
    @test array[2, 1, 3, 1] == -3.0

    subdata_slab = data_slab.:2
    @test subdata_slab[2, 1] == -3.0
    subdata_slab[2, 1] = -5.0
    @test array[2, 1, 3, 1] == -5.0
end
