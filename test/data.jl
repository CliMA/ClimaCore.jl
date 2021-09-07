using Test
using ClimaCore.DataLayouts
using StaticArrays
using ClimaCore.DataLayouts: get_struct, set_struct!, basetype

@testset "basetype" begin
    @test basetype(Float64) === Float64
    @test basetype(Float32) === Float32
    @test basetype(Complex{Float64}) === Float64
    @test basetype(Complex{Float32}) === Float32

    @test basetype(Float64, Complex{Float64}) === Float64
    @test basetype(Tuple{Float64, Complex{Float64}}) === Float64
    @test basetype(typeof((a = 1.0, b = (2.0, 3.0, 4.0)))) === Float64
    @test basetype(typeof(SA[1.0 2.0; 3.0 4.0])) === Float64

    @test_throws Exception basetype(Int)
    @test_throws Exception basetype(Tuple{Float32, Float64})
    @test_throws Exception basetype(typeof((a = 1, b = (2.0, 3.0))))
end

@testset "get_struct / set_struct!" begin
    array = [1.0, 2.0, 3.0]
    S = Tuple{Complex{Float64}, Float64}
    @test get_struct(array, S) == (1.0 + 2.0im, 3.0)
    set_struct!(array, (4.0 + 2.0im, 6.0))
    @test array == [4.0, 2.0, 6.0]
    @test get_struct(array, S) == (4.0 + 2.0im, 6.0)
end

@testset "IJFH" begin
    Nij = 2
    S = Tuple{Complex{Float64}, Float64}
    array = rand(Nij, Nij, 3, 2)
    data = IJFH{S, 2}(array)
    @test getfield(data.:1, :array) == @view(array[:, :, 1:2, :])
    data_slab = slab(data, 1)
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

    @test sum(data.:1) ≈ Complex(sum(array[:, :, 1, :]), sum(array[:, :, 2, :])) atol =
        10eps()
    @test sum(x -> x[2], data) ≈ sum(array[:, :, 3, :]) atol = 10eps()
end


@testset "IJFH type safety" begin
    Nij = 2 # number of nodal points per element
    Nh = 1 # number of elements

    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    array = zeros(Float64, Nij, Nij, 2, Nh)
    data = IJFH{typeof(SA), Nij}(array)
    data_slab = slab(data, 1)
    data_slab[1, 1] = SA
    @test data_slab[1, 1] isa typeof(SA)
    @test_throws MethodError data_slab[1, 1] = SB
end

@testset "broadcasting between data object + scalars" begin
    FT = Float64
    data1 = ones(FT, 2, 2, 2, 2)
    S = Complex{Float64}
    data1 = IJFH{S, 2}(data1)
    res = data1 .+ 1
    @test res isa IJFH{S}
    @test parent(res) ==
          FT[f == 1 ? 2 : 1 for i in 1:2, j in 1:2, f in 1:2, h in 1:2]

    @test sum(res) == Complex(16.0, 8.0)
    @test sum(Base.Broadcast.broadcasted(+, data1, 1)) == Complex(16.0, 8.0)
end

@testset "broadcasting assignment from scalar" begin
    FT = Float64
    S = Complex{FT}
    data = IJFH{S, 2}(Array{FT}, 3)
    data .= Complex(1.0, 2.0)
    @test parent(data) ==
          FT[f == 1 ? 1 : 2 for i in 1:2, j in 1:2, f in 1:2, h in 1:3]

    data .= 1
    @test parent(data) ==
          FT[f == 1 ? 1 : 0 for i in 1:2, j in 1:2, f in 1:2, h in 1:3]

end

@testset "broadcasting between data objects" begin
    FT = Float64
    data1 = ones(FT, 2, 2, 2, 2)
    data2 = ones(FT, 2, 2, 1, 2)
    S1 = Complex{Float64}
    S2 = Float64
    data1 = IJFH{S1, 2}(data1)
    data2 = IJFH{S2, 2}(data2)
    res = data1 .+ data2
    @test res isa IJFH{S1}
    @test parent(res) ==
          FT[f == 1 ? 2 : 1 for i in 1:2, j in 1:2, f in 1:2, h in 1:2]

    @test sum(res) == Complex(16.0, 8.0)
    @test sum(Base.Broadcast.broadcasted(+, data1, data2)) == Complex(16.0, 8.0)
end

@testset "broadcasting complicated function" begin
    FT = Float64
    S1 = NamedTuple{(:a, :b), Tuple{Complex{Float64}, Float64}}
    data1 = ones(FT, 2, 2, 3, 2)
    S2 = Float64
    data2 = ones(FT, 2, 2, 1, 2)
    data1 = IJFH{S1, 2}(data1)
    data2 = IJFH{S2, 2}(data2)

    f(a1, a2) = a1.a.re * a2 + a1.b
    res = f.(data1, data2)
    @test res isa IJFH{Float64}
    @test parent(res) == FT[2 for i in 1:2, j in 1:2, f in 1:1, h in 1:2]
end
