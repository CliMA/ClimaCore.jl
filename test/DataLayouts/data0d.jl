#=
julia --project=test
using Revise; include(joinpath("test", "DataLayouts", "data0d.jl"))
=#
using Test
using JET

using ClimaCore.DataLayouts
using StaticArrays
using ClimaCore.DataLayouts: get_struct, set_struct!

TestFloatTypes = (Float32, Float64)

@testset "DataF" begin
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}
        array = rand(FT, 3)

        data = DataF{S}(array)
        @test getfield(data, :array) == array

        # test tuple assignment
        data[] = (Complex{FT}(-1.0, -2.0), FT(-3.0))
        @test array[1] == -1.0
        @test array[2] == -2.0
        @test array[3] == -3.0

        data2 = DataF(data[])
        @test typeof(data2) == typeof(data)
        @test parent(data2) == parent(data)

        # sum of all the first field elements
        @test data.:1[] == Complex{FT}(array[1], array[2])

        @test data.:2[] == array[3]

        data_copy = copy(data)
        @test data_copy isa DataF
        @test data_copy[] == data[]
    end
end

@testset "DataF boundscheck" begin
    S = Tuple{Complex{Float64}, Float64}
    array = zeros(Float64, 3)
    data = DataF{S}(array)
    @test data[][2] == zero(Float64)
    @test_throws MethodError data[1]
end

@testset "DataF type safety" begin
    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    array = zeros(Float64, 2)
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
        data1 = ones(FT, 2)
        S = Complex{FT}
        data1 = DataF{S}(data1)
        res = data1 .+ 1
        @test res isa DataF
        @test parent(res) == FT[2.0, 1.0]
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
        @test parent(data) == FT[1.0, 2.0]
        data .= 1
        @test parent(data) == FT[1.0, 0.0]
    end
end

@testset "DataF broadcasting between 0D data objects" begin
    for FT in TestFloatTypes
        data1 = ones(FT, 2)
        data2 = ones(FT, 1)
        S1 = Complex{FT}
        S2 = FT
        data1 = DataF{S1}(data1)
        data2 = DataF{S2}(data2)
        res = data1 .+ data2
        @test res isa DataF{S1}
        @test parent(res) == FT[2.0, 1.0]
        @test sum(res) == Complex{FT}(2.0, 1.0)
    end
end

@testset "broadcasting DataF + VF data object => VF" begin
    FT = Float64
    S = Complex{FT}
    Nv = 3
    data_f = DataF{S}(ones(FT, 2))
    data_vf = VF{S, Nv}(ones(FT, Nv, 2))
    data_vf2 = data_f .+ data_vf
    @test data_vf2 isa VF{S, Nv}
    @test size(data_vf2) == (1, 1, 1, 3, 1)
end

@testset "broadcasting DataF + IF data object => IF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 2))
    data_if = IF{S, 2}(ones(FT, 2, 2))
    data_if2 = data_f .+ data_if
    @test data_if2 isa IF{S}
    @test size(data_if2) == (2, 1, 1, 1, 1)
end

@testset "broadcasting DataF + IFH data object => IFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    data_f = DataF{S}(ones(FT, 2))
    data_ifh = IFH{S, 2}(ones(FT, 2, 2, Nh))
    data_ifh2 = data_f .+ data_ifh
    @test data_ifh2 isa IFH{S}
    @test size(data_ifh2) == (2, 1, 1, 1, 3)
end

@testset "broadcasting DataF + IJF data object => IJF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ones(FT, 2))
    data_ijf = IJF{S, 2}(ones(FT, 2, 2, 2))
    data_ijf2 = data_f .+ data_ijf
    @test data_ijf2 isa IJF{S}
    @test size(data_ijf2) == (2, 2, 1, 1, 1)
end

@testset "broadcasting DataF + IJFH data object => IJFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    data_f = DataF{S}(ones(FT, 2))
    data_ijfh = IJFH{S, 2}(ones(2, 2, 2, Nh))
    data_ijfh2 = data_f .+ data_ijfh
    @test data_ijfh2 isa IJFH{S}
    @test size(data_ijfh2) == (2, 2, 1, 1, Nh)
end

@testset "broadcasting DataF + VIFH data object => VIFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 10
    data_f = DataF{S}(ones(FT, 2))
    Nv = 10
    data_vifh = VIFH{S, Nv, 4}(ones(FT, Nv, 4, 2, Nh))
    data_vifh2 = data_f .+ data_vifh
    @test data_vifh2 isa VIFH{S, Nv}
    @test size(data_vifh2) == (4, 1, 1, Nv, Nh)
end

@testset "broadcasting DataF + VIJFH data object => VIJFH" begin
    FT = Float64
    S = Complex{FT}
    Nv = 2
    Nh = 2
    data_f = DataF{S}(ones(FT, 2))
    data_vijfh = VIJFH{S, Nv, 2}(ones(FT, Nv, 2, 2, 2, Nh))
    data_vijfh2 = data_f .+ data_vijfh
    @test data_vijfh2 isa VIJFH{S, Nv}
    @test size(data_vijfh2) == (2, 2, 1, Nv, Nh)
end

@testset "column IF => DataF" begin
    FT = Float64
    S = Complex{FT}
    array = FT[1 2; 3 4]
    data_if = IF{S, 2}(array)
    if_column = column(data_if, 2)
    @test if_column isa DataF
    @test if_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_if, 3)
end

@testset "column IFH => DataF" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    array = ones(FT, 2, 2, Nh)
    array[1, :, 1] .= FT[3, 4]
    data_ifh = IFH{S, 2}(array)
    ifh_column = column(data_ifh, 1, 1)
    @test ifh_column isa DataF
    @test ifh_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_ifh, 3, 2)
    @test_throws BoundsError column(data_ifh, 2, 4)
end

@testset "column IJF => DataF" begin
    FT = Float64
    S = Complex{FT}
    array = ones(FT, 2, 2, 2)
    array[1, 1, :] .= FT[3, 4]
    data_ijf = IJF{S, 2}(array)
    ijf_column = column(data_ijf, 1, 1)
    @test ijf_column isa DataF
    @test ijf_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_ijf, 3, 1)
    @test_throws BoundsError column(data_ijf, 1, 3)
end

@testset "column IJFH => DataF" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    array = ones(2, 2, 2, 3)
    array[1, 1, :, 2] .= FT[3, 4]
    data_ijfh = IJFH{S, 2}(array)
    ijfh_column = column(data_ijfh, 1, 1, 2)
    @test ijfh_column isa DataF
    @test ijfh_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_ijfh, 3, 1, 1)
    @test_throws BoundsError column(data_ijfh, 1, 3, 1)
    @test_throws BoundsError column(data_ijfh, 1, 1, 4)
end

@testset "level VF => DataF" begin
    FT = Float64
    S = Complex{FT}
    array = FT[1 2; 3 4; 5 6]
    Nv = size(array, 1)
    data_vf = VF{S, Nv}(array)
    vf_level = level(data_vf, 2)
    @test vf_level isa DataF
    @test vf_level[] == 3.0 + 4.0im
    @test_throws BoundsError level(data_vf, 4)
end
