#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "data0d.jl"))
=#
using Test
using JET

using ClimaComms
using ClimaCore.DataLayouts
using StaticArrays
using ClimaCore.DataLayouts: get_struct, set_struct!

TestFloatTypes = (Float32, Float64)
device = ClimaComms.device()
ArrayType = ClimaComms.array_type(device)

@testset "DataF" begin
    for FT in TestFloatTypes
        S = Tuple{Complex{FT}, FT}

        data = DataF{S}(ArrayType{FT}, rand)
        array = parent(data)
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
    data = DataF{S}(ArrayType{Float64}, zeros)
    @test data[][2] == zero(Float64)
    @test_throws MethodError data[1]
end

@testset "DataF type safety" begin
    # check that types of the same bitstype throw a conversion error
    SA = (a = 1.0, b = 2.0)
    SB = (c = 1.0, d = 2.0)

    data = DataF{typeof(SA)}(ArrayType{Float64}, zeros)

    ret = begin
        data[] = SA
    end
    @test ret === SA
    @test data[] isa typeof(SA)
    @test_throws MethodError data[] = SB
end

@testset "DataF error messages" begin
    SA = (; a = 1.0)
    data = DataF{typeof(SA)}(ArrayType{Float64})
    @test_throws ErrorException(
        "Invalid field name `oops` for type `$(typeof(SA))`.",
    ) data.oops
end

@testset "DataF broadcasting between 0D data objects and scalars" begin
    for FT in TestFloatTypes
        S = Complex{FT}
        data1 = DataF{S}(ArrayType{FT}, ones)
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
        S1 = Complex{FT}
        S2 = FT
        data1 = DataF{S1}(ArrayType{FT}, ones)
        data2 = DataF{S2}(ArrayType{FT}, ones)
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
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_vf = VF{S}(ArrayType{FT}, ones; Nv)
    data_vf2 = data_f .+ data_vf
    @test data_vf2 isa VF{S, Nv}
    @test size(data_vf2) == (1, 1, 1, 3, 1)
end

@testset "broadcasting DataF + IF data object => IF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_if = IF{S}(ArrayType{FT}, ones; Ni = 2)
    data_if2 = data_f .+ data_if
    @test data_if2 isa IF{S}
    @test size(data_if2) == (2, 1, 1, 1, 1)
end

@testset "broadcasting DataF + IFH data object => IFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_ifh = IFH{S}(ArrayType{FT}, ones; Ni = 2, Nh)
    data_ifh2 = data_f .+ data_ifh
    @test data_ifh2 isa IFH{S}
    @test size(data_ifh2) == (2, 1, 1, 1, 3)
end

@testset "broadcasting DataF + IJF data object => IJF" begin
    FT = Float64
    S = Complex{FT}
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_ijf = IJF{S}(ArrayType{FT}, ones; Nij = 2)
    data_ijf2 = data_f .+ data_ijf
    @test data_ijf2 isa IJF{S}
    @test size(data_ijf2) == (2, 2, 1, 1, 1)
end

@testset "broadcasting DataF + IJFH data object => IJFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_ijfh = IJFH{S}(ArrayType{FT}, ones; Nij = 2, Nh)
    data_ijfh2 = data_f .+ data_ijfh
    @test data_ijfh2 isa IJFH{S}
    @test size(data_ijfh2) == (2, 2, 1, 1, Nh)
end

@testset "broadcasting DataF + VIFH data object => VIFH" begin
    FT = Float64
    S = Complex{FT}
    Nh = 10
    data_f = DataF{S}(ArrayType{FT}, ones)
    Nv = 10
    data_vifh = VIFH{S}(ArrayType{FT}, ones; Nv, Ni = 4, Nh)
    data_vifh2 = data_f .+ data_vifh
    @test data_vifh2 isa VIFH{S, Nv}
    @test size(data_vifh2) == (4, 1, 1, Nv, Nh)
end

@testset "broadcasting DataF + VIJFH data object => VIJFH" begin
    FT = Float64
    S = Complex{FT}
    Nv = 2
    Nh = 2
    data_f = DataF{S}(ArrayType{FT}, ones)
    data_vijfh = VIJFH{S}(ArrayType{FT}, ones; Nv, Nij = 2, Nh)
    data_vijfh2 = data_f .+ data_vijfh
    @test data_vijfh2 isa VIJFH{S, Nv}
    @test size(data_vijfh2) == (2, 2, 1, Nv, Nh)
end

@testset "column IF => DataF" begin
    FT = Float64
    S = Complex{FT}
    data_if = IF{S}(ArrayType{FT}; Ni = 2)
    array = parent(data_if)
    array .= FT[1 2; 3 4]
    if_column = column(data_if, 2)
    @test if_column isa DataF
    @test if_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_if, 3)
end

@testset "column IFH => DataF" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    data_ifh = IFH{S}(ArrayType{FT}; Ni = 2, Nh)
    array = parent(data_ifh)
    array[1, :, 1] .= FT[3, 4]
    ifh_column = column(data_ifh, 1, 1)
    @test ifh_column isa DataF
    @test ifh_column[] == 3.0 + 4.0im
    @test_throws BoundsError column(data_ifh, 3, 2)
    @test_throws BoundsError column(data_ifh, 2, 4)
end

@testset "column IJF => DataF" begin
    FT = Float64
    S = Complex{FT}
    data_ijf = IJF{S}(ArrayType{FT}; Nij = 2)
    array = parent(data_ijf)
    array[1, 1, :] .= FT[3, 4]
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
    data_ijfh = IJFH{S}(ArrayType{FT}; Nij = 2, Nh)
    array = parent(data_ijfh)
    array[1, 1, :, 2] .= FT[3, 4]
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
    Nv = 3
    data_vf = VF{S}(ArrayType{FT}; Nv)
    array = parent(data_vf)
    array .= FT[1 2; 3 4; 5 6]
    vf_level = level(data_vf, 2)
    @test vf_level isa DataF
    @test vf_level[] == 3.0 + 4.0im
    @test_throws BoundsError level(data_vf, 4)
end
