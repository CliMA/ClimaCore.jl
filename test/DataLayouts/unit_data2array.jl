#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_data2array.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaComms

function is_data2array2data_identity(data)
    all(
        parent(DataLayouts.array2data(DataLayouts.data2array(data), data)) .==
        parent(data),
    )
end

@testset "data2array & array2data" begin
    FT = Float64
    Nv = 10 # number of vertical levels
    Ni = Nij = 4  # number of nodal points
    Nh = 10 # number of elements
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)

    data = IF{FT}(ArrayType{FT}, rand; Ni)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    data = IFH{FT}(ArrayType{FT}, rand; Ni, Nh)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    data = IJF{FT}(ArrayType{FT}, rand; Nij)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    data = IJFH{FT}(ArrayType{FT}, rand; Nij, Nh)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    data = VIFH{FT}(ArrayType{FT}, rand; Nv, Ni, Nh)
    @test DataLayouts.data2array(data) == reshape(parent(data), Nv, :)
    @test is_data2array2data_identity(data)

    data = VIJFH{FT}(ArrayType{FT}, rand; Nv, Nij, Nh)
    @test DataLayouts.data2array(data) == reshape(parent(data), Nv, :)
    @test is_data2array2data_identity(data)
end
