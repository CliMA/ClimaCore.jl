#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_data2array.jl"))
=#
using Test
using ClimaCore.DataLayouts

function is_data2array2data_identity(data)
    all(
        parent(DataLayouts.array2data(DataLayouts.data2array(data), data)) .==
        parent(data),
    )
end

@testset "data2array & array2data" begin
    FT = Float64
    Nv = 10 # number of vertical levels
    Nij = 4  # number of nodal points
    Nh = 10 # number of elements

    array = rand(FT, 2, 1)
    data = IF{FT, 2}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    array = rand(FT, 2, 1, Nh)
    data = IFH{FT, 2, Nh}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    array = rand(FT, 2, 2, 1)
    data = IJF{FT, 2}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    array = rand(FT, Nij, Nij, 1, Nh)
    data = IJFH{FT, Nij, Nh}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), :)
    @test is_data2array2data_identity(data)

    array = rand(FT, Nv, Nij, 1, Nh)
    data = VIFH{FT, Nv, Nij, Nh}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), Nv, :)
    @test is_data2array2data_identity(data)

    array = rand(FT, Nv, Nij, Nij, 1, Nh)
    data = VIJFH{FT, Nv, Nij, Nh}(array)
    @test DataLayouts.data2array(data) == reshape(parent(data), Nv, :)
    @test is_data2array2data_identity(data)
end
