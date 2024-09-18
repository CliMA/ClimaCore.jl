#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "has_uniform_datalayouts.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaCore.Geometry
import ClimaComms
import LazyBroadcast: @lazy
using StaticArrays
import Random
Random.seed!(1234)

@testset "has_uniform_datalayouts" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
#! format: off
    data_DataF = DataF{S}(device_zeros(FT,Nf));
    data_IJFH = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));
    data_IFH = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));
    data_IJF = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));
    data_IF = IF{S, Nij}(device_zeros(FT,Nij,Nf));
    data_VF = VF{S, Nv}(device_zeros(FT,Nv,Nf));
    data_VIJFH = VIJFH{S,Nv,Nij,Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));
    data_VIFH = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));
#! format: on

    bc = @lazy @. data_VIFH + data_VIFH
    @test DataLayouts.has_uniform_datalayouts(bc)
    bc = @lazy @. data_IJFH + data_VF
    @test !DataLayouts.has_uniform_datalayouts(bc)

    data_VIJFHᶜ = VIJFH{S, Nv, Nij, Nh}(device_zeros(FT, Nv, Nij, Nij, Nf, Nh))
    data_VIJFHᶠ =
        VIJFH{S, Nv + 1, Nij, Nh}(device_zeros(FT, Nv + 1, Nij, Nij, Nf, Nh))

    # This is not a valid broadcast expression,
    # but these two datalayouts can exist in a
    # valid broadcast expression (e.g., interpolation).
    bc = @lazy @. data_VIJFHᶜ + data_VIJFHᶠ
    @test DataLayouts.has_uniform_datalayouts(bc)
end
