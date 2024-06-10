#=
using Revise; include(joinpath("test", "DataLayouts", "unit_fill.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends

@testset "fill! with Nf = 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    for FT in (Float32, Float64)
        S = FT
        Nf = 1
        Nv = 4
        Nij = 3
        Nh = 5
        Nk = 2
    #! format: off
        data = DataF{S}(device_zeros(FT,Nf));                        fill!(data, 3); @test all(parent(data) .== 3)
        data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         fill!(data, 3); @test all(parent(data) .== 3)
        data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              fill!(data, 3); @test all(parent(data) .== 3)
        data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             fill!(data, 3); @test all(parent(data) .== 3)
        data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  fill!(data, 3); @test all(parent(data) .== 3)
        data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    fill!(data, 3); @test all(parent(data) .== 3)
        data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); fill!(data, 3); @test all(parent(data) .== 3)
        data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      fill!(data, 3); @test all(parent(data) .== 3)
    #! format: on
        # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); fill!(data, (2,3)); @test all(parent(data) .== 3) # TODO: test
        # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             fill!(data, (2,3)); @test all(parent(data) .== 3) # TODO: test
    # end

    # @testset "fill! with Nf > 1" begin
        # device = ClimaComms.device()
        # device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
        # FT = Float64
        S = Tuple{FT, FT}
        Nf = 2
    #! format: off
        data = DataF{S}(device_zeros(FT,Nf));                        fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
        data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      fill!(data, (2,3)); @test all(parent(data.:1) .== 2); @test all(parent(data.:2) .== 3)
    #! format: on
        # TODO: test this
        # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); fill!(data, (2,3)); @test all(parent(data) .== (2,3)) # TODO: test
        # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             fill!(data, (2,3)); @test all(parent(data) .== (2,3)) # TODO: test
    end
end
