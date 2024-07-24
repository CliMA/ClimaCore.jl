#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_ndims.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends

@testset "Base.ndims" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                                            @test ndims(data) == 1; @test ndims(typeof(data)) == 1
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                                      @test ndims(data) == 2; @test ndims(typeof(data)) == 2
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                                        @test ndims(data) == 2; @test ndims(typeof(data)) == 2
    data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));                              @test ndims(data) == 3; @test ndims(typeof(data)) == 3
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));                                 @test ndims(data) == 3; @test ndims(typeof(data)) == 3
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));                         @test ndims(data) == 4; @test ndims(typeof(data)) == 4
    data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));                      @test ndims(data) == 4; @test ndims(typeof(data)) == 4
    data = VIJFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));                 @test ndims(data) == 5; @test ndims(typeof(data)) == 5
    data = DataLayouts.IJKFVH{S,Nij,Nk,Nv,Nh}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); @test ndims(data) == 6; @test ndims(typeof(data)) == 6
    data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                 @test ndims(data) == 2; @test ndims(typeof(data)) == 2
#! format: on
end

function test_perm_to_array(data)
    pa = DataLayouts.permute_axes(map(x -> Base.OneTo(x), size(data)), data)
    @test pa == map(x -> Base.OneTo(x), size(parent(data)))
end

@testset "test_perm_to_array" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                        test_perm_to_array(data)
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));     test_perm_to_array(data)
    data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));          test_perm_to_array(data)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_perm_to_array(data)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_perm_to_array(data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_perm_to_array(data)
    data = VIJFH{S,Nv,Nij,Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));test_perm_to_array(data)
    data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));  test_perm_to_array(data)
#! format: on
    # data = DataLayouts.IJKFVH{S, Nij, Nk, Nv, Nh}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_perm_to_array(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                     test_perm_to_array(data) # TODO: test
end
