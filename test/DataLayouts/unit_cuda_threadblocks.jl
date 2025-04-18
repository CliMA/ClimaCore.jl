#=
julia --project=.buildkite
using Revise; include(joinpath("test", "DataLayouts", "unit_cuda_threadblocks.jl"))
=#
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
using Test
using ClimaCore.DataLayouts
using ClimaCore
import ClimaComms
ClimaComms.@import_required_backends
ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@assert !isnothing(ext) # cuda must be loaded to test this extension

function get_inputs()
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    args = (ArrayType{FT}, zeros)
    return (; S, args)
end

pt_stencil(d) = ext.fd_shmem_stencil_partition(
    DataLayouts.UniversalSize(d),
    DataLayouts.get_Nv(d),
)
pt_sem(d) =
    ext.spectral_partition(DataLayouts.UniversalSize(d), DataLayouts.get_N(d))
get_Nh(h_elem) = h_elem^2 * 6

function pt_masked(d; frac)
    us = DataLayouts.UniversalSize(d)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    n_active_columns = Int(round(prod((Ni, Nj, Nh)) * frac; digits = 0))
    ext.masked_partition(
        DataLayouts.IJHMask,
        n_active_columns,
        DataLayouts.get_N(us),
        us,
    )
end

#! format: off

@testset "linear_partition" begin
    # Fully optimized (but can be 2x slower due to integer division in CartesianIndices).
    # If https://github.com/maleadt/StaticCartesian.jl/issues/1 ever works, we should
    # basically always use that instead.
end

@testset "fd_shmem_stencil_partition" begin
    (; S, args) = get_inputs()
    for DL in (VIFH, VIHF)
        @test pt_stencil(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 1), Nvthreads = 10)
        @test pt_stencil(DL{S}(args...; Nv = 10, Ni = 4, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 4), Nvthreads = 10)
        @test pt_stencil(DL{S}(args...; Nv = 100, Ni = 4, Nh = get_Nh(100))) == (; threads = (100,), blocks = (60000, 1, 4), Nvthreads = 100)
    end
    for DL in (VIJFH, VIJHF)
        @test pt_stencil(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 1), Nvthreads = 10)
        @test pt_stencil(DL{S}(args...; Nv = 10, Nij = 4, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 16), Nvthreads = 10)
        @test pt_stencil(DL{S}(args...; Nv = 100, Nij = 4, Nh = get_Nh(100))) == (; threads = (100,), blocks = (60000, 1, 16), Nvthreads = 100)
    end
    @test pt_stencil(VF{S}(args...; Nv = 10)) == (; threads = (10,), blocks = (1, 1, 1), Nvthreads = 10)
    @test pt_stencil(VF{S}(args...; Nv = 1000)) == (; threads = (1000,), blocks = (1, 1, 1), Nvthreads = 1000)
end

@testset "spectral_partition" begin
    (; S, args) = get_inputs()
    for DL in (VIFH, VIHF)
        @test pt_sem(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (60000, 1), Nvthreads = 64)
        @test pt_sem(DL{S}(args...; Nv = 10, Ni = 4, Nh = get_Nh(100))) == (; threads = (4, 1, 64), blocks = (60000, 1), Nvthreads = 64)
        @test pt_sem(DL{S}(args...; Nv = 100, Ni = 4, Nh = get_Nh(100))) == (; threads = (4, 1, 64), blocks = (60000, 2), Nvthreads = 64)
    end
    for DL in (VIJFH, VIJHF)
        @test pt_sem(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (60000, 1), Nvthreads = 64)
        @test pt_sem(DL{S}(args...; Nv = 10, Nij = 4, Nh = get_Nh(100))) == (; threads = (4, 4, 64), blocks = (60000, 1), Nvthreads = 64)
        @test pt_sem(DL{S}(args...; Nv = 100, Nij = 4, Nh = get_Nh(100))) == (; threads = (4, 4, 64), blocks = (60000, 2), Nvthreads = 64)
    end
    for DL in (IJFH, IJHF)
        @test pt_sem(DL{S}(args...; Nij = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (60000, 1), Nvthreads = 64) # can/should we reduce # of blocks?
        @test pt_sem(DL{S}(args...; Nij = 4, Nh = get_Nh(100))) == (; threads = (4, 4, 64), blocks = (60000, 1), Nvthreads = 64) # can/should we reduce # of blocks?
    end
end

@testset "masked_partition" begin
    (; S, args) = get_inputs()
    for DL in (VIFH, VIHF)
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = 300000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = 60000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = 480000, blocks = 1)

        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = 3000000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = 600000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = 4800000, blocks = 1)
    end
    for DL in (VIJFH, VIJHF)
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = 300000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = 60000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = 480000, blocks = 1)

        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = 3000000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = 600000, blocks = 1)
        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = 4800000, blocks = 1)
    end
end

#! format: on
