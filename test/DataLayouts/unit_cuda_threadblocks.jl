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

pt(d) = ext.partition(d, DataLayouts.get_N(DataLayouts.UniversalSize(d)))
pt_stencil(d) = ext.fd_shmem_stencil_partition(
    DataLayouts.UniversalSize(d),
    DataLayouts.get_Nv(d),
)
pt_columnwise(d) =
    ext.columnwise_partition(DataLayouts.UniversalSize(d), DataLayouts.get_N(d))
pt_mfs(d; Nnames) = ext.multiple_field_solve_partition(
    DataLayouts.UniversalSize(d),
    DataLayouts.get_N(d);
    Nnames,
)
pt_sem(d) =
    ext.spectral_partition(DataLayouts.UniversalSize(d), DataLayouts.get_N(d))
get_Nh(h_elem) = h_elem^2 * 6

function pt_masked(d; frac)
    us = DataLayouts.UniversalSize(d)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    n_active_columns = Int(round(prod((Ni, Nj, Nh)) * frac; digits = 0))
    ext.masked_partition(
        us,
        DataLayouts.get_N(us),
        DataLayouts.IJHMask,
        n_active_columns,
    )
end

#! format: off

@testset "linear_partition" begin
    # Fully optimized (but can be 2x slower due to integer division in CartesianIndices).
    # If https://github.com/maleadt/StaticCartesian.jl/issues/1 ever works, we should
    # basically always use that instead.
end
@testset "DataF partition" begin
    (; S, args) = get_inputs()
    @test pt(DataF{S}(args...)) == (; threads = 1, blocks = 1)
end
@testset "IJFH/IJHF partition" begin
    (; S, args) = get_inputs()
    for DL in (IJFH, IJHF)
        @test pt(DL{S}(args...; Nij = 1, Nh = 1)) == (; threads = (1, 1, 1), blocks = (1,))
        @test pt(DL{S}(args...; Nij = 1, Nh = get_Nh(1))) == (; threads = (1, 1, 6), blocks = (1,))
        @test pt(DL{S}(args...; Nij = 4, Nh = get_Nh(30))) == (; threads = (4, 4, 64), blocks = (85,))
        @test pt(DL{S}(args...; Nij = 4, Nh = get_Nh(100))) == (; threads = (4, 4, 64), blocks = (938,))
        @test pt(DL{S}(args...; Nij = 1, Nh = get_Nh(1))) == (; threads = (1, 1, 6), blocks = (1,))
        @test pt(DL{S}(args...; Nij = 1, Nh = get_Nh(30))) == (; threads = (1, 1, 64), blocks = (85,))
        @test pt(DL{S}(args...; Nij = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (938,))
    end
end
@testset "IFH/IHF partition" begin
    (; S, args) = get_inputs()
    for DL in (IFH, IHF)
        @test pt(DL{S}(args...; Ni = 1, Nh = 1)) == (; threads = (1, 1), blocks = (1,))
        @test pt(DL{S}(args...; Ni = 1, Nh = get_Nh(1))) == (; threads = (1, 6), blocks = (1,))
        @test pt(DL{S}(args...; Ni = 4, Nh = get_Nh(30))) == (; threads = (4, 5400), blocks = (1,))
        @test pt(DL{S}(args...; Ni = 4, Nh = get_Nh(100))) == (; threads = (4, 60000), blocks = (1,)) # TODO: needs fixed (too many threads per block)
        @test pt(DL{S}(args...; Ni = 1, Nh = get_Nh(30))) == (; threads = (1, 5400), blocks = (1,))
        @test pt(DL{S}(args...; Ni = 1, Nh = get_Nh(100))) == (; threads = (1, 60000), blocks = (1,)) # TODO: needs fixed (too many threads per block)
    end
end
@testset "IJF partition" begin
    (; S, args) = get_inputs()
    @test pt(IJF{S}(args...; Nij = 1)) == (; threads = (1, 1), blocks = (1,))
    @test pt(IJF{S}(args...; Nij = 4)) == (; threads = (4, 4), blocks = (1,))
end
@testset "IF partition" begin
    (; S, args) = get_inputs()
    @test pt(IF{S}(args...; Ni = 1)) == (; threads = (1,), blocks = (1,))
    @test pt(IF{S}(args...; Ni = 4)) == (; threads = (4,), blocks = (1,))
end
@testset "VF partition" begin
    (; S, args) = get_inputs()
    @test pt(VF{S}(args...; Nv = 1)) == (; threads = (1, ), blocks = (1, ))
    @test pt(VF{S}(args...; Nv = 10)) == (; threads = (1, ), blocks = (10, ))
    @test pt(VF{S}(args...; Nv = 64)) == (; threads = (1, ), blocks = (64, ))
    @test pt(VF{S}(args...; Nv = 1000)) == (; threads = (1, ), blocks = (1000, ))
end
@testset "VIJFH/VIJHF partition" begin
    (; S, args) = get_inputs()
    for DL in (VIJFH, VIJHF)
        @test pt(DL{S}(args...; Nv = 1, Nij = 1, Nh = 1)) == (; threads = (1, 1, 1), blocks = (1, 1))
        @test pt(DL{S}(args...; Nv = 1, Nij = 1, Nh = get_Nh(1))) == (; threads = (1, 1, 1), blocks = (6, 1))
        @test pt(DL{S}(args...; Nv = 64, Nij = 4, Nh = get_Nh(30))) == (; threads = (64, 4, 4), blocks = (5400, 1))
        @test pt(DL{S}(args...; Nv = 64, Nij = 4, Nh = get_Nh(100))) == (; threads = (64, 4, 4), blocks = (60000, 1))
        @test pt(DL{S}(args...; Nv = 64, Nij = 1, Nh = get_Nh(100))) == (; threads = (64, 1, 1), blocks = (60000, 1)) # need more threads per block?
        @test pt(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100))) == (; threads = (10, 1, 1), blocks = (60000, 1)) # need more threads per block
        @test pt(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(30))) == (; threads = (10, 1, 1), blocks = (5400, 1)) # need more threads per block
        @test pt(DL{S}(args...; Nv = 1000, Nij = 1, Nh = get_Nh(30))) == (; threads = (1000, 1, 1), blocks = (5400, 1))
        @test pt(DL{S}(args...; Nv = 2000, Nij = 1, Nh = get_Nh(30))) == (; threads = (2000, 1, 1), blocks = (5400, 1)) # TODO: fix this? maximum_allowable_threads()[1] == 1024
    end
end
@testset "VIFH/VIHF partition" begin
    (; S, args) = get_inputs()
    for DL in (VIFH, VIHF)
        @test pt(DL{S}(args...; Nv = 1, Ni = 1, Nh = 1)) == (; threads = (1, 1), blocks = (1, 1))
        @test pt(DL{S}(args...; Nv = 1, Ni = 1, Nh = get_Nh(1))) == (; threads = (1, 1), blocks = (6, 1))
        @test pt(DL{S}(args...; Nv = 64, Ni = 4, Nh = get_Nh(30))) == (; threads = (64, 4), blocks = (5400, 1))
        @test pt(DL{S}(args...; Nv = 64, Ni = 4, Nh = get_Nh(100))) == (; threads = (64, 4), blocks = (60000, 1))
        @test pt(DL{S}(args...; Nv = 64, Ni = 1, Nh = get_Nh(100))) == (; threads = (64, 1), blocks = (60000, 1)) # need more threads per block?
        @test pt(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100))) == (; threads = (10, 1), blocks = (60000, 1)) # need more threads per block
    end
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

@testset "columnwise_partition" begin
    (; S, args) = get_inputs()
    for DL in (IFH, IHF)
        @test pt_columnwise(DL{S}(args...; Ni = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (938,))
        @test pt_columnwise(DL{S}(args...; Ni = 4, Nh = get_Nh(100))) == (; threads = (4, 1, 64), blocks = (938,))
        @test pt_columnwise(DL{S}(args...; Ni = 4, Nh = get_Nh(100))) == (; threads = (4, 1, 64), blocks = (938,))
    end
    for DL in (IJFH, IJHF)
        @test pt_columnwise(DL{S}(args...; Nij = 1, Nh = get_Nh(100))) == (; threads = (1, 1, 64), blocks = (938,)) # more threads per block?
        @test pt_columnwise(DL{S}(args...; Nij = 4, Nh = get_Nh(100))) == (; threads = (4, 4, 64), blocks = (938,)) # more threads per block?
    end
end

@testset "multiple_field_solve_partition" begin
    (; S, args) = get_inputs()
    for DL in (IFH, IHF)
        @test pt_mfs(DL{S}(args...; Ni = 1, Nh = get_Nh(100)); Nnames = 1) == (; threads = (1, 1, 1), blocks = (60000,))
        @test pt_mfs(DL{S}(args...; Ni = 4, Nh = get_Nh(100)); Nnames = 2) == (; threads = (4, 1, 2), blocks = (60000,)) # more threads per block?
    end
    for DL in (IJFH, IJHF)
        @test pt_mfs(DL{S}(args...; Nij = 1, Nh = get_Nh(100)); Nnames = 1) == (; threads = (1, 1, 1), blocks = (60000,)) # more threads per block?
        @test pt_mfs(DL{S}(args...; Nij = 4, Nh = get_Nh(100)); Nnames = 2) == (; threads = (4, 4, 2), blocks = (60000,)) # more threads per block?
    end
end

@testset "masked_partition" begin
    (; S, args) = get_inputs()
    for DL in (VIFH, VIHF)
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = (10,), blocks = (30000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = (10,), blocks = (6000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 10, Ni = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = (10,), blocks = (48000, 1)) # need more threads per block

        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = (100,), blocks = (30000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = (100,), blocks = (6000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 100, Ni = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = (100,), blocks = (48000, 1)) # need more threads per block
    end
    for DL in (VIJFH, VIJHF)
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = (10,), blocks = (30000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = (10,), blocks = (6000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 10, Nij = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = (10,), blocks = (48000, 1)) # need more threads per block

        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.5) == (; threads = (100,), blocks = (30000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.1) == (; threads = (100,), blocks = (6000, 1)) # need more threads per block
        @test pt_masked(DL{S}(args...; Nv = 100, Nij = 1, Nh = get_Nh(100)); frac = 0.8) == (; threads = (100,), blocks = (48000, 1)) # need more threads per block
    end
end

#! format: on
