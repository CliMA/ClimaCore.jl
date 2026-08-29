ENV["CLIMACOMMS_DEVICE"] = "CUDA"
using Test
import ClimaCore
import ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends
ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@assert !isnothing(ext) # cuda must be loaded to test this extension

# Construct a layout of undefined values from the parent array type, since only
# sizes matter for computing partitions. Layouts without a J axis (VIFH and
# VIHF) are constructed by setting Nj to 1 in the corresponding unified type.
function make_data(DL, S; Nv = 1, Ni = 1, Nj = 1, Nh = nothing)
    A = ClimaComms.array_type(ClimaComms.device()){S}
    return isnothing(Nh) ? DL{S, Nv, Ni, Nj, 1}(A) : DL{S, Nv, Ni, Nj, nothing}(A, Nh)
end

pt_stencil(d) = ext.fd_shmem_stencil_partition(d, size(d, 1))
get_Nh(h_elem) = h_elem^2 * 6

function pt_masked(d; frac)
    (Nv, Ni, Nj, Nh) = size(d)
    n_active_columns = Int(round(Ni * Nj * Nh * frac; digits = 0))
    return ext.masked_partition(DataLayouts.IJHMask, n_active_columns, length(d), d)
end

#! format: off

@testset "linear_partition" begin
    # Fully optimized (but can be 2x slower due to integer division in CartesianIndices).
    # If https://github.com/maleadt/StaticCartesian.jl/issues/1 ever works, we should
    # basically always use that instead.
end

@testset "fd_shmem_stencil_partition" begin
    S = Float64
    for DL in (DataLayouts.VIJFH, DataLayouts.VIJHF)
        @test pt_stencil(make_data(DL, S; Nv = 10, Ni = 1, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 1), Nvthreads = 10)
        @test pt_stencil(make_data(DL, S; Nv = 10, Ni = 4, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 4), Nvthreads = 10)
        @test pt_stencil(make_data(DL, S; Nv = 100, Ni = 4, Nh = get_Nh(100))) == (; threads = (100,), blocks = (60000, 1, 4), Nvthreads = 100)

        @test pt_stencil(make_data(DL, S; Nv = 10, Ni = 1, Nj = 1, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 1), Nvthreads = 10)
        @test pt_stencil(make_data(DL, S; Nv = 10, Ni = 4, Nj = 4, Nh = get_Nh(100))) == (; threads = (10,), blocks = (60000, 1, 16), Nvthreads = 10)
        @test pt_stencil(make_data(DL, S; Nv = 100, Ni = 4, Nj = 4, Nh = get_Nh(100))) == (; threads = (100,), blocks = (60000, 1, 16), Nvthreads = 100)
    end
    @test pt_stencil(make_data(DataLayouts.VIJFH, S; Nv = 10)) == (; threads = (10,), blocks = (1, 1, 1), Nvthreads = 10)
    @test pt_stencil(make_data(DataLayouts.VIJFH, S; Nv = 1000)) == (; threads = (1000,), blocks = (1, 1, 1), Nvthreads = 1000)
end

@testset "masked_partition" begin
    S = Float64
    for DL in (DataLayouts.VIJFH, DataLayouts.VIJHF)
        @test pt_masked(make_data(DL, S; Nv = 10, Nh = get_Nh(100)); frac = 0.5) == (; threads = 300000, blocks = 1)
        @test pt_masked(make_data(DL, S; Nv = 10, Nh = get_Nh(100)); frac = 0.1) == (; threads = 60000, blocks = 1)
        @test pt_masked(make_data(DL, S; Nv = 10, Nh = get_Nh(100)); frac = 0.8) == (; threads = 480000, blocks = 1)

        @test pt_masked(make_data(DL, S; Nv = 100, Nh = get_Nh(100)); frac = 0.5) == (; threads = 3000000, blocks = 1)
        @test pt_masked(make_data(DL, S; Nv = 100, Nh = get_Nh(100)); frac = 0.1) == (; threads = 600000, blocks = 1)
        @test pt_masked(make_data(DL, S; Nv = 100, Nh = get_Nh(100)); frac = 0.8) == (; threads = 4800000, blocks = 1)
    end
end

#! format: on

# A sub-block wider than a warp has no barrier of its own and uses the
# block-wide sync_threads, so every block must hold exactly one of them; see
# ext.max_subblock_launch_threads.
@testset "a block holds one wide sub-block" begin
    for N in (2, 4, 8, 16, 32, 64, 128, 256)
        subscope = ext.ThisSubBlock{N}()
        cap = ext.max_subblock_launch_threads(subscope)
        block_threads = DataLayouts.subscope_launch_threads(subscope, cap)

        # Whole sub-blocks, so none is missing threads (see the invariant in
        # DataLayouts.subscope_launch_threads).
        @test block_threads % N == 0

        # One wide sub-block per block, so a block-wide barrier is exactly that
        # sub-block's barrier. Narrower sub-blocks keep sharing a block.
        @test cld(block_threads, N) ==
              (N > ext.THREADS_PER_WARP ? 1 : cld(ext.MAX_SUBBLOCK_LAUNCH_THREADS, N))

        # The trailing dimension of a sub-block's shared memory has one entry per
        # sub-block a block can hold, so the largest index a launched block can
        # produce has to be in range.
        @test cld(cap, N) == cld(block_threads, N)
    end
end
