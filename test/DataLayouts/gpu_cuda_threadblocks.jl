ENV["CLIMACOMMS_DEVICE"] = "CUDA"
using Test
import ClimaCore
import ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends
ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@assert !isnothing(ext) # cuda must be loaded to test this extension

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
