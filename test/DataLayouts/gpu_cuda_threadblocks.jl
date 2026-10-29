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
# ext.max_subblock_launch_threads. Narrower sub-blocks share blocks of up to 1024
# threads, since their shared memory is sized by the launch (see
# ext.scoped_static_array). Either way a block holds whole sub-blocks, the
# granularity of its launch configuration (see the whole subscopes invariant in
# DataLayouts.subscope_indices).
@testset "a block holds one wide sub-block" begin
    for N in (2, 4, 8, 16, 32, 64, 128, 256)
        block_threads = ext.max_subblock_launch_threads(ext.ThisSubBlock{N}())
        @test block_threads % N == 0
        @test cld(block_threads, N) == (N > ext.THREADS_PER_WARP ? 1 : cld(1024, N))
    end
end
