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

# The scopes of a chain of partitions, from scope down to the scope that is its own partition.
function partition_chain(scope)
    chain = Any[scope]
    while DataLayouts.partition(chain[end]) != chain[end]
        push!(chain, DataLayouts.partition(chain[end]))
    end
    return chain
end

# DataLayouts.points_subscope, one partition of the chain at a time.
function points_subscope_by_chain(scope, max_slice_points)
    chain = partition_chain(scope)
    for (scope, subscope) in zip(chain, chain[2:end])
        subscope == DataLayouts.ThisThread() && return subscope
        if max_slice_points > DataLayouts.num_threads(DataLayouts.partition(subscope))
            exceeds =
                DataLayouts.slice_exceeds_subscope(scope, subscope, max_slice_points)
            return exceeds ? scope : subscope
        end
    end
    return DataLayouts.ThisThread()
end

@testset "scope queries of sub-blocks" begin
    sizes = (1:33..., 48, 63:65..., 96, 127:129..., 255:257...)
    subblocks = [ext.ThisSubBlock{N, 0}() for N in sizes]
    for scope in subblocks
        chain = partition_chain(scope)
        for subscope in (subblocks..., DataLayouts.ThisThread(), ext.ThisBlock())
            @test DataLayouts.is_subscope(subscope, scope) == (subscope in chain)
        end
        for max_slice_points in 0:260
            @test DataLayouts.points_subscope(scope, max_slice_points) ===
                  points_subscope_by_chain(scope, max_slice_points)
        end
    end
    kernel_chain = partition_chain(ext.ThisKernel())
    for subscope in
        (subblocks..., DataLayouts.ThisThread(), ext.ThisKernel(), ext.ThisHost())
        @test DataLayouts.is_subscope(subscope, ext.ThisKernel()) ==
              (subscope in kernel_chain)
    end
    @test DataLayouts.is_subscope(ext.ThisBlock(), ext.ThisKernel())
    for max_slice_points in 0:260
        @test DataLayouts.points_subscope(ext.ThisKernel(), max_slice_points) ==
              points_subscope_by_chain(ext.ThisKernel(), max_slice_points)
    end
end

@testset "scope queries of a kernel launch are constants" begin
    kernel = ext.ThisKernel()
    for subscope in (ext.ThisHost(), DataLayouts.ThisThread(), ext.ThisSubBlock{4}())
        f() = DataLayouts.is_subscope(subscope, kernel)
        code = only(code_typed(f, ()))[1].code
        @test code == Any[Core.ReturnNode(f())]
    end
    g() = DataLayouts.points_subscope(kernel, 74)
    code = only(code_typed(g, ()))[1].code
    @test code == Any[Core.ReturnNode(g())]
end
