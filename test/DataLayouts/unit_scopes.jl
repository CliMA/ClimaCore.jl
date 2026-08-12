# Direct unit tests of the DataScope interface: scope assignment from array
# types, subscope ordering, scope combination, thread accounting, and scoped
# allocation.
using Test
import StaticArrays
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts:
    DataScope,
    ThisThread,
    ThisThreadPool,
    is_subscope,
    partition,
    num_threads,
    thread_rank,
    num_partitions,
    scoped_array,
    scoped_static_array,
    VIJFH

@testset "DataScope assignment from array types" begin
    @test DataScope(Array{Float64, 2}) == ThisThreadPool()
    @test DataScope(zeros(2, 2)) == ThisThreadPool()
    @test DataScope(StaticArrays.SVector{3, Float64}) == ThisThread()
    @test DataScope(StaticArrays.MMatrix{2, 2, Float64, 4}) == ThisThread()
    # Wrappers inherit the scope of their parent array.
    @test DataScope(view(zeros(4, 4), 1:2, :)) == ThisThreadPool()
    @test DataScope(reshape(zeros(4), 2, 2)) == ThisThreadPool()
    # A DataLayout carries its scope as a type parameter.
    data = VIJFH{Float64, 2, 3, 3, nothing}(Array{Float64}, 5)
    @test DataScope(data) == ThisThreadPool()
    @test DataScope(typeof(data)) == ThisThreadPool()
end

@testset "subscope ordering and combination" begin
    # is_subscope is reflexive, and ThisThread is a partition of the pool.
    @test is_subscope(ThisThread(), ThisThread())
    @test is_subscope(ThisThreadPool(), ThisThreadPool())
    @test partition(ThisThreadPool()) == ThisThread()
    @test is_subscope(ThisThread(), ThisThreadPool())
    @test !is_subscope(ThisThreadPool(), ThisThread())
    # Combining scopes always selects the smallest one.
    @test DataScope(ThisThread(), ThisThreadPool()) == ThisThread()
    @test DataScope(ThisThreadPool(), ThisThread()) == ThisThread()
    @test DataScope(zeros(2), StaticArrays.SA[1, 2]) == ThisThread()
end

@testset "thread accounting" begin
    @test num_threads(ThisThread()) == 1
    @test thread_rank(ThisThread()) == 1
    @test num_partitions(ThisThread()) == 1
    # Outside of any pool loop, the pool spans the default thread pool, and
    # the calling thread has rank 1.
    @test num_threads(ThisThreadPool()) >= 1
    @test thread_rank(ThisThreadPool()) == 1
    @test num_partitions(ThisThreadPool()) == num_threads(ThisThreadPool())
    # Synchronizing a single thread is a no-op; synchronizing across more
    # than one thread from serial code must throw.
    @test DataLayouts.synchronize(ThisThread()) == true
    if num_threads(ThisThreadPool()) > 1
        @test_throws ArgumentError DataLayouts.synchronize(ThisThreadPool())
    end
end

@testset "scoped allocation" begin
    array = scoped_array(ThisThread(), Float64, 3)
    @test array isa Vector{Float64} && length(array) == 3
    buffer = scoped_array(ThisThread(), Float64, 3; buffer = true)
    @test length(buffer) == 3
    # The task-local buffer is reused across calls, not reallocated.
    @test parent(scoped_array(ThisThread(), Float64, 3; buffer = true)) ===
          parent(buffer)
    static = scoped_static_array(ThisThread(), Float64, (2, 2))
    @test static isa StaticArrays.MArray{Tuple{2, 2}, Float64}
    # CPU pool threads share host memory, so pool-scoped allocation is
    # supported too.
    pool_array = scoped_array(ThisThreadPool(), Float64, 3)
    @test pool_array isa Vector{Float64} && length(pool_array) == 3
end
