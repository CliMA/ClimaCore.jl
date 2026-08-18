using Test
import ClimaCore
import ClimaCore: DebugOnly

# Unit tests for the DebugOnly module (debugging hooks).
# NOTE: `call_post_op_callback`/`allow_mismatched_spaces_unsafe` are meant to
# be overloaded by users while debugging; overriding them here would leak a
# global method redefinition into the rest of the suite, so only the default
# behavior and the pure helpers are tested.

@testset "DebugOnly" begin
    @testset "safe defaults" begin
        # Debug hooks must be off by default: production behavior.
        @test DebugOnly.call_post_op_callback() == false
        @test DebugOnly.allow_mismatched_spaces_unsafe() == false
    end

    @testset "example_debug_post_op_callback" begin
        # Clean results pass through silently.
        @test isnothing(DebugOnly.example_debug_post_op_callback(1.0))
        @test isnothing(DebugOnly.example_debug_post_op_callback([1.0, 2.0]))
        # NaNs and Infs are flagged, for scalars and arrays.
        @test_throws ErrorException DebugOnly.example_debug_post_op_callback(NaN)
        @test_throws ErrorException DebugOnly.example_debug_post_op_callback(Inf)
        @test_throws ErrorException DebugOnly.example_debug_post_op_callback([1.0, NaN])
        @test_throws ErrorException DebugOnly.example_debug_post_op_callback([1.0, -Inf])
    end

    @testset "depth-limited stack traces" begin
        st = stacktrace()
        limited = DebugOnly.depth_limited_stack_trace(devnull, st; maxtypedepth = 2)
        @test limited isa Vector{String}
        @test length(limited) == length(st)

        io = IOBuffer()
        DebugOnly.print_depth_limited_stack_trace(io, st; maxtypedepth = 2)
        @test !isempty(String(take!(io)))
    end
end
