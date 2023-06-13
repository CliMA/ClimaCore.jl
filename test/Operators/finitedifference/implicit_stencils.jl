#=
The performance of this test needs to be improved
because the compiler is doing a lot of inference,
which makes this test take ~1hr.

# For running all tests:
```
julia -O0 --check-bounds=yes --project=test
push!(ARGS, "--float_type", "Float32")
include(joinpath("test", "Operators", "finitedifference", "implicit_stencils.jl"))
```

# For interactive use:
```julia
julia -O0 --check-bounds=yes --project=test
push!(ARGS, "--float_type", "Float32")
include(joinpath("test", "Operators", "finitedifference", "implicit_stencils_utils.jl"))

const FT = parsed_args["float_type"] == "Float64" ? Float64 : Float32

# Once launched
center_space = get_space(FT);
all_ops = get_all_ops(center_space);

@testset "Test pointwise throws" begin
    @time test_pointwise_stencils_throws(all_ops)
end
@testset "Test pointwise apply" begin
    @time test_pointwise_stencils_apply(all_ops)
end
@testset "Test pointwise compose" begin
    @time test_pointwise_stencils_compose(all_ops)
end
```
=#

include(joinpath(@__DIR__, "implicit_stencils_utils.jl"))
(s, parsed_args) = parse_commandline()

const FT = parsed_args["float_type"] == "Float64" ? Float64 : Float32

function main(::Type{FT}) where {FT}
    center_space = get_space(FT)
    all_ops = get_all_ops(center_space)
    @testset "Test pointwise throws" begin
        #      @time test_pointwise_stencils_throws(all_ops)
    end
    @testset "Test pointwise apply" begin
        @time test_pointwise_stencils_apply(all_ops)
    end
    @testset "Test pointwise compose" begin
        @time test_pointwise_stencils_compose(all_ops)
    end
end

main(FT)
