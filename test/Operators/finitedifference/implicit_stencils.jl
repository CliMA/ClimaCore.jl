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
push!(ARGS, "--float_type", "Float32");
include(joinpath("test", "Operators", "finitedifference", "implicit_stencils_utils.jl"))
(s, parsed_args) = parse_commandline();
const FT = parsed_args["float_type"] == "Float64" ? Float64 : Float32;

# Once launched
center_space = get_space(FT);
all_ops = get_all_ops(center_space);
```
Then see tests below
=#

include(joinpath(@__DIR__, "implicit_stencils_utils.jl"))
(s, parsed_args) = parse_commandline()

const FT = parsed_args["float_type"] == "Float64" ? Float64 : Float32

# Need to remove main and put `all_ops` in Main scope.
center_space = get_space(FT)
all_ops = get_all_ops(center_space)
@testset "Test pointwise throws" begin
    @time test_pointwise_stencils_throws(all_ops)
end
@testset "Test pointwise apply" begin
    # @time test_pointwise_stencils_apply(all_ops)
    #! format: off
    # Manually unroll for better inference:
    @time begin
    Base.Cartesian.@nexprs 5 i -> apply_single(all_ops.a_FS, all_ops.a_FS, all_ops.ops_F2C_S2S[i])
    Base.Cartesian.@nexprs 2 i -> apply_single(all_ops.a_FS, all_ops.a_FS, all_ops.ops_F2C_S2V[i])
    Base.Cartesian.@nexprs 5 i -> apply_single(all_ops.a_CS, all_ops.a_CS, all_ops.ops_C2F_S2S[i])
    Base.Cartesian.@nexprs 2 i -> apply_single(all_ops.a_CS, all_ops.a_CS, all_ops.ops_C2F_S2V[i])
    Base.Cartesian.@nexprs 8 i -> apply_single(all_ops.a_FS, all_ops.a_FV, all_ops.ops_F2C_V2V[i])
    Base.Cartesian.@nexprs 5 i -> apply_single(all_ops.a_FS, all_ops.a_FV, all_ops.ops_F2C_V2S[i])
    Base.Cartesian.@nexprs 7 i -> apply_single(all_ops.a_CS, all_ops.a_CV, all_ops.ops_C2F_V2V[i])
    Base.Cartesian.@nexprs 3 i -> apply_single(all_ops.a_CS, all_ops.a_CV, all_ops.ops_C2F_V2S[i])
    end
    #! format: on
end
@testset "Test pointwise compose" begin
    # @time test_pointwise_stencils_compose(all_ops)
    #! format: off
    @time begin
    ctr = Int[0]
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FS, all_ops.a_CS, all_ops.ops_F2C_S2S[i], all_ops.ops_C2F_S2S[j], i, j, 1)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FS, all_ops.a_CS, all_ops.ops_F2C_S2S[i], all_ops.ops_C2F_S2V[j], i, j, 2)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CS, all_ops.a_FS, all_ops.ops_C2F_S2S[i], all_ops.ops_F2C_S2S[j], i, j, 3)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CS, all_ops.a_FS, all_ops.ops_C2F_S2S[i], all_ops.ops_F2C_S2V[j], i, j, 4)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 7 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FS, all_ops.a_CV, all_ops.ops_F2C_S2S[i], all_ops.ops_C2F_V2V[j], i, j, 5)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 3 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FS, all_ops.a_CV, all_ops.ops_F2C_S2S[i], all_ops.ops_C2F_V2S[j], i, j, 6)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 8 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CS, all_ops.a_FV, all_ops.ops_C2F_S2S[i], all_ops.ops_F2C_V2V[j], i, j, 7)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CS, all_ops.a_FV, all_ops.ops_C2F_S2S[i], all_ops.ops_F2C_V2S[j], i, j, 8)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FV, all_ops.a_CS, all_ops.ops_F2C_V2S[i], all_ops.ops_C2F_S2S[j], i, j, 9)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FV, all_ops.a_CS, all_ops.ops_F2C_V2S[i], all_ops.ops_C2F_S2V[j], i, j, 10)
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CV, all_ops.a_FS, all_ops.ops_C2F_V2S[i], all_ops.ops_F2C_S2S[j], i, j, 11)
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 2 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CV, all_ops.a_FS, all_ops.ops_C2F_V2S[i], all_ops.ops_F2C_S2V[j], i, j, 12)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 7 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FV, all_ops.a_CV, all_ops.ops_F2C_V2S[i], all_ops.ops_C2F_V2V[j], i, j, 13)
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 3 j -> compose_single(ctr, all_ops.a_FS, all_ops.a_FV, all_ops.a_CV, all_ops.ops_F2C_V2S[i], all_ops.ops_C2F_V2S[j], i, j, 14)
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 8 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CV, all_ops.a_FV, all_ops.ops_C2F_V2S[i], all_ops.ops_F2C_V2V[j], i, j, 15)
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 5 j -> compose_single(ctr, all_ops.a_CS, all_ops.a_CV, all_ops.a_FV, all_ops.ops_C2F_V2S[i], all_ops.ops_F2C_V2S[j], i, j, 16)
    end
    #! format: on
end
