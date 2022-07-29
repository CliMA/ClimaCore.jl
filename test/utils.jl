import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--bin_id"
        help = "Bin ID for load balancing parallelized tests"
        arg_type = Int
        default = -1
        "--nbins"
        help = "Number of bins for load balancing parallelized tests"
        arg_type = Int
        default = -1
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    @assert parsed_args["bin_id"] ≠ -1
    return (s, parsed_args)
end

# Logic borrowed from SafeTestsets.jl
export @safetimedtestset
const errt = ArgumentError(
    """
Use `@safetimedtestset` like the following:
times = Dict()
@safetimedtestset times "Benchmark Tests" begin include("benchmark_tests.jl") end
""",
)
macro safetimedtestset(times, args...)
    length(args) != 2 && throw(err)
    name, expr = args
    if name isa String
        mod = gensym(name)
        testname = name
    elseif name isa Expr && name.head == :(=) && length(name.args) == 2
        mod, testname = name.args
    else
        throw(err)
    end
    quote
        $(esc(times))[$testname] = @elapsed @time @eval module $mod
        using Test, SafeTestsets
        @testset $testname $expr
        end
        nothing
    end
end

#=
Return a Dict of bin IDs per test such
that they are load balanced
=#
function load_balanced_bins(times::Dict)
    # TODO: implement
end
