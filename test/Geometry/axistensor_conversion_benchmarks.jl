using Test
using StaticArrays
import BenchmarkTools
import StatsBase
import OrderedCollections
import Random
#! format: off
using ClimaCore.Geometry:
    AbstractAxis,
    CovariantAxis,
    AxisVector,
    ContravariantAxis,
    LocalAxis,
    CartesianAxis,
    AxisTensor,
    Covariant1Vector,
    Covariant13Vector,
    UVVector,
    UWVector,
    UVector,
    WVector,
    Covariant12Vector,
    UVWVector,
    Covariant123Vector,
    Covariant3Vector,
    Contravariant12Vector,
    Contravariant3Vector,
    Contravariant123Vector,
    Contravariant13Vector,
    Contravariant2Vector,
    Axis2Tensor

include("transform_project.jl") # compact, generic but unoptimized reference
include("used_transform_args.jl")
include("used_project_args.jl")

function benchmark_axistensor_conversions(args, ::Type{FT}, func::F) where {FT, F}
    results = OrderedCollections.OrderedDict()
    for (aTo, x) in args
        func(aTo, x) # compile first

        # Setting up a benchmark is _really_ slow
        # for many many microbenchmarks. So, let's
        # just hardcode something simple that runs
        # fast..

        #### Using BenchmarkTools

        # b = BenchmarkTools.@benchmarkable $func($aTo, $x)
        # trial = BenchmarkTools.run(b, samples = 3)
        # time = StatsBase.mean(trial.times)
        # time = BenchmarkTools.@btime begin; result = func($aTo, $x); end

        #### Hard-code average of 25 calls
        time = @elapsed begin
            func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x)
            func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x)
            func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x)
            func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x)
            func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x); func(aTo, x)
        end
        ns = 1e9
        time = time/25*ns # average and convert to ns

        key = typeof.((aTo, x))
        result = func(aTo, x)
        t_pretty = BenchmarkTools.prettytime(time)
        @info "Benchmarking $t_pretty $func with $key"
        results[key] = (time, t_pretty, result)
    end
    return results
end

function get_conversion_args(ucs)
    map(ucs) do (axt, axtt)
        (axt(), rand(axtt))
    end
end

function test_optimized_transform(::Type{FT}) where {FT}
    @info "Testing optimized transform..."
    utat = used_transform_arg_types(FT)
    args = get_conversion_args(utat)
    rt_results = benchmark_axistensor_conversions(args, FT, ref_transform)
    ot_results = benchmark_axistensor_conversions(args, FT, ref_transform) # (opt_transform)
    for key in keys(rt_results)
        @test last(ot_results[key]) == last(rt_results[key])            # test correctness
        @test_broken first(ot_results[key])*30 < first(rt_results[key]) # test performance
    end
end

function test_optimized_project(::Type{FT}) where {FT}
    @info "Testing optimized project..."
    upat = used_project_arg_types(FT)
    args = get_conversion_args(upat)
    rp_results = benchmark_axistensor_conversions(args, FT, ref_project)
    op_results = benchmark_axistensor_conversions(args, FT, ref_project) # (opt_project)
    for key in keys(rp_results)
        @test last(op_results[key]) == last(rp_results[key])            # test correctness
        @test_broken first(op_results[key])*30 < first(rp_results[key]) # test performance
    end
end

# TODO: figure out how to make error checking in `transform`

# @testset "Test optimized transform" begin
#     test_optimized_transform(Float64)
# end

@testset "Test optimized project" begin
    test_optimized_project(Float64)
end

#! format: on
