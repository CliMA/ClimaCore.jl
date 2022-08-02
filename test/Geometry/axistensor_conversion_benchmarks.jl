using Test, StaticArrays
#! format: off
import Random, BenchmarkTools, StatsBase,
    OrderedCollections, LinearAlgebra, Combinatorics
using ClimaCore.Geometry:Geometry, AbstractAxis, CovariantAxis,
    AxisVector, ContravariantAxis, LocalAxis, CartesianAxis, AxisTensor,
    Covariant1Vector, Covariant13Vector, UVVector, UWVector, UVector,
    WVector, Covariant12Vector, UVWVector, Covariant123Vector, Covariant3Vector,
    Contravariant12Vector, Contravariant3Vector, Contravariant123Vector,
    Contravariant13Vector, Contravariant2Vector, Axis2Tensor, Contravariant3Axis,
    LocalGeometry, CovariantTensor, CartesianTensor, LocalTensor, ContravariantTensor,
    XZPoint, XYZPoint, LatLongZPoint, XYPoint, ZPoint, LatLongPoint, XPoint,
    Contravariant1Axis, Contravariant2Axis

include("ref_funcs.jl") # compact, generic but unoptimized reference
include("func_arg_types.jl")
include("func_args.jl")

# time_func(func::F, args...) where {F} = time_func_accurate(func, args...)
time_func(f::F, args...) where {F} = time_func_fast(f, args...)

function time_func_accurate(f::F, x, y) where {F}
    b = BenchmarkTools.@benchmarkable $f($x, $y)
    trial = BenchmarkTools.run(b)
    # show(stdout, MIME("text/plain"), trial)
    time = StatsBase.mean(trial.times)
    return time
end

function time_func_accurate(f::F, x, y, z) where {F}
    b = BenchmarkTools.@benchmarkable $f($x, $y, $z)
    trial = BenchmarkTools.run(b)
    # show(stdout, MIME("text/plain"), trial)
    time = StatsBase.mean(trial.times)
    return time
end

function time_func_fast(f::F, x, y) where {F}
    time = Float64(0)
    nsamples = 100
    for i in 1:nsamples
        time += @elapsed f(x, y)
    end
    ns = Float64(1e9)
    time = time/nsamples*ns # average and convert to ns
    return time
end

function time_func_fast(f::F, x, y, z) where {F}
    time = Float64(0)
    nsamples = 100
    for i in 1:nsamples
        time += @elapsed f(x, y, z)
    end
    ns = Float64(1e9)
    time = time/nsamples*ns # average and convert to ns
    return time
end
precent_speedup(opt, ref) = trunc((opt.time-ref.time)/ref.time*100, digits=0)
function print_colored(percentspeedup)
    color = percentspeedup < 0 ? :green : :red
    printstyled(percentspeedup; color)
end

function benchmark_func(args, key, f)
    f_ref = reference_func(f)
    # Reference
    result = f_ref(args...) # compile first
    time = time_func(f_ref, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    ref = (;time, t_pretty, result)
    # Optimized
    result = f(args...) # compile first
    time = time_func(f, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    opt = (;time, t_pretty, result)

    percentspeedup = precent_speedup(opt, ref)

    # if abs(percentspeedup) > 3 # only print significant changes
        print("Benchmark: (%speedup, opt, ref): (")
        print_colored(percentspeedup)
        print(", $(opt.t_pretty), $(ref.t_pretty)). Key: $key\n")
    # end
    bm = (;
        opt,
        ref,
        correctness = compare(opt.result, ref.result), # test correctness
        perf_pass = (opt.time - ref.time)/ref.time*100 < -50, # test performance
    )
    return bm
end

dict_key(f, args) = (nameof(f), typeof.(args)...)

# expensive/slow. Can we (safely) parallelize this?
function benchmark_conversions!(benchmarks, all_args, f)
    for args in all_args
        benchmarks[dict_key(f, args)] = benchmark_func(args, dict_key(f, args), f)
    end
    return nothing
end

reference_func(::typeof(Geometry.contravariant1)) = ref_contravariant1
reference_func(::typeof(Geometry.contravariant2)) = ref_contravariant2
reference_func(::typeof(Geometry.contravariant3)) = ref_contravariant3
reference_func(::typeof(Geometry.project)) = ref_project
reference_func(::typeof(Geometry.transform)) = ref_transform

# Correctness comparisons
components(x::T) where {T <: Real} = x
components(x) = Geometry.components(x)
compare(x::T, y::T) where {T<: Real} = x ≈ y || (x < eps(T)/100 && y < eps(T)/100)
compare(x::T, y::T) where {T <: SMatrix} = all(compare.(x, y))
compare(x::T, y::T) where {T <: SVector} = all(compare.(x, y))
compare(x::T, y::T) where {T <: AxisTensor} = compare(components(x), components(y))

function test_optimized_functions(::Type{FT}) where {FT}
    benchmarks = OrderedCollections.OrderedDict()
    for f in (
        Geometry.project,
        Geometry.transform,
        Geometry.contravariant1,
        Geometry.contravariant2,
        Geometry.contravariant3,
    )
        @info "Testing optimized $f..."
        all_args = func_args(FT, f)
        benchmark_conversions!(benchmarks, all_args, f)
    end

    # TODO: add mechanism for expected pass
    for key in keys(benchmarks)
        @test benchmarks[key].correctness      # test correctness
        @test_broken benchmarks[key].perf_pass # test performance
    end
end

# TODO: figure out how to make error checking in `transform` optional

@testset "Test optimized functions" begin
    test_optimized_functions(Float64)
end

#! format: on
