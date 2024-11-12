using Test, StaticArrays
#! format: off
import Random, BenchmarkTools, StatsBase,
    OrderedCollections, LinearAlgebra, CountFlops
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
include("method_info.jl")
include("func_args.jl")

count_flops(f::F, x, y) where {F} = CountFlops.@count_ops f($x, $y)
count_flops(f::F, x, y, z) where {F} = CountFlops.@count_ops f($x, $y, $z)

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
function print_colored(Δflops)
    color = Δflops < 0 ? :green :
            Δflops == 0 ? :yellow : :red
    printstyled(Δflops; color)
end

function total_flops(flops)
    n_flops = 0
    for pn in propertynames(flops)
        n_flops += getproperty(flops, pn)
    end
    return n_flops
end

function benchmark_func(args, key, f, flops, ::Type{FT}; print_method_info) where {FT}
    f_ref = reference_func(f)
    # Reference
    result = f_ref(args...) # compile first
    ref_flops = count_flops(f_ref, args...)
    time = time_func(f_ref, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    ref = (;time, t_pretty, result)
    # Optimized
    result = f(args...) # compile first
    opt_flops = count_flops(f, args...)
    time = time_func(f, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    opt = (;time, t_pretty, result)

    percentspeedup = precent_speedup(opt, ref)

    computed_flops = total_flops(opt_flops)
    reduced_flops = computed_flops < flops
    Δflops = computed_flops - flops

    if print_method_info
        key_str = replace("$key", "Float64" => "FT", "Float32" => "FT", " " => "")
        key_str = join(split(key_str, ",")[2:end], ",")[1:end-1]
        # For conveniently copying into method_info.jl:
        println("    ($key_str,$computed_flops),")
    else
        key_str = replace("$key", "Float64" => "FT", "Float32" => "FT", " " => "")
        # if reduced_flops # only print significant changes
            print("Flops (Δ, now, main): (")
            print_colored(Δflops)
            print(", $computed_flops, $flops).")
            print("Time (opt, ref): ($(opt.t_pretty), $(ref.t_pretty)). Key: $key_str\n")
        # end
    end
    bm = (;
        opt,
        ref,
        Δflops,
        opt_flops,
        flops, # current flops
        computed_flops,
        reduced_flops,
        correctness = compare(opt.result, ref.result), # test correctness
        perf_pass = (opt.time - ref.time)/ref.time*100 < -100, # test performance
    )
    return bm
end

dict_key(f, args) = (nameof(f), typeof.(args)...)

# expensive/slow. Can we (safely) parallelize this?
function benchmark_conversions!(benchmarks, f, FT; print_method_info)
    for (args, flops) in func_args(FT, f)
        benchmarks[dict_key(f, args)] = benchmark_func(args, dict_key(f, args), f, flops, FT; print_method_info)
    end
    return nothing
end

reference_func(::typeof(Geometry.contravariant1)) = ref_contravariant1
reference_func(::typeof(Geometry.contravariant2)) = ref_contravariant2
reference_func(::typeof(Geometry.contravariant3)) = ref_contravariant3
reference_func(::typeof(Geometry.Jcontravariant3)) = ref_Jcontravariant3
reference_func(::typeof(Geometry.project)) = ref_project
reference_func(::typeof(Geometry.transform)) = ref_transform

# Correctness comparisons
components(x::T) where {T <: Real} = x
components(x) = Geometry.components(x)
compare(x::T, y::T) where {T<: Real} = x ≈ y || (x < eps(T)/100 && y < eps(T)/100)
compare(x::T, y::T) where {T <: SMatrix} = all(compare.(x, y))
compare(x::T, y::T) where {T <: SVector} = all(compare.(x, y))
compare(x::T, y::T) where {T <: AxisTensor} = compare(components(x), components(y))

function test_optimized_functions(::Type{FT}; print_method_info=false) where {FT}
    @info "Testing optimized functions with $FT"
    benchmarks = OrderedCollections.OrderedDict()
    for f in (
        Geometry.project,
        Geometry.transform,
        Geometry.contravariant1,
        Geometry.contravariant2,
        Geometry.contravariant3,
        Geometry.Jcontravariant3,
    )
        @info "Testing optimized $f..."
        benchmark_conversions!(benchmarks, f, FT; print_method_info)
    end

    for key in keys(benchmarks)
        @test benchmarks[key].correctness       # test correctness
        @test benchmarks[key].Δflops ≤ 0        # Don't regress
        # @test_broken benchmarks[key].Δflops < 0 # Error on improvements. TODO: fix, this is somehow flakey
        @test_broken benchmarks[key].perf_pass  # rough timing test (benchmarking is hard for ns funcs)
    end
end

# TODO: figure out how to make error checking in `transform` optional

test_optimized_functions(Float64; print_method_info=true)
@testset "Test optimized functions" begin
    test_optimized_functions(Float64)
end


#! format: on
