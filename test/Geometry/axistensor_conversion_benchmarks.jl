using Test, StaticArrays
import Random, BenchmarkTools, StatsBase, OrderedCollections, LinearAlgebra, Combinatorics
#! format: off
using ClimaCore.Geometry:Geometry, AbstractAxis, CovariantAxis,
    AxisVector, ContravariantAxis, LocalAxis, CartesianAxis, AxisTensor,
    Covariant1Vector, Covariant13Vector, UVVector, UWVector, UVector,
    WVector, Covariant12Vector, UVWVector, Covariant123Vector, Covariant3Vector,
    Contravariant12Vector, Contravariant3Vector, Contravariant123Vector,
    Contravariant13Vector, Contravariant2Vector, Axis2Tensor, Contravariant3Axis,
    LocalGeometry, CovariantTensor, CartesianTensor, LocalTensor, ContravariantTensor

include("transform_project.jl") # compact, generic but unoptimized reference
include("used_transform_args.jl")
include("ref_funcs.jl")
include("used_project_args.jl")

# time_func(func::F, x, y) where {F} = time_func_accurate(func, x, y)
time_func(func::F, x, y) where {F} = time_func_fast(func, x, y)

function time_func_accurate(func::F, x, y) where {F}
    b = BenchmarkTools.@benchmarkable $func($x, $y)
    trial = BenchmarkTools.run(b)
    # show(stdout, MIME("text/plain"), trial)
    time = StatsBase.mean(trial.times)
    return time
end

function time_func_fast(func::F, x, y) where {F}
    time = 0
    nsamples = 20
    for i in 1:nsamples
        time += @elapsed func(x, y)
    end
    ns = 1e9
    time = time/nsamples*ns # average and convert to ns
    return time
end
precent_speedup(opt, ref) = trunc((opt.time-ref.time)/ref.time*100, digits=0)

function print_colored(percentspeedup)
    color = percentspeedup < 0 ? :green : :red
    printstyled(percentspeedup; color)
end
function benchmark_func(args, key, func)
    ref_func = reference_func(func)
    # Reference
    result = ref_func(args...) # compile first
    time = time_func(ref_func, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    ref = (;time, t_pretty, result)
    # Optimized
    result = func(args...) # compile first
    time = time_func(func, args...)
    t_pretty = BenchmarkTools.prettytime(time)
    opt = (;time, t_pretty, result)

    percentspeedup = precent_speedup(opt, ref)

    # if abs(percentspeedup) > 3 # only print significant changes
        # @info "Benchmark: (%speedup, opt, ref): ($(percentspeedup), $(opt.t_pretty), $(ref.t_pretty)). Key: $key"
        print("Benchmark: (%speedup, opt, ref): (")
        print_colored(percentspeedup)
        print(", $(opt.t_pretty), $(ref.t_pretty)). Key: $key\n")
    # end
    return (;opt, ref)
end

benchmark_conversion(arg_set, func) =
    benchmark_conversion_serial(arg_set, func)
    # benchmark_conversion_parallel(arg_set, func)

function benchmark_conversion_serial(arg_set, func)
    results = OrderedCollections.OrderedDict()
    map(arg_set) do args
        results[typeof.(args)] = benchmark_func(args, typeof.(args), func)
    end
    return results
end

function benchmark_conversion_parallel(arg_set, func)
    # asyncmap(arg_set) do args
    #     results[typeof.(args)] = benchmark_func(args, typeof.(args), func)
    # end
    results = asyncmap(arg_set; ntasks=20) do args
        benchmark_func(args, typeof.(args), func)
    end

    results_dict = OrderedCollections.OrderedDict()
    for x in results
        results_dict[typeof.(arg_set[i])] = results[i]
    end
    return results_dict
end

function all_axes()
    all_Is() = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    collect(Iterators.flatten(map(all_Is()) do I
        (
            CovariantAxis{I}(),
            ContravariantAxis{I}(),
            LocalAxis{I}(),
            CartesianAxis{I}()
        )
    end))
end

function all_axistensors(::Type{FT}) where {FT}
    axis_vecs1 = map(all_axes()) do ax
        sv = @SVector rand(FT, 1)
        AxisTensor((ax,), sv)
    end
    axis_vecs2 = map(all_axes()) do ax
        sv = @SVector rand(FT, 2)
        AxisTensor((ax,), sv)
    end
    axis_vecs3 = map(all_axes()) do ax
        sv = @SVector rand(FT, 3)
        AxisTensor((ax,), sv)
    end
    at(m,n) = map(Combinatorics.combinations(all_axes(), 2)) do ax_combo
        sm = @SMatrix rand(FT, m,n )
        AxisTensor(Tuple(ax_combo), sm)
    end
    axis_tensors = vcat(at(1,1), at(1,2), at(1,3),
                        at(2,1), at(2,2), at(2,3),
                        at(3,1), at(3,2), at(3,3))
    aat = vcat(axis_vecs1, axis_vecs2, axis_vecs3, axis_tensors)
    println("Number of all plausible axis tensors: $(length(aat))")
    return aat
end

all_observed_axistensors(::Type{FT}) where {FT} =
    vcat(map(x-> rand(last(x)), used_project_arg_types(FT)),
         map(x-> rand(last(x)), used_transform_arg_types(FT)))

function all_possible_func_args(FT, ::typeof(Geometry.contravariant3))
    # TODO: this is not accurate yet, since we don't yet
    # vary over all possible LocalGeometry's.
    M = @SMatrix [
        FT(4) FT(1)
        FT(0.5) FT(2)
    ]
    J = LinearAlgebra.det(M)
    ∂x∂ξ = rand(Geometry.AxisTensor{FT, 2, Tuple{Geometry.LocalAxis{(3,)}, Geometry.CovariantAxis{(3,)}}, SMatrix{1, 1, FT, 1}})
    lg = Geometry.LocalGeometry(Geometry.XYPoint(FT(0), FT(0)), J, J, ∂x∂ξ)
    # Geometry.LocalGeometry{(3,), Geometry.ZPoint{FT}, FT, SMatrix{1, 1, FT, 1}} 
    map(all_axistensors(FT)) do (at)
        (at, lg)
    end
end

function all_possible_func_args(FT, ::typeof(Geometry.project))
    # TODO: this is not accurate yet, since we don't yet
    # vary over all possible LocalGeometry's.
    M = @SMatrix [
        FT(4) FT(1)
        FT(0.5) FT(2)
    ]
    J = LinearAlgebra.det(M)
    ∂x∂ξ = rand(Geometry.AxisTensor{FT, 2, Tuple{Geometry.LocalAxis{(3,)}, Geometry.CovariantAxis{(3,)}}, SMatrix{1, 1, FT, 1}})
    lg = Geometry.LocalGeometry(Geometry.XYPoint(FT(0), FT(0)), J, J, ∂x∂ξ)
    # Geometry.LocalGeometry{(3,), Geometry.ZPoint{FT}, FT, SMatrix{1, 1, FT, 1}} 
    Iterators.flatten(map(all_axistensors(FT)) do (at)
        map(all_axes()) do ax
            (at, lg)
        end
    end)
end

function func_args(FT, f::typeof(Geometry.project))
    apfa = all_possible_func_args(FT, f)
    @info "Number of all possible args for $f: $(length(apfa))"
    @info "Filtering applicable methods (this can take a while)..."
    args_dict = Dict()
    for _args in apfa
        hasmethod(f, typeof(_args)) || continue
        try
            f(_args...)
            args_dict[typeof.(_args)] = _args
        catch
        end
        # args_dict[typeof.(_args)] = _args
    end
    @info "Finished filtering"
    println("Number of applicable methods: $(length(args_dict))")
    @assert length(args_dict) == 2474
    error("Success!")
    return values(args_dict)
    # return collect(values(args_dict))[1:10]
    # map(used_project_arg_types(FT)) do (axt, axtt)
    #     (axt(), rand(axtt))
    # end
end
func_args(FT, ::typeof(Geometry.transform)) =
    map(used_transform_arg_types(FT)) do (axt, axtt)
        (axt(), rand(axtt))
    end

function func_args(FT, f::typeof(Geometry.contravariant3))
    # TODO: fix this..
    apfa = all_possible_func_args(FT, f)
    @info "Number of all possible args for $f: $(length(apfa))"
    @info "Filtering applicable methods (this can take a while)..."
    args_dict = Dict()
    for _args in apfa
        hasmethod(f, typeof(_args)) || continue
        try
            f(_args...)
            args_dict[typeof.(_args)] = _args
        catch
        end
        # args_dict[typeof.(_args)] = _args
    end
    @info "Finished filtering"
    println("Number of applicable methods: $(length(args_dict))")
    @assert length(args_dict) == 2474
    error("Success!")
    return values(args_dict)
    # return collect(values(args_dict))[1:10]
end

reference_func(::typeof(Geometry.contravariant3)) = ref_contravariant3
reference_func(::typeof(Geometry.project)) = ref_project
reference_func(::typeof(Geometry.transform)) = ref_transform

function keys_expected_to_pass(::Type{FT}, f::typeof(Geometry.project)) where {FT}
    return Dict(
        # AxisTensor{FT, 2, Tuple{CovariantAxis{(3,)}, CovariantAxis{(1, 2)}}, SMatrix{1, 2, FT, 2}} => true,
    )
end
function keys_expected_to_pass(::Type{FT}, f::typeof(Geometry.contravariant3)) where {FT}
    return Dict(
        # AxisTensor{FT, 2, Tuple{CovariantAxis{(3,)}, CovariantAxis{(1, 2)}}, SMatrix{1, 2, FT, 2}} => true,
    )
end

function test_optimized_function(::Type{FT}, func) where {FT}
    @info "Testing optimized $func..."
    args = func_args(FT, func)
    bm = @time benchmark_conversion(args, func)
    bm_results = OrderedCollections.OrderedDict()

    # Correctness comparisons
    components(x::T) where {T <: Real} = x
    components(x) = Geometry.components(x)
    compare(x::T, y::T) where {T<: Real} = x ≈ y || (x < eps(T)/100 && y < eps(T)/100)
    compare(x::T, y::T) where {T <: SMatrix} = all(compare.(x, y))
    compare(x::T, y::T) where {T <: SVector} = all(compare.(x, y))
    compare(x::T, y::T) where {T <: AxisTensor} = compare(components(x), components(y))

    expected_perf_pass = keys_expected_to_pass(FT, func)
    for key in keys(bm)
        bm_results[key] = (;
            correctness = compare(bm[key].opt.result, bm[key].ref.result), # test correctness
            perf_pass =  (bm[key].opt.time - bm[key].ref.time)/bm[key].ref.time*100 < -50, # test performance
        )
    end
    bmr = values(bm_results)
    if !all(getproperty.(bmr, :correctness))
        for key in keys(bm_results)
            bm_results[key].correctness && continue
            @info "Correctness failure.  (opt, ref): ($(bm[key].opt.result), $(bm[key].ref.result)). key: $key"
        end
    end
    if any(getproperty.(bmr, :perf_pass))
        for key in keys(bm_results)
            success_expected = get(expected_perf_pass, key, false)
            @show success_expected
            success_expected && continue
            key_str = replace("$key", "Float64" => "FT", "Float32" => "FT")
            print("Unexpected perf pass. (speedup, opt, ref): (")
            print_colored(precent_speedup(bm[key].opt, bm[key].ref))
            print("$(bm[key].opt.t_pretty), $(bm[key].ref.t_pretty)). key: $key_str\n")
        end
    end
    if !all(getproperty.(bmr, :correctness))
        error("Error in optimization (correctness) tests for $func. See above for results")
    end
    expected_pass_perf_tests = getproperty.(Ref(bm_results), keys(expected_perf_pass))
    if !all(getproperty.(expected_pass_perf_tests, :perf_pass))
        error("Error in optimization (performance) tests for $func. See above for results")
    end

    expected_perf_fail_keys = setdiff(keys(bm_results), keys(expected_perf_pass))
    for key in keys(expected_perf_pass)
        @test bm_results[key].perf_pass
    end
    for key in expected_perf_fail_keys
        @test_broken bm_results[key].perf_pass
    end

end

# TODO: figure out how to make error checking in `transform` optional

@testset "Test optimized functions" begin
    test_optimized_function(Float64, Geometry.project)
    # test_optimized_function(Float64, Geometry.contravariant3)
    # test_optimized_function(Float64, Geometry.transform)
end

#! format: on
