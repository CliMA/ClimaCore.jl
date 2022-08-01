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
    nsamples = 100
    for i in 1:nsamples
        time += @elapsed func(x, y)
    end
    ns = 1e9
    time = time/nsamples*ns # average and convert to ns
    return time
end

benchmark_conversion(arg_set, ::Type{FT}, func) where {FT} =
    benchmark_conversion(arg_set, FT, func, reference_func(func))

function benchmark_conversion(arg_set, ::Type{FT}, func::F, ref_func::RF) where {FT, F, RF}
    results = OrderedCollections.OrderedDict()
    for args in arg_set
        key = typeof.(args)
        # Reference
        result = ref_func(args...) # compile first
        result = ref_func(args...) # compile first
        time = time_func(ref_func, args...)
        t_pretty = BenchmarkTools.prettytime(time)
        ref = (;time, t_pretty, result)
        # Optimized
        result = func(args...) # compile first
        result = func(args...) # compile first
        time = time_func(func, args...)
        t_pretty = BenchmarkTools.prettytime(time)
        opt = (;time, t_pretty, result)

        percentspeedup = trunc((opt.time-ref.time)/ref.time*100, digits=0)

        if abs(percentspeedup) > 3 # only print significant changes
            # @info "Benchmark: (%speedup, opt, ref): ($(percentspeedup), $(opt.t_pretty), $(ref.t_pretty)). Key: $key"
            color = percentspeedup < 0 ? :green : :red
            print("Benchmark: (%speedup, opt, ref): (")
            printstyled(percentspeedup; color)
            print(", $(opt.t_pretty), $(ref.t_pretty)). Key: $key\n")
        end
        results[key] = (;opt, ref)
    end
    return results
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

func_args(FT, ::typeof(Geometry.project)) =
    map(used_project_arg_types(FT)) do (axt, axtt)
        (axt(), rand(axtt))
    end
func_args(FT, ::typeof(Geometry.transform)) =
    map(used_transform_arg_types(FT)) do (axt, axtt)
        (axt(), rand(axtt))
    end

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

function func_args(FT, f::typeof(Geometry.contravariant3))
    # TODO: fix this..
    apfa = all_possible_func_args(FT, f)
    args_dict = Dict()
    for _args in apfa
        try
            f(_args...)
            args_dict[typeof.(_args)] = _args
        catch
        end
        # hasmethod(f, typeof(_args)) || continue
        # args_dict[typeof.(_args)] = _args
    end
    return values(args_dict)
end

reference_func(::typeof(Geometry.contravariant3)) = ref_contravariant3
reference_func(::typeof(Geometry.project)) = ref_project
reference_func(::typeof(Geometry.transform)) = ref_transform

function keys_expected_to_pass(::Type{FT}, f::typeof(Geometry.contravariant3)) where {FT}
    ketp = [
        AxisTensor{FT, 2, Tuple{CovariantAxis{(3,)}, CovariantAxis{(1, 2)}}, SMatrix{1, 2, FT, 2}},
    ]
end

function test_optimized_function(::Type{FT}, func) where {FT}
    @info "Testing optimized $func..."
    args = func_args(FT, func)
    bm = benchmark_conversion(args, FT, func)
    bm_results = OrderedCollections.OrderedDict()
    components(x::FT) = x
    components(x) = Geometry.components(x)
    for key in keys(bm)
        bm_results[key] = (;
            correctness = all(
                components(bm[key].opt.result) .≈
                components(bm[key].ref.result)), # test correctness
            perf_pass =  (bm[key].opt.time - bm[key].ref.time)/bm[key].ref.time*100 < -50, # test performance
            expect_pass = false,
        )
    end
    # ketp = keys_expected_to_pass(FT, f)
    # for key in keys(bm_results)
    #     bm_results[key].expect_pass = haskey(keys_expected_to_pass)
    # end
    bmr = values(bm_results)
    if !all(getproperty.(bmr, :correctness))
        for key in keys(bm_results)
            bm_results[key].correctness && continue
            # TODO: silenced for now, uncomment
            # @info "Correctness failure.  (opt, ref): ($(bm[key].opt.result), $(bm[key].ref.result)). key: $key"
        end
    end
    if any(getproperty.(bmr, :perf_pass))
        for key in keys(bm_results)
            bm_results[key].perf_pass || continue
            # key in keys_expected_to_pass(FT, func) || continue
            key_str = replace("$key", "Float64" => "FT", "Float32" => "FT")
            @info "Unexpected perf pass. (opt, ref): ($(bm[key].opt.time), $(bm[key].ref.time)). key: $key_str"
        end
    end
    if !all(getproperty.(bmr, :correctness)) || !all(getproperty.(bmr, :perf_pass))
        error("Error in optimization tests for $func. See above for results")
    end
end

# TODO: figure out how to make error checking in `transform` optional

@testset "Test optimized functions" begin
    # test_optimized_function(Float64, Geometry.project)
    test_optimized_function(Float64, Geometry.contravariant3)
    # test_optimized_function(Float64, Geometry.transform)
end

#! format: on
