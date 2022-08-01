using Test, StaticArrays
import Random, BenchmarkTools, StatsBase, OrderedCollections, LinearAlgebra
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
    trial = BenchmarkTools.run(b; samples = 3)
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

function benchmark_conversion(arg_set, ::Type{FT}, func::F) where {FT, F}
    results = OrderedCollections.OrderedDict()
    ref_func = reference_func(func)
    for args in arg_set
        key = typeof.(args)
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

        @info "Benchmark: opt: $(opt.t_pretty) ref: $(ref.t_pretty). Key: $key"
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
    Iterators.flatten(
        map(used_project_arg_types(FT)) do (axt, axtt)
            map(all_axes()) do ax
                (rand(axtt), lg)
            end
        end
    )
end

function func_args(FT, f::typeof(Geometry.contravariant3))
    # TODO: fix this..
    apfa = all_possible_func_args(FT, f)
    args_dict = Dict()
    for _args in apfa
        hasmethod(f, typeof(_args)) || continue
        args_dict[typeof.(_args)] = _args
    end
    return values(args_dict)
end

reference_func(::typeof(Geometry.contravariant3)) = ref_contravariant3
reference_func(::typeof(Geometry.project)) = ref_project
reference_func(::typeof(Geometry.transform)) = ref_transform

function test_optimized_function(::Type{FT}, func) where {FT}
    @info "Testing optimized $func..."
    args = func_args(FT, func)
    bm = benchmark_conversion(args, FT, func)
    for key in keys(bm)
        @test bm[key].opt.result == bm[key].ref.result      # test correctness
        @test_broken bm[key].opt.time*10 < bm[key].ref.time # test performance
    end
end

# TODO: figure out how to make error checking in `transform` optional

@testset "Test optimized functions" begin
    test_optimized_function(Float64, Geometry.project)
    # test_optimized_function(Float64, Geometry.contravariant3)
    # test_optimized_function(Float64, Geometry.transform)
end

#! format: on
