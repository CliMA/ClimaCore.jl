#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "indexing_and_static_ndranges.jl"))

# Info:
This script compares two things:
 - linear vs cartesian indexing
 - impact of static vs dynamic NDRanges (https://juliagpu.github.io/KernelAbstractions.jl/dev/examples/memcopy_static/)

Linear indexing, when possible, has performance advantages
over using Cartesian indexing. Julia Base's Broadcast only
supports Cartesian indexing as it provides more general support
for "extruded"-style broadcasting, where shapes of input/output
arrays can change.

This script (re-)defines some broadcast machinery and tests
the performance of vector vs array operations in a broadcast
setting where linear indexing is allowed.

# Summary:
 - On the CPU:
    static NDRanges do not play an important role,
    but linear indexing is 2x faster than cartesian
    indexing.
 - On the GPU:
    static NDRanges DO play an important role,
    but we could (alternatively) see an improvement
    by using linear indexing. Supporting StaticNDRanges
    also impacts non-pointwise kernels, and yields
    nearly the same benefit as linear indexing.

# References:
 - https://githubSR.com/CliMA/ClimaCore.jl/issues/1889
 - https://githubSR.com/JuliaLang/julia/issues/28126
 - https://githubSR.com/JuliaLang/julia/issues/32051

# Benchmark results:

Clima A100:
```
[ Info: ArrayType = identity
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                                      │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│ BSR.at_dot_call!(X_array, Y_array; nreps=1000, bm)                         │ 422 microseconds, 223 nanoseconds │ 2.35535 │ 48.0256     │ 1              │ 1000   │
│ BSR.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)                       │ 242 microseconds, 740 nanoseconds │ 4.09692 │ 83.5362     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, us; nreps=1000, bm)              │ 242 microseconds, 30 nanoseconds  │ 4.10894 │ 83.7812     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, uss; nreps=1000, bm)             │ 244 microseconds, 279 nanoseconds │ 4.0711  │ 83.0097     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false, nreps=1000, bm)  │ 499 microseconds, 283 nanoseconds │ 1.99182 │ 40.6133     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false, nreps=1000, bm) │ 541 microseconds, 506 nanoseconds │ 1.83651 │ 37.4465     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true, nreps=1000, bm)   │ 247 microseconds, 108 nanoseconds │ 4.02449 │ 82.0593     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true, nreps=1000, bm)  │ 242 microseconds, 209 nanoseconds │ 4.10589 │ 83.7192     │ 1              │ 1000   │
└────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
[ Info: ArrayType = identity
Problem size: (63, 4, 4, 1, 5400), float_type = Float64, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                                      │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│ BSR.at_dot_call!(X_array, Y_array; nreps=1000, bm)                         │ 1 millisecond, 446 microseconds   │ 1.37517 │ 28.0397     │ 1              │ 1000   │
│ BSR.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)                       │ 984 microseconds, 854 nanoseconds │ 2.01955 │ 41.1787     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, us; nreps=1000, bm)              │ 987 microseconds, 438 nanoseconds │ 2.01427 │ 41.0709     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, uss; nreps=1000, bm)             │ 985 microseconds, 779 nanoseconds │ 2.01766 │ 41.1401     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false, nreps=1000, bm)  │ 1 millisecond, 475 microseconds   │ 1.34834 │ 27.4927     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false, nreps=1000, bm) │ 1 millisecond, 473 microseconds   │ 1.34985 │ 27.5234     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true, nreps=1000, bm)   │ 983 microseconds, 811 nanoseconds │ 2.0217  │ 41.2224     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true, nreps=1000, bm)  │ 984 microseconds, 683 nanoseconds │ 2.0199  │ 41.1858     │ 1              │ 1000   │
└────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```

Clima A100
```
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                                       │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│ BSR.at_dot_call!(X_array, Y_array; nreps=1000, bm)                          │ 68 microseconds, 641 nanoseconds │ 14.4882 │ 295.415     │ 1              │ 1000   │
│ BSR.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)                        │ 13 microseconds, 787 nanoseconds │ 72.1366 │ 1470.86     │ 1              │ 1000   │
│ iscpu || BSR.custom_sol_kernel!(X_vector, Y_vector, Val(N); nreps=1000, bm) │ 12 microseconds, 925 nanoseconds │ 76.943  │ 1568.87     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, us; nreps=1000, bm)               │ 13 microseconds, 364 nanoseconds │ 74.4195 │ 1517.41     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, uss; nreps=1000, bm)              │ 12 microseconds, 929 nanoseconds │ 76.9247 │ 1568.49     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false, nreps=1000, bm)   │ 41 microseconds, 5 nanoseconds   │ 24.2533 │ 494.525     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false, nreps=1000, bm)  │ 26 microseconds, 652 nanoseconds │ 37.3141 │ 760.835     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true, nreps=1000, bm)    │ 13 microseconds, 582 nanoseconds │ 73.2243 │ 1493.04     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true, nreps=1000, bm)   │ 12 microseconds, 922 nanoseconds │ 76.9613 │ 1569.24     │ 1              │ 1000   │
└─────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 1, 5400), float_type = Float64, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                                       │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│ BSR.at_dot_call!(X_array, Y_array; nreps=1000, bm)                          │ 69 microseconds, 10 nanoseconds  │ 28.8217 │ 587.673     │ 1              │ 1000   │
│ BSR.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)                        │ 28 microseconds, 219 nanoseconds │ 70.4848 │ 1437.18     │ 1              │ 1000   │
│ iscpu || BSR.custom_sol_kernel!(X_vector, Y_vector, Val(N); nreps=1000, bm) │ 25 microseconds, 460 nanoseconds │ 78.1221 │ 1592.91     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, us; nreps=1000, bm)               │ 25 microseconds, 625 nanoseconds │ 77.6194 │ 1582.66     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_vector, Y_vector, uss; nreps=1000, bm)              │ 25 microseconds, 436 nanoseconds │ 78.1975 │ 1594.45     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false, nreps=1000, bm)   │ 41 microseconds, 621 nanoseconds │ 47.7881 │ 974.4       │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false, nreps=1000, bm)  │ 27 microseconds, 111 nanoseconds │ 73.3654 │ 1495.92     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true, nreps=1000, bm)    │ 25 microseconds, 931 nanoseconds │ 76.703  │ 1563.97     │ 1              │ 1000   │
│ BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true, nreps=1000, bm)   │ 25 microseconds, 464 nanoseconds │ 78.1095 │ 1592.65     │ 1              │ 1000   │
└─────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```
=#

#! format: off

module IndexStaticRangeBench

include("benchmark_utils.jl")

# ============================================================ Non-extruded broadcast (start)
import Base.Broadcast: BroadcastStyle
struct PointWiseBC{
    Style <: Union{Nothing, BroadcastStyle},
    Axes,
    F,
    Args <: Tuple,
} <: Base.AbstractBroadcasted
    style::Style
    f::F
    args::Args
    axes::Axes          # the axes of the resulting object (may be bigger than implied by `args` if this is nested inside a larger `PointWiseBC`)

    PointWiseBC(style::Union{Nothing, BroadcastStyle}, f::Tuple, args::Tuple) =
        error() # disambiguation: tuple is not callable
    function PointWiseBC(
        style::Union{Nothing, BroadcastStyle},
        f::F,
        args::Tuple,
        axes = nothing,
    ) where {F}
        # using Core.Typeof rather than F preserves inferrability when f is a type
        return new{typeof(style), typeof(axes), Core.Typeof(f), typeof(args)}(
            style,
            f,
            args,
            axes,
        )
    end
    function PointWiseBC(f::F, args::Tuple, axes = nothing) where {F}
        PointWiseBC(combine_styles(args...)::BroadcastStyle, f, args, axes)
    end
    function PointWiseBC{Style}(f::F, args, axes = nothing) where {Style, F}
        return new{Style, typeof(axes), Core.Typeof(f), typeof(args)}(
            Style()::Style,
            f,
            args,
            axes,
        )
    end
    function PointWiseBC{Style, Axes, F, Args}(
        f,
        args,
        axes,
    ) where {Style, Axes, F, Args}
        return new{Style, Axes, F, Args}(Style()::Style, f, args, axes)
    end
end

import Adapt
import CUDA
function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    bc::PointWiseBC{Style},
) where {Style}
    PointWiseBC{Style}(
        Adapt.adapt(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )
end

@inline to_pointwise_bc(bc::Base.Broadcast.Broadcasted) =
    PointWiseBC(bc.style, bc.f, bc.args, bc.axes)
@inline to_pointwise_bc(x) = x
PointWiseBC(bc::Base.Broadcast.Broadcasted) = to_pointwise_bc(bc)

@inline to_pointwise_bc_args(args::Tuple, inds...) = (
    to_pointwise_bc(args[1], inds...),
    to_pointwise_bc_args(Base.tail(args), inds...)...,
)
@inline to_pointwise_bc_args(args::Tuple{Any}, inds...) =
    (to_pointwise_bc(args[1], inds...),)
@inline to_pointwise_bc_args(args::Tuple{}, inds...) = ()

@inline function to_pointwise_bc(bc::Base.Broadcast.Broadcasted, symb, axes)
    Base.Broadcast.Broadcasted(
        bc.f,
        to_pointwise_bc_args(bc.args, symb, axes),
        axes,
    )
end
@inline to_pointwise_bc(x, symb, axes) = x

@inline function Base.getindex(
    bc::PointWiseBC,
    I::Union{Integer, CartesianIndex},
)
    @boundscheck Base.checkbounds(bc, I) # is this really the only issue?
    @inbounds _broadcast_getindex(bc, I)
end
Base.@propagate_inbounds _broadcast_getindex(
    A::Union{Ref, AbstractArray{<:Any, 0}, Number},
    I::Integer,
) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds _broadcast_getindex(
    ::Ref{Type{T}},
    I::Integer,
) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds _broadcast_getindex(A::Tuple{Any}, I::Integer) = A[1]
Base.@propagate_inbounds _broadcast_getindex(A::Tuple, I::Integer) = A[I[1]]
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
# Base.@propagate_inbounds _broadcast_getindex(A, I) = A[newindex(A, I)]
Base.@propagate_inbounds _broadcast_getindex(A, I::Integer) = A[I]
Base.@propagate_inbounds function _broadcast_getindex(
    bc::PointWiseBC{<:Any, <:Any, <:Any, <:Any},
    I::Integer,
)
    args = _getindex(bc.args, I)
    return _broadcast_getindex_evalf(bc.f, args...)
end
@inline _broadcast_getindex_evalf(f::Tf, args::Vararg{Any, N}) where {Tf, N} =
    f(args...)  # not propagate_inbounds
Base.@propagate_inbounds _getindex(args::Tuple, I) =
    (_broadcast_getindex(args[1], I), _getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _getindex(args::Tuple{Any}, I) =
    (_broadcast_getindex(args[1], I),)
Base.@propagate_inbounds _getindex(args::Tuple{}, I) = ()

@inline Base.axes(bc::PointWiseBC) = _axes(bc, bc.axes)
_axes(::PointWiseBC, axes::Tuple) = axes
@inline _axes(bc::PointWiseBC, ::Nothing) =
    Base.Broadcast.combine_axes(bc.args...)
_axes(bc::PointWiseBC{<:Base.Broadcast.AbstractArrayStyle{0}}, ::Nothing) = ()
@inline Base.axes(bc::PointWiseBC{<:Any, <:NTuple{N}}, d::Integer) where {N} =
    d <= N ? axes(bc)[d] : OneTo(1)
Base.IndexStyle(::Type{<:PointWiseBC{<:Any, <:Tuple{Any}}}) = IndexLinear()
# ============================================================ Non-extruded broadcast (end)

myadd(x1, x2, x3) = zero(x1)
function at_dot_call!(X, Y; nreps = 1, bm=nothing, n_trials = 30)
    (; x1, x2, x3) = X
    (; y1) = Y
    @. y1 = myadd(x1, x2, x3) # compile
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps # reduce variance / impact of launch latency
                @. y1 = myadd(x1, x2, x3) # 3 reads, 1 write
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=1)
    return nothing
end;

function custom_sol_kernel!(X, Y, ::Val{N}; nreps = 1, bm=nothing, n_trials = 30) where {N}
    (; x1, x2, x3) = X
    (; y1) = Y
    kernel = CUDA.@cuda always_inline = true launch = false custom_kernel_knl!(
        y1,
        x1,
        x2,
        x3,
        Val(N),
    )
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    kernel(y1, x1, x2, x3, Val(N); threads, blocks) # compile
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps # reduce variance / impact of launch latency
                kernel(y1, x1, x2, x3, Val(N); threads, blocks)
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=1)

    return nothing
end;
function custom_kernel_knl!(y1, x1, x2, x3, ::Val{N}) where {N}
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ N
            y1[I] = myadd(x1[I], x2[I], x3[I])
        end
    end
    return nothing
end;

function custom_kernel_bc!(X, Y, us::AbstractUniversalSizes; printtb=false, use_pw=true, nreps = 1, bm=nothing, n_trials = 30)
    (; x1, x2, x3) = X
    (; y1) = Y
    bc_base = @lazy @. y1 = myadd(x1, x2, x3)
    bc = use_pw ? to_pointwise_bc(bc_base) : bc_base
    e = Inf
    if y1 isa Array
        if bc isa Base.Broadcast.Broadcasted
            for t in 1:n_trials
                et = Base.@elapsed begin
                    for i in 1:nreps # reduce variance / impact of launch latency
                        @inbounds @simd for j in eachindex(bc)
                            y1[j] = bc[j]
                        end
                    end
                end
                e = min(e, et)
            end
        else
            for t in 1:n_trials
                et = Base.@elapsed begin
                    for i in 1:nreps # reduce variance / impact of launch latency
                        @inbounds @simd for j in 1:get_N(us)
                            y1[j] = bc[j]
                        end
                    end
                end
                e = min(e, et)
            end
        end
    else
        kernel =
            CUDA.@cuda always_inline = true launch = false custom_kernel_knl_bc!(
                y1,
                bc,
                us,
            )
        N = get_N(us)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(N, config.threads)
        blocks = cld(N, threads)
        printtb && @show blocks, threads
        kernel(y1, bc,us; threads, blocks) # compile
        for t in 1:n_trials
            et = CUDA.@elapsed begin
                for i in 1:nreps # reduce variance / impact of launch latency
                    kernel(y1, bc,us; threads, blocks)
                end
            end
            e = min(e, et)
        end
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=1)
    return nothing
end;
@inline get_cart_lin_index(bc, n, I) = I
@inline get_cart_lin_index(bc::Base.Broadcast.Broadcasted, n, I) =
    CartesianIndices(map(x -> Base.OneTo(x), n))[I]
function custom_kernel_knl_bc!(y1, bc, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if 1 ≤ I ≤ get_N(us)
            n = (get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us))
            ci = get_cart_lin_index(bc, n, I)
            y1[ci] = bc[ci]
        end
    end
    return nothing
end;

end # module
import .IndexStaticRangeBench as BSR

using CUDA
using Test
bm = BSR.Benchmark(;problem_size=(63,4,4,1,5400), float_type=Float32)
# bm = BSR.Benchmark(;problem_size=(63,4,4,1,5400), float_type=Float64)
ArrayType = CUDA.CuArray;
# ArrayType = Base.identity;

us_tup = (1, 2, 3)
@test BSR.get_Nv(BSR.UniversalSizesCC(us_tup...))  == BSR.get_Nv(BSR.UniversalSizesStatic(us_tup...))
@test BSR.get_Nij(BSR.UniversalSizesCC(us_tup...)) == BSR.get_Nij(BSR.UniversalSizesStatic(us_tup...))
@test BSR.get_Nh(BSR.UniversalSizesCC(us_tup...))  == BSR.get_Nh(BSR.UniversalSizesStatic(us_tup...))
@test BSR.get_N(BSR.UniversalSizesCC(us_tup...))   == BSR.get_N(BSR.UniversalSizesStatic(us_tup...))

arr(bm, T) = T(zeros(bm.float_type, bm.problem_size...))
X_array = (;x1 = arr(bm, ArrayType),x2 = arr(bm, ArrayType),x3 = arr(bm, ArrayType));
Y_array = (;y1 = arr(bm, ArrayType),);
to_vec(ξ) = (;zip(propertynames(ξ), map(θ -> vec(θ), values(ξ)))...);
X_vector = to_vec(X_array);
Y_vector = to_vec(Y_array);
BSR.at_dot_call!(X_array, Y_array)
BSR.at_dot_call!(X_vector, Y_vector)
N = length(X_vector.x1)
(Nv, Nij, _, Nf, Nh) = size(Y_array.y1);
us = BSR.UniversalSizesCC(Nv, Nij, Nh);
uss = BSR.UniversalSizesStatic(Nv, Nij, Nh);
@test BSR.get_N(us) == N
@test BSR.get_N(uss) == N
iscpu = ArrayType === identity
BSR.custom_kernel_bc!(X_vector, Y_vector, us)
BSR.custom_kernel_bc!(X_vector, Y_vector, uss)
iscpu || BSR.custom_sol_kernel!(X_vector, Y_vector, Val(N))

BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false)
BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false)

BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true)
BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true)

BSR.at_dot_call!(X_array, Y_array; nreps=1000, bm)
BSR.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)
iscpu || BSR.custom_sol_kernel!(X_vector, Y_vector, Val(N); nreps=1000, bm)

BSR.custom_kernel_bc!(X_vector, Y_vector, us; nreps=1000, bm)
BSR.custom_kernel_bc!(X_vector, Y_vector, uss; nreps=1000, bm)

BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=false, nreps=1000, bm)
BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=false, nreps=1000, bm)

BSR.custom_kernel_bc!(X_array, Y_array, us; use_pw=true, nreps=1000, bm)
BSR.custom_kernel_bc!(X_array, Y_array, uss; use_pw=true, nreps=1000, bm)

@info "ArrayType = $ArrayType"
BSR.tabulate_benchmark(bm)

#! format: on
