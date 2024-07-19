#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "linear_vs_cartesian_indexing.jl"))

# Info:
Linear indexing, when possible, has performance advantages
over using Cartesian indexing. Julia Base's Broadcast only
supports Cartesian indexing as it provides more general support
for "extruded"-style broadcasting, where shapes of input/output
arrays can change.

This script (re-)defines some broadcast machinery and tests
the performance of vector vs array operations in a broadcast
setting where linear indexing is allowed.

# References:
 - https://github.com/CliMA/ClimaCore.jl/issues/1889
 - https://github.com/JuliaLang/julia/issues/28126
 - https://github.com/JuliaLang/julia/issues/32051

# Benchmark results:

Local Apple M1 Mac (CPU):
```
at_dot_call!($X_array, $Y_array):
     146 milliseconds, 558 microseconds
at_dot_call!($X_vector, $Y_vector):
     65 milliseconds, 531 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $(Val(length(X_vector.x1))); printtb = false):
     66 milliseconds, 735 microseconds
custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1))); printtb = false, use_pw = false):
     145 milliseconds, 957 microseconds
custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1))); printtb = false, use_pw = true):
     66 milliseconds, 320 microseconds
```

Clima A100
```
at_dot_call!($X_vector, $Y_vector):
     2 milliseconds, 848 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $(Val(length(X_vector.x1))); printtb = false):
     2 milliseconds, 537 microseconds
custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1))); printtb = false, use_pw = false):
     8 milliseconds, 804 microseconds
custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1))); printtb = false, use_pw = true):
     2 milliseconds, 545 microseconds
```
=#

#! format: off
import CUDA
using BenchmarkTools, Dates
using LazyBroadcast: @lazy
ArrayType = CUDA.CuArray;
# ArrayType = identity;

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

if ArrayType === identity
    macro pretty_belapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(BenchmarkTools.@belapsed(esc($expr)))
        end
    end
    macro pretty_elapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(BenchmarkTools.@elapsed(esc($expr)))
        end
    end
else
    macro pretty_belapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(
                BenchmarkTools.@belapsed(CUDA.@sync((esc($expr))))
            )
        end
    end
    macro pretty_elapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(
                BenchmarkTools.@elapsed(CUDA.@sync((esc($expr))))
            )
        end
    end
end
print_time_and_units(x) = println(time_and_units_str(x))
time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end
trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s
myadd(x1, x2, x3) = zero(x1)
function at_dot_call!(X, Y)
    (; x1, x2, x3) = X
    (; y1) = Y
    for i in 1:100 # reduce variance / impact of launch latency
        @. y1 = myadd(x1, x2, x3) # 3 reads, 1 write
        # @. y1 = 0 # 3 reads, 1 write
    end
    return nothing
end;

function custom_kernel!(X, Y, ::Val{N}) where {N}
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
    for i in 1:100 # reduce variance / impact of launch latency
        kernel(y1, x1, x2, x3, Val(N); threads, blocks)
    end
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

function custom_kernel_bc!(X, Y, ::Val{N}; printtb=true, use_pw=true) where {N}
    (; x1, x2, x3) = X
    (; y1) = Y
    bc_base = @lazy @. y1 = myadd(x1, x2, x3)
    bc = use_pw ? to_pointwise_bc(bc_base) : bc_base
    if y1 isa Array
        if bc isa Base.Broadcast.Broadcasted
            for i in 1:100 # reduce variance / impact of launch latency
                @inbounds @simd for j in eachindex(bc)
                    y1[j] = bc[j]
                end
            end
        else
            for i in 1:100 # reduce variance / impact of launch latency
                @inbounds @simd for j in 1:N
                    y1[j] = bc[j]
                end
            end
        end
    else
        kernel =
            CUDA.@cuda always_inline = true launch = false custom_kernel_knl_bc!(
                y1,
                bc,
                Val(N),
            )
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(N, config.threads)
        blocks = cld(N, threads)
        printtb && @show blocks, threads
        for i in 1:100 # reduce variance / impact of launch latency
            kernel(y1, bc, Val(N); threads, blocks)
        end
    end
    return nothing
end;
@inline get_cart_lin_index(bc, n, I) = I
@inline get_cart_lin_index(bc::Base.Broadcast.Broadcasted, n, I) =
    CartesianIndices(map(x -> Base.OneTo(x), n))[I]
function custom_kernel_knl_bc!(y1, bc, ::Val{N}) where {N}
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        n = size(y1)
        if 1 ≤ I ≤ N
            ind = get_cart_lin_index(bc, n, I)
            y1[ind] = bc[ind]
        end
    end
    return nothing
end;

FT = Float32;
arr(T) = T(zeros(63,4,4,1,5400))
X_array = (;x1 = arr(ArrayType),x2 = arr(ArrayType),x3 = arr(ArrayType));
Y_array = (;y1 = arr(ArrayType),);
to_vec(ξ) = (;zip(propertynames(ξ), map(θ -> vec(θ), values(ξ)))...);
X_vector = to_vec(X_array);
Y_vector = to_vec(Y_array);
at_dot_call!(X_array, Y_array)
at_dot_call!(X_vector, Y_vector)
# custom_kernel!(X_vector, Y_vector, Val(length(X_vector.x1)))
custom_kernel_bc!(X_vector, Y_vector, Val(length(X_vector.x1)))
custom_kernel_bc!(X_array, Y_array, Val(length(X_vector.x1)); use_pw=false)
custom_kernel_bc!(X_array, Y_array, Val(length(X_vector.x1)); use_pw=true)

@pretty_belapsed at_dot_call!($X_array, $Y_array) # slow
@pretty_belapsed at_dot_call!($X_vector, $Y_vector) # fast
# @pretty_belapsed custom_kernel!($X_vector, $Y_vector, $(Val(length(X_vector.x1))))
@pretty_belapsed custom_kernel_bc!($X_vector, $Y_vector, $(Val(length(X_vector.x1)));printtb=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1)));printtb=false, use_pw=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $(Val(length(X_vector.x1)));printtb=false, use_pw=true)

#! format: on
