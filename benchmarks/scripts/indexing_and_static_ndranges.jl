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
 - https://github.com/CliMA/ClimaCore.jl/issues/1889
 - https://github.com/JuliaLang/julia/issues/28126
 - https://github.com/JuliaLang/julia/issues/32051

# Benchmark results:

Local Apple M1 Mac (CPU):
```
at_dot_call!($X_array, $Y_array):
     143 milliseconds, 774 microseconds
at_dot_call!($X_vector, $Y_vector):
     65 milliseconds, 567 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $us; printtb = false):
     66 milliseconds, 870 microseconds
custom_kernel_bc!($X_array, $Y_array, $us; printtb = false, use_pw = false):
     143 milliseconds, 643 microseconds
custom_kernel_bc!($X_array, $Y_array, $us; printtb = false, use_pw = true):
     65 milliseconds, 778 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $uss; printtb = false):
     65 milliseconds, 765 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss; printtb = false, use_pw = false):
     144 milliseconds, 271 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss; printtb = false, use_pw = true):
     66 milliseconds, 376 microseconds
```

Clima A100
```
at_dot_call!($X_array, $Y_array):
     6 milliseconds, 775 microseconds
at_dot_call!($X_vector, $Y_vector):
     2 milliseconds, 834 microseconds
custom_sol_kernel!($X_vector, $Y_vector, $(Val(N))):
     2 milliseconds, 547 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $us; printtb = false):
     2 milliseconds, 561 microseconds
custom_kernel_bc!($X_array, $Y_array, $us; printtb = false, use_pw = false):
     4 milliseconds, 160 microseconds
custom_kernel_bc!($X_array, $Y_array, $us; printtb = false, use_pw = true):
     2 milliseconds, 584 microseconds
custom_kernel_bc!($X_vector, $Y_vector, $uss; printtb = false):
     2 milliseconds, 540 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss; printtb = false, use_pw = false):
     2 milliseconds, 715 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss; printtb = false, use_pw = true):
     2 milliseconds, 547 microseconds
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

function custom_sol_kernel!(X, Y, ::Val{N}) where {N}
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

abstract type AbstractUniversalSizes{Nv, Nij} end
struct UniversalSizesCC{Nv, Nij} <: AbstractUniversalSizes{Nv, Nij}
    Nh::Int
end
struct UniversalSizesStatic{Nv, Nij, Nh} <: AbstractUniversalSizes{Nv, Nij} end

get_Nv(::AbstractUniversalSizes{Nv}) where {Nv} = Nv
get_Nij(::AbstractUniversalSizes{Nv, Nij}) where {Nv, Nij} = Nij
get_Nh(us::UniversalSizesCC) = us.Nh
get_Nh(::UniversalSizesStatic{Nv, Nij, Nh}) where {Nv, Nij, Nh} = Nh
get_N(us::AbstractUniversalSizes{Nv, Nij}) where {Nv, Nij} = prod((Nv,Nij,Nij,1,get_Nh(us)))
UniversalSizesCC(Nv, Nij, Nh) = UniversalSizesCC{Nv, Nij}(Nh)
UniversalSizesStatic(Nv, Nij, Nh) = UniversalSizesStatic{Nv, Nij, Nh}()
using Test
us_tup = (1, 2, 3)
@test get_Nv(UniversalSizesCC(us_tup...))  == get_Nv(UniversalSizesStatic(us_tup...))
@test get_Nij(UniversalSizesCC(us_tup...)) == get_Nij(UniversalSizesStatic(us_tup...))
@test get_Nh(UniversalSizesCC(us_tup...))  == get_Nh(UniversalSizesStatic(us_tup...))
@test get_N(UniversalSizesCC(us_tup...))   == get_N(UniversalSizesStatic(us_tup...))

function custom_kernel_bc!(X, Y, us::AbstractUniversalSizes; printtb=true, use_pw=true)
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
                @inbounds @simd for j in 1:get_N(us)
                    y1[j] = bc[j]
                end
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
        for i in 1:100 # reduce variance / impact of launch latency
            kernel(y1, bc,us; threads, blocks)
        end
    end
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

FT = Float32;
arr(T) = T(zeros(63,4,4,1,5400))
X_array = (;x1 = arr(ArrayType),x2 = arr(ArrayType),x3 = arr(ArrayType));
Y_array = (;y1 = arr(ArrayType),);
to_vec(ξ) = (;zip(propertynames(ξ), map(θ -> vec(θ), values(ξ)))...);
X_vector = to_vec(X_array);
Y_vector = to_vec(Y_array);
at_dot_call!(X_array, Y_array)
at_dot_call!(X_vector, Y_vector)
N = length(X_vector.x1)
(Nv, Nij, _, Nf, Nh) = size(Y_array.y1);
us = UniversalSizesCC(Nv, Nij, Nh);
uss = UniversalSizesStatic(Nv, Nij, Nh);
@test get_N(us) == N
@test get_N(uss) == N
iscpu = ArrayType === identity
iscpu || custom_sol_kernel!(X_vector, Y_vector, Val(N))
custom_kernel_bc!(X_vector, Y_vector, us)
custom_kernel_bc!(X_array, Y_array, us; use_pw=false)
custom_kernel_bc!(X_array, Y_array, us; use_pw=true)

custom_kernel_bc!(X_vector, Y_vector, uss)
custom_kernel_bc!(X_array, Y_array, uss; use_pw=false)
custom_kernel_bc!(X_array, Y_array, uss; use_pw=true)

@pretty_belapsed at_dot_call!($X_array, $Y_array) # slow
@pretty_belapsed at_dot_call!($X_vector, $Y_vector) # fast
iscpu || @pretty_belapsed custom_sol_kernel!($X_vector, $Y_vector, $(Val(N)))
@pretty_belapsed custom_kernel_bc!($X_vector, $Y_vector, $us; printtb=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $us; printtb=false, use_pw=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $us; printtb=false, use_pw=true)

@pretty_belapsed custom_kernel_bc!($X_vector, $Y_vector, $uss; printtb=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $uss; printtb=false, use_pw=false)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $uss; printtb=false, use_pw=true)

#! format: on
