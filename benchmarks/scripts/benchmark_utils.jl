import CUDA
using BenchmarkTools, Dates
using LazyBroadcast: @lazy

"""
    caller_name(@__FILE__)

Returns a string of the (single) line pointing to the function that
called the function we're in.
"""
macro caller_name(f)
    quote
        string(readlines($f)[StackTraces.stacktrace()[4].line])
    end
end

Base.@kwdef mutable struct Benchmark
    problem_size::Tuple
    float_type::Type
    device_bandwidth_GBs::Int = 2_039
    data::Vector = []
end

function perf_stats(; bm::Benchmark, kernel_time_s, n_reads_writes)
    N = prod(bm.problem_size)
    GB = N * n_reads_writes * sizeof(bm.float_type) / 1024^3
    achieved_bandwidth_GBs = GB / kernel_time_s
    bandwidth_efficiency =
        achieved_bandwidth_GBs / bm.device_bandwidth_GBs * 100
    return (; N, GB, achieved_bandwidth_GBs, bandwidth_efficiency)
end;

time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end
trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s

abstract type AbstractUniversalSizes{Nv, Nij} end
struct UniversalSizesCC{Nv, Nij} <: AbstractUniversalSizes{Nv, Nij}
    Nh::Int
end
struct UniversalSizesStatic{Nv, Nij, Nh} <: AbstractUniversalSizes{Nv, Nij} end

get_Nv(::AbstractUniversalSizes{Nv}) where {Nv} = Nv
get_Nij(::AbstractUniversalSizes{Nv, Nij}) where {Nv, Nij} = Nij
get_Nh(us::UniversalSizesCC) = us.Nh
get_Nh(::UniversalSizesStatic{Nv, Nij, Nh}) where {Nv, Nij, Nh} = Nh
get_N(us::AbstractUniversalSizes{Nv, Nij}) where {Nv, Nij} =
    prod((Nv, Nij, Nij, 1, get_Nh(us)))
UniversalSizesCC(Nv, Nij, Nh) = UniversalSizesCC{Nv, Nij}(Nh)
UniversalSizesStatic(Nv, Nij, Nh) = UniversalSizesStatic{Nv, Nij, Nh}()

import PrettyTables
function tabulate_benchmark(bm)
    funcs = map(x -> x.caller, bm.data)
    timings = map(x -> time_and_units_str(x.kernel_time_s), bm.data)
    n_reads_writes = map(x -> x.n_reads_writes, bm.data)
    nreps = map(x -> x.nreps, bm.data)
    achieved_bandwidth_GBs = map(x -> x.achieved_bandwidth_GBs, bm.data)
    bandwidth_efficiency = map(x -> x.bandwidth_efficiency, bm.data)
    header = [
        "funcs",
        "time per call",
        "bw %",
        "achieved bw",
        "n-reads/writes",
        "n-reps",
    ]
    data = hcat(
        funcs,
        timings,
        bandwidth_efficiency,
        achieved_bandwidth_GBs,
        n_reads_writes,
        nreps,
    )
    title = "Problem size: $(bm.problem_size), float_type = $(bm.float_type), device_bandwidth_GBs=$(bm.device_bandwidth_GBs)"
    PrettyTables.pretty_table(
        data;
        title,
        header,
        alignment = :l,
        crop = :none,
    )
end
