# import CUDA
import ClimaComms
using BenchmarkTools, Dates
using ClimaCore: @lazy

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

"""
    device_info(device_name::String)

Call with `device_info(CUDA.name(CUDA.device()))`
"""
function device_info(device_name)
    device_specs = Dict(
        "NVIDIA A100-SXM4-80GB" => (; device_bandwidth_GBs = 2_039), # https://www.nvidia.com/en-us/data-center/a100/
        "Tesla P100-PCIE-16GB" => (; device_bandwidth_GBs = 732), # https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf
        "NVIDIA H100 80GB HBM3" => (; device_bandwidth_GBs = 3_350), # https://www.nvidia.com/en-us/data-center/h100/
        "NVIDIA GeForce GTX 1050" => (; device_bandwidth_GBs = 112.1), # https://www.techpowerup.com/gpu-specs/geforce-gtx-1050.c2875
    )
    is_cuda = ClimaComms.device() isa ClimaComms.CUDADevice
    if is_cuda && haskey(device_specs, device_name)
        (; device_bandwidth_GBs) = device_specs[device_name]
        return (; device_bandwidth_GBs, exists = true, name = device_name)
    else
        return (; device_bandwidth_GBs = 1, exists = false, name = device_name)
    end
end

Base.@kwdef mutable struct Benchmark
    problem_size = nothing
    float_type::Type
    data::Vector = []
    unfound_device::Bool = false
    unfound_device_name::String = ""
    device_name::String = ""
end

function print_unfound_devices(bm::Benchmark)
    bm.unfound_device || return nothing
    println("\nUnfound device: $(bm.unfound_device_name). Please")
    println("look up specs and add to device_bandwidth() in")
    println("$(@__FILE__).\n")
end

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
Base.size(us::AbstractUniversalSizes{Nv, Nij}) where {Nv, Nij} =
    (Nv, Nij, Nij, 1, get_Nh(us))
UniversalSizesCC(Nv, Nij, Nh) = UniversalSizesCC{Nv, Nij}(Nh)
UniversalSizesStatic(Nv, Nij, Nh) = UniversalSizesStatic{Nv, Nij, Nh}()

import PrettyTables
function tabulate_benchmark(bm)
    funcs = map(x -> strip(x.caller), bm.data)
    timings = map(x -> time_and_units_str(x.kernel_time_s), bm.data)
    n_reads_writes = map(x -> x.n_reads_writes, bm.data)
    nreps = map(x -> x.nreps, bm.data)
    dinfo = device_info(bm.device_name)
    achieved_bandwidth_GBs = map(x -> x.achieved_bandwidth_GBs, bm.data)
    bandwidth_efficiency = if dinfo.exists
        map(x -> x / dinfo.device_bandwidth_GBs * 100, achieved_bandwidth_GBs)
    else
        ()
    end
    problem_size = map(x -> x.problem_size, bm.data)
    # if we specify the problem size up front, then make
    # sure that there is no variation when collecting:
    if !isnothing(bm.problem_size)
        @assert all(prod.(problem_size) .== prod(bm.problem_size))
    end
    N = map(x -> prod(x), problem_size)
    no_bw_efficiency = length(bandwidth_efficiency) == 0
    header = [
        "funcs",
        "time per call",
        (no_bw_efficiency ? () : ("bw %",))...,
        "achieved bw",
        (allequal(n_reads_writes) ? () : ("N reads-writes",))...,
        (allequal(N) ? () : ("problem size",))...,
        (allequal(nreps) ? () : ("n-reps",))...,
    ]
    args = (
        funcs,
        timings,
        (no_bw_efficiency ? () : (bandwidth_efficiency,))...,
        achieved_bandwidth_GBs,
        (allequal(n_reads_writes) ? () : (n_reads_writes,))...,
        (allequal(N) ? () : (problem_size,))...,
        (allequal(nreps) ? () : (nreps,))...,
    )
    data = hcat(args...)
    n_reads_writes_str =
        allequal(n_reads_writes) ? "N reads-writes: $(n_reads_writes[1]), " : ""
    problem_size_str = allequal(N) ? "Problem size: $(problem_size[1]), " : ""
    nreps_str = allequal(nreps) ? "N-reps: $(nreps[1]), " : ""
    device_bandwidth_GBs_str =
        dinfo.exists ? "Device_bandwidth_GBs=$(dinfo.device_bandwidth_GBs)" : ""
    print_unfound_devices(bm)
    title = strip(
        "$problem_size_str$n_reads_writes_str$nreps_str Float_type = $(bm.float_type), $device_bandwidth_GBs_str",
    )
    PrettyTables.pretty_table(data; title, header, alignment = :l, crop = :none)
end

push_info(
    bm::Nothing;
    kernel_time_s,
    nreps,
    caller,
    n_reads_writes,
    problem_size,
) = nothing
function push_info(
    bm;
    kernel_time_s,
    nreps,
    caller,
    n_reads_writes,
    problem_size,
)
    N = prod(problem_size)
    GB = N * n_reads_writes * sizeof(bm.float_type) / 1024^3
    achieved_bandwidth_GBs = GB / kernel_time_s
    dinfo = device_info(bm.device_name)
    if !dinfo.exists
        bm.unfound_device = true
        bm.unfound_device_name = dinfo.name
    end

    nt = (;
        caller,
        kernel_time_s,
        n_reads_writes,
        nreps,
        problem_size,
        N,
        GB,
        achieved_bandwidth_GBs,
    )
    push!(bm.data, nt)
end
