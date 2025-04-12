#=
julia --project=.buildkite
ENV["CLIMACOMMS_DEVICE"] = "CPU"
ENV["BUILDKITE"] = "true" # to also run opt tests
using Revise; include(joinpath("test", "MatrixFields", "matrix_field_broadcasting.jl"))

# For profiling/benchmarking:

```
import Profile, ProfileCanvas

function do_work(space, bc, idx, hidx, n)
    for i in 1:n
        call_getidx(space, bc, idx, hidx)
    end
    return nothing
end

(; space, bc, idx_l, idx_i, idx_r, hidx) = get_getidx_args(bc);
do_work(space, bc, idx_i, hidx, 1)
Profile.clear()
prof = Profile.@profile do_work(space, bc, idx_i, hidx, 10^6)
results = Profile.fetch()
Profile.clear()
ProfileCanvas.html_file("flame.html", results)

perf_getidx(bc)
```
=#
using Test

print_mem = get(ENV, "BUILDKITE", "") == "true"
#! format: off
@testset "Scalar Matrix Field Broadcasting" begin
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_1.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_2.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_3.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_4.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_5.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_6.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_7.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_8.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_9.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_10.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_11.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_12.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_13.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_14.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_15.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_16.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc()
end

@testset "Non-scalar Matrix Field Broadcasting" begin
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_1.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_2.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_3.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_4.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_5.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc()
end
#! format: on

nothing
