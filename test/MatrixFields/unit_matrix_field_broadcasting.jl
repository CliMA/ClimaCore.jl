#=
# For profiling/benchmarking:

```
import Profile, ProfileCanvas

function do_work(bc_column, n)
    for i in 1:n
        call_apply_operators(bc_column)
    end
    return nothing
end

bc_column = column_broadcast(bc);
do_work(bc_column, 1)
Profile.clear()
prof = Profile.@profile do_work(bc_column, 10^5)
results = Profile.fetch()
Profile.clear()
ProfileCanvas.html_file("flame.html", results)

perf_apply_operators(bc)
```
=#
using Test
include(joinpath(@__DIR__, "matrix_field_test_utils.jl"))
using ClimaCore.MatrixFields

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
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_17.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
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
