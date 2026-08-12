# GPU counterpart of `unit_matrix_field_broadcasting.jl`, including all
# matrix_fields_broadcasting/ files (see the README there for what each one
# tests).
using Test
include(joinpath(@__DIR__, "matrix_field_test_utils.jl"))
using ClimaCore.MatrixFields

print_mem = get(ENV, "BUILDKITE", "") == "true"
#! format: off
@testset "Scalar Matrix Field Broadcasting (GPU)" begin
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

@testset "Non-scalar Matrix Field Broadcasting (GPU)" begin
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_1.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_2.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_3.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_4.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_5.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc()
end
#! format: on
