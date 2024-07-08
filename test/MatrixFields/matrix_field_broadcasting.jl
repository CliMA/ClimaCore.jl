#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_field_broadcasting.jl"))
=#
using Test

#! format: off
@testset "Scalar Matrix Field Broadcasting" begin
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_1.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_2.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_3.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_4.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_5.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_6.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_7.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_8.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_9.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_10.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_11.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_12.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_13.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_14.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_15.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_scalar_16.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc()
end

@testset "Non-scalar Matrix Field Broadcasting" begin
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_1.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_2.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_3.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc(); include(joinpath("matrix_fields_broadcasting", "test_non_scalar_4.jl")); @info "mem usage" rss = Sys.maxrss() / 2^30
    GC.gc()
end
#! format: on
