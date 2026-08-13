# GPU counterpart of `unit_matrix_field_broadcasting.jl`, covering the
# matrix_fields_broadcasting/ files (see the README there for what each one
# tests).
#
# On GPU each case gets its own Buildkite job. Several of these compile for
# minutes apiece, and running all of them in one process starved ptxas, which
# was killed with SIGKILL partway through `test_scalar_15`. Set
# `MATRIX_FIELD_BROADCASTING_CASE` to a case name to run only that one; with
# the variable unset every case runs here, which is what a local `include` of
# this file does.
using Test
include(joinpath(@__DIR__, "matrix_field_test_utils.jl"))
using ClimaCore.MatrixFields

print_mem = get(ENV, "BUILDKITE", "") == "true"
const SELECTED_CASE = get(ENV, "MATRIX_FIELD_BROADCASTING_CASE", "")

# `matched` guards against a typo in the Buildkite matrix: without it an
# unrecognized case name would run nothing and the job would still pass.
const matched = Ref(false)
function selected(name)
    isempty(SELECTED_CASE) && return true
    name == SELECTED_CASE || return false
    matched[] = true
    return true
end

isempty(SELECTED_CASE) || @info "Running one broadcasting case" SELECTED_CASE

#! format: off
@testset "Scalar Matrix Field Broadcasting (GPU)" begin
    GC.gc(); selected("test_scalar_1") && include(joinpath("matrix_fields_broadcasting", "test_scalar_1.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_2") && include(joinpath("matrix_fields_broadcasting", "test_scalar_2.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_3") && include(joinpath("matrix_fields_broadcasting", "test_scalar_3.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_4") && include(joinpath("matrix_fields_broadcasting", "test_scalar_4.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_5") && include(joinpath("matrix_fields_broadcasting", "test_scalar_5.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_6") && include(joinpath("matrix_fields_broadcasting", "test_scalar_6.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_7") && include(joinpath("matrix_fields_broadcasting", "test_scalar_7.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_8") && include(joinpath("matrix_fields_broadcasting", "test_scalar_8.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_9") && include(joinpath("matrix_fields_broadcasting", "test_scalar_9.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_10") && include(joinpath("matrix_fields_broadcasting", "test_scalar_10.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_11") && include(joinpath("matrix_fields_broadcasting", "test_scalar_11.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_12") && include(joinpath("matrix_fields_broadcasting", "test_scalar_12.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_13") && include(joinpath("matrix_fields_broadcasting", "test_scalar_13.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_14") && include(joinpath("matrix_fields_broadcasting", "test_scalar_14.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_15") && include(joinpath("matrix_fields_broadcasting", "test_scalar_15.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_16") && include(joinpath("matrix_fields_broadcasting", "test_scalar_16.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_scalar_17") && include(joinpath("matrix_fields_broadcasting", "test_scalar_17.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc()
end

@testset "Non-scalar Matrix Field Broadcasting (GPU)" begin
    GC.gc(); selected("test_non_scalar_1") && include(joinpath("matrix_fields_broadcasting", "test_non_scalar_1.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_non_scalar_2") && include(joinpath("matrix_fields_broadcasting", "test_non_scalar_2.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_non_scalar_3") && include(joinpath("matrix_fields_broadcasting", "test_non_scalar_3.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_non_scalar_4") && include(joinpath("matrix_fields_broadcasting", "test_non_scalar_4.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc(); selected("test_non_scalar_5") && include(joinpath("matrix_fields_broadcasting", "test_non_scalar_5.jl")); print_mem && @info "mem usage: rss = $(Sys.maxrss() / 2^30)"
    GC.gc()
end
#! format: on


isempty(SELECTED_CASE) ||
    matched[] ||
    error("MATRIX_FIELD_BROADCASTING_CASE=$SELECTED_CASE names no case in \
           this file")
