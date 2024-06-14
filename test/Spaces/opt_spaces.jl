#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Spaces", "opt_spaces.jl"))
=#
import ClimaCore
using Test
include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU
import ClimaComms
ClimaComms.@import_required_backends

using JET
function test_n_failures(n_allowed, f::F, context) where {F}
    result = JET.@report_opt f(Float32; context = context)
    n_found = length(JET.get_reports(result.analyzer, result.result))
    @test n_found ≤ n_allowed
    if n_found < n_allowed
        @info "Inference may have improved for $f: (n_found, n_allowed) = ($n_found, $n_allowed)"
    end
    return nothing
end

@testset "Number of JET failures" begin
    FT = Float32
    zelem = 4
    helem = 4
    # context is not fully inferred due to nthreads() and cuda_ext_is_loaded(),
    # so let's ignore these for now.
    context = ClimaComms.context()

#! format: off
    if ClimaComms.device(context) isa ClimaComms.CUDADevice
        test_n_failures(86,   TU.PointSpace, context)
        test_n_failures(144,  TU.SpectralElementSpace1D, context)
        test_n_failures(1103,  TU.SpectralElementSpace2D, context)
        test_n_failures(4,    TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,    TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(1104,  TU.SphereSpectralElementSpace, context)
        test_n_failures(1109,  TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(1109,  TU.FaceExtrudedFiniteDifferenceSpace, context)
    else
        test_n_failures(0,    TU.PointSpace, context)
        test_n_failures(137,  TU.SpectralElementSpace1D, context)
        test_n_failures(279,  TU.SpectralElementSpace2D, context)
        test_n_failures(4,    TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,    TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(280,  TU.SphereSpectralElementSpace, context)
        test_n_failures(285,  TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(285,  TU.FaceExtrudedFiniteDifferenceSpace, context)
    end
#! format: on
end

nothing
