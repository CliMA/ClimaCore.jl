#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Spaces", "opt_spaces.jl"))
=#
import ClimaCore
import ClimaCore: Spaces, Grids
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
        test_n_failures(1106, TU.SpectralElementSpace2D, context)
        test_n_failures(4,    TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,    TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(1112, TU.SphereSpectralElementSpace, context)
        test_n_failures(1114, TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(1114, TU.FaceExtrudedFiniteDifferenceSpace, context)
    else
        test_n_failures(0,    TU.PointSpace, context)
        test_n_failures(137,  TU.SpectralElementSpace1D, context)
        test_n_failures(272,  TU.SpectralElementSpace2D, context)
        test_n_failures(4,    TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,    TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(278,  TU.SphereSpectralElementSpace, context)
        test_n_failures(283,  TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(283,  TU.FaceExtrudedFiniteDifferenceSpace, context)

        # The OBJECT_CACHE causes inference failures that inhibit understanding
        # inference failures in _SpectralElementGrid2D, so let's `@test_opt`
        # _SpectralElementGrid2D separately:
        space = TU.CenterExtrudedFiniteDifferenceSpace(Float32; context=ClimaComms.context())
        result = JET.@report_opt Grids._SpectralElementGrid2D(Spaces.topology(space), Spaces.quadrature_style(space); enable_bubble=false)
        n_found = length(JET.get_reports(result.analyzer, result.result))
        n_allowed = 351
        @test n_found ≤ n_allowed
        n_found < n_allowed && @info "Inference may have improved. (found, allowed) = ($n_found, $n_allowed)"
    end

#! format: on
end

nothing
