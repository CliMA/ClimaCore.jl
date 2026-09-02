import ClimaCore
import ClimaCore: Spaces, Grids, Topologies
using Test
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;
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
    # Context is not fully inferred due to nthreads() and cuda_ext_is_loaded(),
    # so let's ignore these for now.
    context = ClimaComms.context()

#! format: off
    if ClimaComms.device(context) isa ClimaComms.CUDADevice
        test_n_failures(56,   TU.PointSpace, context)
        test_n_failures(713,  TU.SpectralElementSpace1D, context)
        test_n_failures(402,  TU.SpectralElementSpace2D, context)
        test_n_failures(4,  TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,  TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(408,  TU.SphereSpectralElementSpace, context)
        test_n_failures(418,  TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(418,  TU.FaceExtrudedFiniteDifferenceSpace, context)
    else
        test_n_failures(0,    TU.PointSpace, context)
        test_n_failures(128,  TU.SpectralElementSpace1D, context)
        test_n_failures(295,  TU.SpectralElementSpace2D, context)
        test_n_failures(4,  TU.ColumnCenterFiniteDifferenceSpace, context)
        test_n_failures(5,  TU.ColumnFaceFiniteDifferenceSpace, context)
        test_n_failures(301,  TU.SphereSpectralElementSpace, context)
        test_n_failures(311,  TU.CenterExtrudedFiniteDifferenceSpace, context)
        test_n_failures(311,  TU.FaceExtrudedFiniteDifferenceSpace, context)

        # The OBJECT_CACHE causes inference failures that inhibit understanding
        # inference failures in _SpectralElementGrid2D, so let's `@test_opt` those
        # separately:

        space = TU.CenterExtrudedFiniteDifferenceSpace(Float32; context=ClimaComms.context())
        result = JET.@report_opt Grids._SpectralElementGrid2D(
            Spaces.topology(space),
            Spaces.quadrature_style(space),
            ClimaCore.DataLayouts.VIJFH;
            enable_bubble = false,
            autodiff_metric = true,
            enable_mask = false,
            discretization = Grids.CG(),
        )
        n_found = length(JET.get_reports(result.analyzer, result.result))
        # Measured at Float32, GLL{4}, on a CenterExtrudedFiniteDifferenceSpace.
        n_allowed = 106
        @test n_found ≤ n_allowed
        if n_found < n_allowed
            @info "Inference may have improved for _SpectralElementGrid2D: (n_found, n_allowed) = ($n_found, $n_allowed)"
        end
    end

#! format: on
end

nothing
