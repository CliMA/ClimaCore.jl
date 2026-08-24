"""
    TestUtilities

TestUtilities is designed to:

  - Reduce the test boilerplate
  - Provide testers with iterators over
    instances of types to ease testing
    a diverse set of inputs to functions
"""
module TestUtilities

using IntervalSets
using Test
using LinearAlgebra
import ClimaComms
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.Operators
import ClimaCore.Utilities
import ClimaCore.Quadratures
import ClimaCore.Geometry
import ClimaCore.Meshes
import ClimaCore.Spaces
import ClimaCore.CommonSpaces
import ClimaCore.Topologies
import ClimaCore.Domains
import ClimaCore.Hypsography

export convergence_rate,
    PointSpace,
    SpectralElementSpace1D,
    SpectralElementSpace2D,
    ColumnCenterFiniteDifferenceSpace,
    ColumnFaceFiniteDifferenceSpace,
    SphereSpectralElementSpace,
    CenterExtrudedFiniteDifferenceSpace,
    FaceExtrudedFiniteDifferenceSpace,
    all_spaces,
    bycolumnable,
    levelable,
    fc_index,
    has_z_coordinates,
    test_column_operators,
    ssp33!,
    @test_zero_allocations,
    @test_precisions

"""
    @test_zero_allocations expr

Evaluates `expr` in an isolated `@noinline` runner to prevent closure/box capture artifacts,
warms up the evaluation, and asserts `@test allocs == 0`.
"""
macro test_zero_allocations(expr)
    quote
        let
            @noinline function _run_zero_alloc_eval()
                $(esc(expr))
                return nothing
            end
            _run_zero_alloc_eval() # Warmup
            $Test.@test ($Base.@allocated _run_zero_alloc_eval()) == 0
        end
    end
end

"""
    @test_precisions [Float32, Float64] FT begin ... end
    @test_precisions FT begin ... end  # Defaults to (Float32, Float64)

Executes a test block iteratively across floating point precisions, binding `FT` to each type.
"""
macro test_precisions(args...)
    if length(args) == 2
        var = args[1]
        block = args[2]
        types = :((Float32, Float64))
    elseif length(args) == 3
        types = args[1]
        var = args[2]
        block = args[3]
    else
        error("Usage: @test_precisions [types] FT block")
    end
    quote
        for $(esc(var)) in $(esc(types))
            $(esc(block))
        end
    end
end

"""
    ssp33!(rhs!, y, dy, y1, y2, params, dt, nsteps)

Hand-rolled SSP RK33 (Shu-Osher three-stage, third-order strong stability
preserving) time integrator, so that smoke tests need no external
time-stepper dependency. `rhs!(dy, y, params, t)` writes the tendency in
place; `y` may be a scalar `Field` or a `FieldVector`. The full step `dt` is
implicit in `params` where a scheme needs it (e.g. FCT limiters read `Δt`
from `params`, not a stage substep).
"""
function ssp33!(rhs!, y, dy, y1, y2, params, dt, nsteps)
    for _ in 1:nsteps
        rhs!(dy, y, params, zero(dt))
        @. y1 = y + dt * dy
        rhs!(dy, y1, params, zero(dt))
        @. y2 = (3 * y + y1 + dt * dy) / 4
        rhs!(dy, y2, params, zero(dt))
        @. y = (y + 2 * y2 + 2 * dt * dy) / 3
    end
    return y
end

"""
    convergence_rate(err, Δh)

Estimate pairwise convergence rates given vectors or tuples `err` and `Δh`:
r[i] = log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1])
"""
convergence_rate(err::AbstractVector, Δh::AbstractVector) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

convergence_rate(err::Tuple, Δh::Tuple) =
    ntuple(i -> log(err[i + 1] / err[i]) / log(Δh[i + 1] / Δh[i]), length(Δh) - 1)

function PointSpace(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    coord = Geometry.XPoint(FT(π))
    space = Spaces.PointSpace(context, coord)
    return space
end

function SpectralElementSpace1D(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    # 1d domain space
    domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(context, mesh)
    quad = Quadratures.GLL{4}()
    return Spaces.SpectralElementSpace1D(topology, quad)
end

function SpectralElementSpace2D(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    # 1×1 domain space
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

#= (single column) =#
function ColumnCenterFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    zlim = (0, 1)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain, nelems = zelem)
    topology = Topologies.IntervalTopology(context, mesh)
    return Spaces.CenterFiniteDifferenceSpace(topology)
end

#= (single column) =#
function ColumnFaceFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    cspace = ColumnCenterFiniteDifferenceSpace(FT; zelem, context)
    return Spaces.FaceFiniteDifferenceSpace(cspace)
end

function SphereSpectralElementSpace(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function CenterExtrudedFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
    helem = 4,
    Nq = 4,
    deep = false,
    topography = false,
    autodiff_metric = true,
    VIJH = DataLayouts.VIJFH,
) where {FT}
    radius = FT(128)
    zlim = (FT(0), FT(1))

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zlim[1]),
        Geometry.ZPoint(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(context, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    quad = Quadratures.GLL{Nq}()
    hspace = Spaces.SpectralElementSpace2D(
        htopology,
        quad;
        autodiff_metric,
        VIJH,
    )

    hypsography = if topography
        # A function of latitude and longitude
        H = (zlim[2] - zlim[1]) / zelem
        (; lat, long) = Fields.coordinate_field(hspace)
        surface_elevation =
            @. Geometry.ZPoint(H * (cosd(lat) + cosd(long) + 1))
        Hypsography.LinearAdaption(surface_elevation)
    else
        Hypsography.Flat()
    end
    return Spaces.ExtrudedFiniteDifferenceSpace(
        hspace,
        vspace,
        hypsography;
        deep,
    )
end

function FaceExtrudedFiniteDifferenceSpace(::Type{FT}; kwargs...) where {FT}
    cspace = CenterExtrudedFiniteDifferenceSpace(FT; kwargs...)
    return Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
end

function PointColumnEnsembleSpace(::Type{FT}; context, zelem = 10, kwargs...) where {FT}
    staggering = Spaces.CellCenter()
    lats = FT.([0.0, 1.0, 2.0])
    longs = FT.([3.0, 4.0, 5.0])
    points = [Geometry.LatLongPoint(lat, long) for (lat, long) in zip(lats, longs)]
    z_min = -10.0
    z_max = 10.0
    (; device) = context
    return CommonSpaces.PointColumnEnsembleSpace(
        FT;
        z_elem = zelem,
        staggering,
        points,
        z_min,
        z_max,
        device,
        kwargs...,
    )
end

function all_spaces(
    ::Type{FT};
    zelem = 10,
    helem = 4,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    return [
        PointSpace(FT; context),
        SpectralElementSpace1D(FT; context),
        SpectralElementSpace2D(FT; context),
        # TODO: add these
        # SpectralElementRectilinearSpace2D(FT; context),
        # SpectralElementFiniteDifferenceRectilinearSpace2D(FT; context),
        ColumnCenterFiniteDifferenceSpace(FT; zelem, context),
        ColumnFaceFiniteDifferenceSpace(FT; zelem, context),
        PointColumnEnsembleSpace(FT; zelem, context),
        SphereSpectralElementSpace(FT; context),
        CenterExtrudedFiniteDifferenceSpace(FT; zelem, context, helem),
        FaceExtrudedFiniteDifferenceSpace(FT; zelem, context, helem),
        # TODO: incorporate this list of spaces somehow:
        #     space_vf = Spaces.CenterFiniteDifferenceSpace(topology_z)
        #     space_ifh = Spaces.SpectralElementSpace1D(topology_x, quad)
        #     space_ijfh = Spaces.SpectralElementSpace2D(topology_xy, quad)
        #     space_vifh = Spaces.ExtrudedFiniteDifferenceSpace(space_ifh, space_vf)
        #     space_vijfh = Spaces.ExtrudedFiniteDifferenceSpace(space_ijfh, space_vf)
    ]
end

bycolumnable(space) = (
    space isa Spaces.ExtrudedFiniteDifferenceSpace ||
    space isa Spaces.SpectralElementSpace1D ||
    space isa Spaces.SpectralElementSpace2D ||
    space isa Spaces.MultiColumnFiniteDifferenceSpace
)

levelable(space) = (
    space isa Spaces.ExtrudedFiniteDifferenceSpace ||
    space isa Spaces.FiniteDifferenceSpace ||
    space isa Spaces.MultiColumnFiniteDifferenceSpace
)

fc_index(
    i,
    ::Union{
        Spaces.FaceExtrudedFiniteDifferenceSpace,
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceMultiColumnFiniteDifferenceSpace,
    },
) = Utilities.PlusHalf(i)

fc_index(
    i,
    ::Union{
        Spaces.CenterExtrudedFiniteDifferenceSpace,
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterMultiColumnFiniteDifferenceSpace,
    },
) = i

has_z_coordinates(space) = :z in propertynames(Spaces.coordinates_data(space))

# Helper function to test all three operators on a column space
function test_column_operators(column_space, expect_zero_div = false)
    FT = Spaces.undertype(column_space)
    test_scalar = ones(FT, column_space)

    @testset "Gradient" begin
        grad_op = Operators.Gradient()
        result = grad_op.(test_scalar)
        @test axes(result) == axes(test_scalar)
        @test eltype(result) <: Geometry.CovariantVector
    end

    @testset "Divergence" begin
        grad_op = Operators.Gradient()
        vector_field = grad_op.(test_scalar)
        div_op = Operators.Divergence()
        result = div_op.(vector_field)
        @test axes(result) == axes(test_scalar)
        if expect_zero_div
            @test maximum(abs.(parent(result))) == 0.0
        end
    end

    @testset "Curl" begin
        grad_op = Operators.Gradient()
        vector_field = grad_op.(test_scalar)
        curl_op = Operators.Curl()
        result = curl_op.(vector_field)
        @test axes(result) == axes(test_scalar)
        @test eltype(result) <: Geometry.ContravariantVector
    end
end

"""
    allocation_checks_meaningful()

Whether `@allocated` in this process reflects the allocations of optimized
code.

`--check-bounds=yes` inhibits the optimizations that make ClimaCore's in-place
broadcasts allocation-free, so under it a zero-allocation sentinel measures the
flag rather than the code (a spectral-element gradient goes from 0 to ~14 kB).
The GitHub Actions job runs the suite through `Pkg.test`, which passes
`--check-bounds=yes`; Buildkite's curated jobs do not, so the sentinels still
gate there.
"""
allocation_checks_meaningful() = Base.JLOptions().check_bounds == 0

end
