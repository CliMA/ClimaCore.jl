# Edge-case meshes for GPU operator kernels, compared against a CPU reference:
#  - single-element meshes, where every element face is a periodic self-neighbor
#    (1x1 rectangle) or a panel seam (ne = 1 cubed sphere), so DSS and operator
#    kernels get no interior element connectivity;
#  - boundary-only columns, where the boundary windows of a finite difference
#    stencil cover the whole column and the interior loop is empty.
# Registered :gpu_only because the point is the CPU-vs-GPU comparison; on a CPU
# device it degenerates to comparing two CPU runs.
using Test
using LinearAlgebra: norm
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Geometry,
    Operators,
    Quadratures
import ClimaCore.CommonSpaces: PointColumnEnsembleSpace, CellCenter

const test_device = ClimaComms.device()
const cpu_device = ClimaComms.CPUSingleThreaded()
@info "gpu_edge_cases: comparing $test_device against $cpu_device"

# Compare a field computed on the test device against the CPU reference.
device_matches_cpu(field, field_cpu; rtol) = isapprox(
    Array(parent(field)),
    parent(field_cpu),
    rtol = rtol,
    norm = x -> norm(x, Inf),
)

function periodic_1x1_space(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(-FT(π)),
            Geometry.XPoint(FT(π)),
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(-FT(π)),
            Geometry.YPoint(FT(π)),
            periodic = true,
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
end

function ne1_sphere_space(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.SphereDomain(FT(1))
    mesh = Meshes.EquiangularCubedSphere(domain, 1)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
end

function single_level_column_spaces(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(1));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(context, mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(topology)
    return (center_space, Spaces.FaceFiniteDifferenceSpace(center_space))
end

function se_operator_results(space)
    coords = Fields.coordinate_field(space)
    is_sphere = :lat in propertynames(coords)
    f =
        is_sphere ? (@. sind(coords.lat) + cosd(coords.long)) :
        (@. sin(coords.x) + cos(coords.y))
    u = @. Geometry.UVVector(f, 2 * f)
    grad_f = Operators.Gradient().(f)
    wdiv_u = Operators.WeakDivergence().(u)
    Spaces.weighted_dss!(grad_f)
    Spaces.weighted_dss!(wdiv_u)
    return (grad_f, wdiv_u)
end

@testset "single-element SE meshes (device vs CPU) [$FT]" for FT in (
    Float32,
    Float64,
)
    for get_space in (periodic_1x1_space, ne1_sphere_space)
        results = se_operator_results(get_space(FT, test_device))
        results_cpu = se_operator_results(get_space(FT, cpu_device))
        for (field, field_cpu) in zip(results, results_cpu)
            @test device_matches_cpu(field, field_cpu; rtol = 10 * eps(FT))
            @test all(isfinite, Array(parent(field)))
        end
    end
end

function fd_f2c_results(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zf = Fields.coordinate_field(face_space).z
    u = @. Geometry.WVector(zf + FT(0.5))
    w = @. zf^2 + 1
    divf2c = Operators.DivergenceF2C()
    gradf2c = Operators.GradientF2C()
    interpf2c = Operators.InterpolateF2C()
    return (divf2c.(u), gradf2c.(w), interpf2c.(w))
end

function fd_c2f_results(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zc = Fields.coordinate_field(center_space).z
    f = @. zc^2 + 1
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(2))),
    )
    interpc2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    return (gradc2f.(f), interpc2f.(f))
end

@testset "boundary-only column, Nv = 1 (device vs CPU) [$FT]" for FT in (
    Float32,
    Float64,
)
    spaces = single_level_column_spaces(FT, test_device)
    spaces_cpu = single_level_column_spaces(FT, cpu_device)

    # Face-to-center stencils need no boundary windows, so they support a
    # single-level column, where the boundary faces are the whole column.
    for (field, field_cpu) in zip(fd_f2c_results(spaces...), fd_f2c_results(spaces_cpu...))
        @test device_matches_cpu(field, field_cpu; rtol = 10 * eps(FT))
        @test all(isfinite, Array(parent(field)))
    end

    # Center-to-face stencils on a column whose interior window is empty:
    # `Operators.window_bounds` used to reject these, but it now clamps the
    # overlapping boundary windows (boundary handling is dispatched per index,
    # with the left window taking precedence), so both faces are computed with
    # their boundary rows.
    for (field, field_cpu) in
        zip(fd_c2f_results(spaces...), fd_c2f_results(spaces_cpu...))
        @test device_matches_cpu(field, field_cpu; rtol = 10 * eps(FT))
        @test all(isfinite, Array(parent(field)))
    end
end

@testset "smallest column with an interior, Nv = 2 (device vs CPU) [$FT]" for FT in (
    Float32,
    Float64,
)
    two_level_spaces(device) = begin
        context = ClimaComms.SingletonCommsContext(device)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(FT(0)),
            Geometry.ZPoint(FT(1));
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = 2)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        (center_space, Spaces.FaceFiniteDifferenceSpace(center_space))
    end
    spaces = two_level_spaces(test_device)
    spaces_cpu = two_level_spaces(cpu_device)
    results = (fd_f2c_results(spaces...)..., fd_c2f_results(spaces...)...)
    results_cpu =
        (fd_f2c_results(spaces_cpu...)..., fd_c2f_results(spaces_cpu...)...)
    for (field, field_cpu) in zip(results, results_cpu)
        @test device_matches_cpu(field, field_cpu; rtol = 10 * eps(FT))
        @test all(isfinite, Array(parent(field)))
    end
end

# Advection and nested-multiply stencils, covering the eager GPU kernel's
# advection (pointwise and operator-matrix) and matrix-multiply handlers.
function fd_advection_results(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zc = Fields.coordinate_field(center_space).z
    x = @. sin(3 * zc / 1000)
    w = Geometry.WVector.(ones(FT, face_space))
    up3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(1),
        top = Operators.Extrapolate(1),
    )
    lvl = Operators.LinVanLeerC2F(
        constraint = Operators.MonotoneLocalExtrema(),
    )
    div_sv = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    divf2c = Operators.DivergenceF2C()
    return (
        (@. div_sv(up3(w, x))),
        lvl.(w, x, FT(0.1)),
        (@. divf2c(gradc2f(x))),
    )
end

@testset "multi-column FD space (device vs CPU) [$FT]" for FT in (
    Float32,
    Float64,
)
    # MultiColumnFiniteDifferenceSpace is in the eager GPU kernel's supported
    # space families (`Operators.AllFiniteDifferenceSpace`); its `Field`
    # handler must read each leaf field at the thread's level, not misread the
    # space as a level field and read everything at level 1 -- the CPU
    # references vary with z, so that failure mode cannot match them.
    mc_spaces(device) = begin
        points = [
            Geometry.LatLongPoint(FT(10i - 40), FT(20i - 80)) for i in 1:7
        ]
        center_space = PointColumnEnsembleSpace(
            FT;
            points,
            z_elem = 8,
            z_min = FT(0),
            z_max = FT(1000),
            device,
            staggering = CellCenter(),
        )
        (center_space, Spaces.face_space(center_space))
    end
    spaces = mc_spaces(test_device)
    spaces_cpu = mc_spaces(cpu_device)
    results = (
        fd_f2c_results(spaces...)...,
        fd_c2f_results(spaces...)...,
        fd_advection_results(spaces...)...,
    )
    results_cpu = (
        fd_f2c_results(spaces_cpu...)...,
        fd_c2f_results(spaces_cpu...)...,
        fd_advection_results(spaces_cpu...)...,
    )
    for (field, field_cpu) in zip(results, results_cpu)
        @test device_matches_cpu(field, field_cpu; rtol = 10 * eps(FT))
        @test all(isfinite, Array(parent(field)))
    end
end
