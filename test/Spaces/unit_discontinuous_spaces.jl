# Asserts the discontinuous-grid (DG) contract: `discretization = Grids.DG()`
# marks a spectral-element grid as DG, `discretization`/`is_continuous` report
# it (also through extruded spaces), and `weighted_dss!` is the identity there
# while changing perimeter values on the continuous twin of the same topology.
using Test
using Random
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Grids,
    Meshes,
    Quadratures,
    Spaces,
    Topologies,
    InputOutput
import ClimaCore.Utilities.Cache

FT = Float64

function sphere_topology(FT; helem = 4)
    context = ClimaComms.context()
    domain = Domains.SphereDomain(FT(6.371e6))
    mesh = Meshes.EquiangularCubedSphere(domain, helem)
    return Topologies.Topology2D(context, mesh)
end

function extruded_spaces(hspace; zelem = 4)
    FT = Spaces.undertype(hspace)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(FT(30e3));
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = zelem)
    device = ClimaComms.device(ClimaComms.context())
    vspace = Spaces.CenterFiniteDifferenceSpace(device, vmesh)
    center = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    return center, Spaces.FaceExtrudedFiniteDifferenceSpace(center)
end

topology = sphere_topology(FT)
quad = Quadratures.GLL{4}()
cg_space = Spaces.SpectralElementSpace2D(topology, quad)
dg_space = Spaces.SpectralElementSpace2D(
    topology,
    quad;
    discretization = Grids.DG(),
)

@testset "discretization, is_continuous, and grid caching" begin
    @test Spaces.discretization(cg_space) === Grids.CG()
    @test Spaces.discretization(dg_space) === Grids.DG()
    @test Spaces.is_continuous(cg_space)
    @test !Spaces.is_continuous(dg_space)
    # the discretization is part of the grid cache key
    @test Spaces.grid(cg_space) !== Spaces.grid(dg_space)
    @test Spaces.grid(dg_space).dss_weights === nothing

    # GL quadrature has no shared boundary nodes, so the stored discretization
    # normalizes to DG even when constructed as CG (the default)
    gl_space = Spaces.SpectralElementSpace2D(topology, Quadratures.GL{4}())
    @test Spaces.discretization(gl_space) === Grids.DG()
    @test !Spaces.is_continuous(gl_space)

    cg_center, cg_face = extruded_spaces(cg_space)
    dg_center, dg_face = extruded_spaces(dg_space)
    # extruded spaces forward to the horizontal grid
    @test Spaces.discretization(cg_center) === Grids.CG()
    @test Spaces.discretization(dg_center) === Grids.DG()
    @test Spaces.is_continuous(cg_center) && Spaces.is_continuous(cg_face)
    @test !Spaces.is_continuous(dg_center) && !Spaces.is_continuous(dg_face)

    # spaces without horizontal spectral elements are continuous
    device = ClimaComms.device(ClimaComms.context())
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(FT(1));
        boundary_names = (:bottom, :top),
    )
    vspace = Spaces.CenterFiniteDifferenceSpace(
        device,
        Meshes.IntervalMesh(vdomain, nelems = 4),
    )
    @test Spaces.discretization(vspace) === Grids.CG()
    @test Spaces.is_continuous(vspace)
end

# A coordinate-seeded field is smooth but its nodal values differ across
# element boundaries only after perturbation; perturb per-node so DSS on the
# CG twin must change the perimeter values.
function perturbed_field(space)
    coords = Fields.coordinate_field(space)
    f = @. sind(coords.long) * cosd(coords.lat)^2
    Random.seed!(1)
    perturbation = similar(f)
    parent(perturbation) .=
        ClimaComms.array_type(ClimaComms.device(ClimaComms.context()))(
            rand(FT, size(parent(perturbation))),
        )
    return f .+ perturbation
end

@testset "weighted_dss! is the identity on discontinuous spaces" begin
    f_dg = perturbed_field(dg_space)
    f_before = copy(parent(f_dg))
    @test Spaces.create_dss_buffer(f_dg) === nothing
    Spaces.weighted_dss!(f_dg)
    @test parent(f_dg) == f_before

    f_cg = perturbed_field(cg_space)
    f_before = copy(parent(f_cg))
    Spaces.weighted_dss!(f_cg)
    @test parent(f_cg) != f_before

    # the split (start/internal/ghost) and pair paths are gated too
    f_dg2 = perturbed_field(dg_space)
    f_before = copy(parent(f_dg2))
    buffer = Spaces.create_dss_buffer(f_dg2)
    Spaces.weighted_dss_start!(f_dg2, buffer)
    Spaces.weighted_dss_internal!(f_dg2, buffer)
    Spaces.weighted_dss_ghost!(f_dg2, buffer)
    @test parent(f_dg2) == f_before
    Spaces.weighted_dss!(f_dg2 => buffer)
    @test parent(f_dg2) == f_before
end

# Round-trip a field through HDF5 and report the restart space's continuity.
# The grids under test are evicted from the object cache first, so the reader
# reconstructs them from the file attributes rather than returning the cached
# objects.
function roundtrip_is_continuous(space)
    context = ClimaComms.context(space)
    filename = tempname(; cleanup = true)
    f = Fields.local_geometry_field(space).J
    InputOutput.HDF5Writer(filename, context) do writer
        InputOutput.write!(writer, "f" => f)
    end
    grid = Spaces.grid(space)
    Cache.clean_cache!(grid)
    grid isa Grids.ExtrudedFiniteDifferenceGrid &&
        Cache.clean_cache!(grid.horizontal_grid)
    InputOutput.HDF5Reader(filename, context) do reader
        Spaces.is_continuous(axes(InputOutput.read_field(reader, "f")))
    end
end

function plane_hspace(FT; discretization)
    context = ClimaComms.context()
    hdomain = Domains.IntervalDomain(
        Geometry.XPoint(zero(FT)),
        Geometry.XPoint(FT(1));
        periodic = true,
    )
    hmesh = Meshes.IntervalMesh(hdomain, nelems = 4)
    htopology = Topologies.IntervalTopology(ClimaComms.device(context), hmesh)
    return Spaces.SpectralElementSpace1D(
        htopology,
        Quadratures.GLL{4}();
        discretization,
    )
end

@testset "discretization survives an InputOutput round-trip" begin
    @test !roundtrip_is_continuous(dg_space)
    @test roundtrip_is_continuous(cg_space)
    # extruded plane spaces cover the SpectralElementGrid1D writer/reader
    dg_plane, _ =
        extruded_spaces(plane_hspace(FT; discretization = Grids.DG()))
    cg_plane, _ =
        extruded_spaces(plane_hspace(FT; discretization = Grids.CG()))
    @test !roundtrip_is_continuous(dg_plane)
    @test roundtrip_is_continuous(cg_plane)
end

@testset "legacy \"discontinuous\" attribute reads back as DG" begin
    # Files written before "discretization" replaced the "discontinuous"
    # attribute must still restore their DG grids.
    context = ClimaComms.context()
    filename = tempname(; cleanup = true)
    f = Fields.local_geometry_field(dg_space).J
    InputOutput.HDF5Writer(filename, context) do writer
        InputOutput.write!(writer, "f" => f)
    end
    InputOutput.HDF5.h5open(filename, "r+") do file
        for name in keys(file["grids"])
            group = file["grids"][name]
            grid_attrs = InputOutput.HDF5.attrs(group)
            haskey(grid_attrs, "discretization") || continue
            InputOutput.HDF5.delete_attribute(group, "discretization")
            grid_attrs["discontinuous"] = "true"
        end
    end
    Cache.clean_cache!(Spaces.grid(dg_space))
    InputOutput.HDF5Reader(filename, context) do reader
        @test !Spaces.is_continuous(axes(InputOutput.read_field(reader, "f")))
    end
end

@testset "CG and DG grids round-trip through a single file" begin
    # Both grids request the group name "horizontal_grid"; the writer must
    # give the second one a distinct group instead of aliasing it to the
    # first, or one field restarts with the wrong discretization.
    context = ClimaComms.context()
    filename = tempname(; cleanup = true)
    f_dg = Fields.local_geometry_field(dg_space).J
    f_cg = Fields.local_geometry_field(cg_space).J
    InputOutput.HDF5Writer(filename, context) do writer
        InputOutput.write!(writer, "f_dg" => f_dg, "f_cg" => f_cg)
    end
    Cache.clean_cache!(Spaces.grid(dg_space))
    Cache.clean_cache!(Spaces.grid(cg_space))
    InputOutput.HDF5Reader(filename, context) do reader
        restart_dg = InputOutput.read_field(reader, "f_dg")
        restart_cg = InputOutput.read_field(reader, "f_cg")
        @test !Spaces.is_continuous(axes(restart_dg))
        @test Spaces.is_continuous(axes(restart_cg))
    end
end
