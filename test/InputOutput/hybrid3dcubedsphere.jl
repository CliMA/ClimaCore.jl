using Test
import ClimaCore
using ClimaCore:
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    DataLayouts,
    InputOutput

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())
pid, nprocs = ClimaComms.init(comms_ctx)
filename = ClimaComms.bcast(comms_ctx, tempname(pwd()))
if ClimaComms.iamroot(comms_ctx)
    @info "Comms context" comms_ctx nprocs filename
end

@testset "HDF5 restart test for 3d hybrid cubed sphere" begin
    FT = Float32
    R = FT(6.371229e6)

    npoly = 4
    z_max = FT(30e3)
    z_elem = 10
    h_elem = 4
    # horizontal space
    domain = Domains.SphereDomain(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    topology = Topologies.Topology2D(
        comms_ctx,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(topology, quad)
    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    # Extruded 3D space
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)

    Y = Fields.FieldVector(c = ᶜlocal_geometry, f = ᶠlocal_geometry)

    # write field vector to hdf5 file
    writer = InputOutput.HDF5Writer(filename, comms_ctx)
    InputOutput.write!(writer, Y, "Y")
    close(writer)

    reader = InputOutput.HDF5Reader(filename, comms_ctx)
    restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
    close(reader)
    @test restart_Y == Y # test if restart is exact
end
