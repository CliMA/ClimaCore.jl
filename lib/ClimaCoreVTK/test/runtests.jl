using Test
using ClimaCoreVTK
using IntervalSets
import ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators


dir = mktempdir()

@testset "sphere" begin
    R = 6.37122e6

    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    grid_topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{5}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    u = map(coords) do coord
        u0 = 20.0
        α0 = 45.0
        ϕ = coord.lat
        λ = coord.long

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end

    writevtk(joinpath(dir, "sphere"), (coords = coords, u = u); basis = :point)

    times = 0:10:350
    A = [
        map(coords) do coord
            cosd(coord.lat) *
            (sind(coord.long) * sind(α) + cosd(coord.long) * cosd(α))
        end for α in times
    ]
    writevtk(joinpath(dir, "sphere_scalar_series"), times, (A = A, B = A))

    U = Array{Fields.Field}(undef, length(times))
    for t in 1:(div(350, 10) + 1)
        u = map(coords) do coord
            u0 = 20.0
            α0 = 45.0
            ϕ = coord.lat
            λ = coord.long

            uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
            uv = -u0 * sind(α0) * sind(λ)
            Geometry.UVVector(uu, uv)
        end
        U[t] = u
    end
    writevtk(joinpath(dir, "sphere_vector_series"), times, (U = U,))

end

@testset "rectangle" begin

    domain = Domains.RectangleDomain(
        Geometry.XPoint(0) .. Geometry.XPoint(2π),
        Geometry.YPoint(0) .. Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )

    n1, n2 = 4, 4
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    coords = Fields.coordinate_field(space)
    sinxy = map(coords) do coord
        sin(coord.x + coord.y)
    end
    u = map(coords) do coord
        Geometry.UVVector(-coord.y, coord.x)
    end

    writevtk(joinpath(dir, "plane"), (sinxy = sinxy, u = u))

end


@testset "hybrid 2d" begin
    hdomain = Domains.IntervalDomain(
        Geometry.XPoint(0) .. Geometry.XPoint(10.0),
        periodic = true,
    )
    hmesh = Meshes.IntervalMesh(hdomain, nelems = 10)
    htopology = Topologies.IntervalTopology(hmesh)
    quad = Spaces.Quadratures.GLL{4}()
    hspace = Spaces.SpectralElementSpace1D(htopology, quad)

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0) .. Geometry.ZPoint(20.0),
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = 20)
    vtopology = Topologies.IntervalTopology(vmesh)
    vspace = Spaces.FaceFiniteDifferenceSpace(vtopology)

    fspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    cspace = Spaces.CenterExtrudedFiniteDifferenceSpace(fspace)
    writevtk(
        joinpath(dir, "hybrid2d_point"),
        Fields.coordinate_field(fspace);
        basis = :point,
    )
    writevtk(
        joinpath(dir, "hybrid2d_cell"),
        Fields.coordinate_field(cspace);
        basis = :cell,
    )
end

@testset "hybrid 3d" begin

    hdomain = Domains.RectangleDomain(
        Geometry.XPoint(0) .. Geometry.XPoint(2π),
        Geometry.YPoint(0) .. Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )

    hmesh = Meshes.RectilinearMesh(hdomain, 4, 4)
    htopology = Topologies.Topology2D(hmesh)
    quad = Spaces.Quadratures.GLL{4}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0) .. Geometry.ZPoint(20.0),
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = 20)
    vtopology = Topologies.IntervalTopology(vmesh)
    vspace = Spaces.FaceFiniteDifferenceSpace(vtopology)

    fspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    cspace = Spaces.CenterExtrudedFiniteDifferenceSpace(fspace)
    writevtk(
        joinpath(dir, "hybrid3d_point"),
        Fields.coordinate_field(fspace);
        basis = :point,
    )
    writevtk(
        joinpath(dir, "hybrid3d_cell"),
        Fields.coordinate_field(cspace);
        basis = :cell,
    )
end
