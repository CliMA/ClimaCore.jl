
abstract type Staggering end

"""
    CellCenter()

Cell center location
"""
struct CellCenter <: Staggering end

"""
    CellFace()

Cell face location
"""
struct CellFace <: Staggering end






abstract type AbstractFiniteDifferenceGrid <: AbstractGrid end

"""
    FiniteDifferenceGrid(topology::Topologies.IntervalTopology)
    FiniteDifferenceGrid(mesh::Meshes.IntervalMesh)

Construct a `FiniteDifferenceGrid` from an `IntervalTopology` (or an
`IntervalMesh`).

This is an object which contains all the necessary geometric information.

To avoid unnecessary duplication, we memoize the construction of the grid.
"""
mutable struct FiniteDifferenceGrid{
    T <: Topologies.AbstractIntervalTopology,
    GG,
    LG,
} <: AbstractFiniteDifferenceGrid
    topology::T
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end


@memoize IdDict function FiniteDifferenceGrid(
    topology::Topologies.IntervalTopology,
)
    global_geometry = Geometry.CartesianGlobalGeometry()
    mesh = topology.mesh
    CT = Meshes.coordinate_type(mesh)
    AIdx = Geometry.coordinate_axis(CT)
    # TODO: FD operators  hardcoded to work over the 3-axis, need to generalize
    # similar to spectral operators
    @assert AIdx == (3,) "FiniteDifference operations only work over the 3-axis (ZPoint) domain"
    FT = eltype(CT)
    ArrayType = ClimaComms.array_type(topology)
    face_coordinates = collect(mesh.faces)
    LG = Geometry.LocalGeometry{AIdx, CT, FT, SMatrix{1, 1, FT, 1}}
    nface = length(face_coordinates) - Topologies.isperiodic(topology)
    ncent = length(face_coordinates) - 1
    # contstruct on CPU, copy to device at end
    center_local_geometry = DataLayouts.VF{LG}(Array{FT}, ncent)
    face_local_geometry = DataLayouts.VF{LG}(Array{FT}, nface)
    for i in 1:ncent
        # centers
        coord⁻ = Geometry.component(face_coordinates[i], 1)
        coord⁺ = Geometry.component(face_coordinates[i + 1], 1)
        # at the moment we use a "discrete Jacobian"
        # ideally we should use the continuous quantity via the derivative of the warp function
        # could we just define this then as deriv on the mesh element coordinates?
        coord = (coord⁺ + coord⁻) / 2
        Δcoord = coord⁺ - coord⁻
        J = Δcoord
        WJ = Δcoord
        ∂x∂ξ = SMatrix{1, 1}(J)
        center_local_geometry[i] = Geometry.LocalGeometry(
            CT(coord),
            J,
            WJ,
            Geometry.AxisTensor(
                (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
                ∂x∂ξ,
            ),
        )
    end
    for i in 1:nface
        coord = Geometry.component(face_coordinates[i], 1)
        if i == 1
            # bottom face
            if Topologies.isperiodic(topology)
                Δcoord⁺ =
                    Geometry.component(face_coordinates[2], 1) -
                    Geometry.component(face_coordinates[1], 1)
                Δcoord⁻ =
                    Geometry.component(face_coordinates[end], 1) -
                    Geometry.component(face_coordinates[end - 1], 1)
                J = (Δcoord⁺ + Δcoord⁻) / 2
                WJ = J
            else
                coord⁺ = Geometry.component(face_coordinates[2], 1)
                J = coord⁺ - coord
                WJ = J / 2
            end
        elseif !Topologies.isperiodic(topology) && i == nface
            # top face
            coord⁻ = Geometry.component(face_coordinates[i - 1], 1)
            J = coord - coord⁻
            WJ = J / 2
        else
            coord⁺ = Geometry.component(face_coordinates[i + 1], 1)
            coord⁻ = Geometry.component(face_coordinates[i - 1], 1)
            J = (coord⁺ - coord⁻) / 2
            WJ = J
        end
        ∂x∂ξ = SMatrix{1, 1}(J)
        ∂ξ∂x = SMatrix{1, 1}(inv(J))
        face_local_geometry[i] = Geometry.LocalGeometry(
            CT(coord),
            J,
            WJ,
            Geometry.AxisTensor(
                (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
                ∂x∂ξ,
            ),
        )
    end
    return FiniteDifferenceGrid(
        topology,
        global_geometry,
        Adapt.adapt(ArrayType, center_local_geometry),
        Adapt.adapt(ArrayType, face_local_geometry),
    )
end


FiniteDifferenceGrid(mesh::Meshes.IntervalMesh) =
    FiniteDifferenceGrid(Topologies.IntervalTopology(mesh))

# accessors
topology(grid::FiniteDifferenceGrid) = grid.topology
vertical_topology(grid::FiniteDifferenceGrid) = grid.topology

local_geometry_data(grid::FiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::FiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::FiniteDifferenceGrid) = grid.global_geometry

## GPU compatibility
struct DeviceFiniteDifferenceGrid{T, GG, LG} <: AbstractFiniteDifferenceGrid
    topology::T
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end

Adapt.adapt_structure(to, grid::FiniteDifferenceGrid) =
    DeviceFiniteDifferenceGrid(
        Adapt.adapt(to, grid.topology),
        Adapt.adapt(to, grid.global_geometry),
        Adapt.adapt(to, grid.center_local_geometry),
        Adapt.adapt(to, grid.face_local_geometry),
    )

topology(grid::DeviceFiniteDifferenceGrid) = grid.topology
vertical_topology(grid::DeviceFiniteDifferenceGrid) = grid.topology

local_geometry_data(grid::DeviceFiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::DeviceFiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::DeviceFiniteDifferenceGrid) = grid.global_geometry
