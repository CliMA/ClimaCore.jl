
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


function FiniteDifferenceGrid(topology::Topologies.IntervalTopology)
    get!(Cache.OBJECT_CACHE, (FiniteDifferenceGrid, topology)) do
        _FiniteDifferenceGrid(topology)
    end
end

function _FiniteDifferenceGrid(topology::Topologies.IntervalTopology)
    global_geometry = Geometry.CartesianGlobalGeometry()
    ArrayType = ClimaComms.array_type(topology)
    face_coordinates = collect(mesh.faces)
    # construct on CPU, adapt to GPU
    center_local_geometry, face_local_geometry = fd_geometry_data(
        face_coordinates;
        periodic = Topologies.isperiodic(topology),
    )
    return FiniteDifferenceGrid(
        topology,
        global_geometry,
        Adapt.adapt(ArrayType, center_local_geometry),
        Adapt.adapt(ArrayType, face_local_geometry),
    )
end

function fd_geometry_data(
    face_coordinates::AbstractVector{Geometry.ZPoint{FT}};
    periodic,
) where {FT}
    AIdx == (3,)
    CT = Geometry.ZPoint{FT}
    LG = Geometry.LocalGeometry{AIdx, CT, FT, SMatrix{1, 1, FT, 1}}

    nface = length(face_coordinates) - periodic
    ncent = length(face_coordinates) - 1
    center_local_geometry = DataLayouts.VF{LG}(Array{FT}, ncent)
    face_local_geometry = DataLayouts.VF{LG}(Array{FT}, nface)

    for i in 1:ncent
        # centers
        coord⁻ = Geometry.component(face_coordinates[i], 1)
        coord⁺ = Geometry.component(face_coordinates[i + 1], 1)
        # use a "discrete Jacobian"
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
            if periodic
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
        elseif i == ncent + 1
            @assert !periodic
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
    return (center_local_geometry, face_local_geometry)
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
