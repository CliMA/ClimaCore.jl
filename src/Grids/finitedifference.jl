
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
    FiniteDifferenceGrid(device::ClimaComms.AbstractDevice, mesh::Meshes.IntervalMesh)

Construct a `FiniteDifferenceGrid` from an `IntervalTopology` (or an
`IntervalMesh`).

This is an object which contains all the necessary geometric information.

To avoid unnecessary duplication, we memoize the construction of the grid.
"""
mutable struct FiniteDifferenceGrid{
    T <: Topologies.AbstractIntervalTopology,
    GG,
    CLG,
    FLG,
} <: AbstractFiniteDifferenceGrid
    topology::T
    global_geometry::GG
    center_local_geometry::CLG
    face_local_geometry::FLG
end
Adapt.@adapt_structure FiniteDifferenceGrid

function FiniteDifferenceGrid(topology::Topologies.IntervalTopology)
    get!(Cache.OBJECT_CACHE, (FiniteDifferenceGrid, topology)) do
        _FiniteDifferenceGrid(topology)
    end
end

function _FiniteDifferenceGrid(topology::Topologies.IntervalTopology)
    global_geometry = Geometry.CartesianGlobalGeometry()
    ArrayType = ClimaComms.array_type(topology)

    mesh = Topologies.mesh(topology)
    CT = Topologies.coordinate_type(mesh)
    FT = Geometry.float_type(CT)
    Nv_face = length(mesh.faces)
    Nv_cent = Nv_face - 1
    # construct on CPU, adapt to GPU
    center_coordinates = DataLayouts.VF{CT, Nv_cent}(Array{FT}, Nv_cent)
    face_coordinates = DataLayouts.VF{CT, Nv_face}(Array{FT}, Nv_face)
    for v in 1:Nv_cent
        center_coordinates[vindex(v)] = (mesh.faces[v] + mesh.faces[v + 1]) / 2
    end
    for v in 1:Nv_face
        face_coordinates[vindex(v)] = mesh.faces[v]
    end
    center_local_geometry, face_local_geometry = fd_geometry_data(
        center_coordinates,
        face_coordinates,
        Val(Topologies.isperiodic(topology)),
    )

    return FiniteDifferenceGrid(
        topology,
        global_geometry,
        Adapt.adapt(ArrayType, center_local_geometry),
        Adapt.adapt(ArrayType, face_local_geometry),
    )
end

# called by the FiniteDifferenceGrid constructor, and the ExtrudedFiniteDifferenceGrid constructor with Hypsography
function fd_geometry_data(
    center_coordinates::DataLayouts.AbstractData{Geometry.ZPoint{FT}},
    face_coordinates::DataLayouts.AbstractData{Geometry.ZPoint{FT}},
    ::Val{periodic},
) where {FT, periodic}
    CT = Geometry.ZPoint{FT}
    AIdx = (3,)
    LG = Geometry.FullLocalGeometry{AIdx, CT, FT, SMatrix{1, 1, FT, 1}}
    ∂x∂ξ_axes = (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}())
    (Ni, Nj, Nk, Nv, Nh) = size(face_coordinates)
    Nv_face = Nv - periodic
    Nv_cent = Nv - 1
    center_local_geometry = similar(center_coordinates, LG, Val(Nv_cent))
    face_local_geometry = similar(face_coordinates, LG, Val(Nv_face))
    cent_coord(args...) =
        Geometry.component(center_coordinates[CartesianIndex(args...)], 1)
    face_coord(args...) =
        Geometry.component(face_coordinates[CartesianIndex(args...)], 1)
    for h in 1:Nh, k in 1:Nk, j in 1:Nj, i in 1:Ni
        for v in 1:Nv_cent
            J = face_coord(i, j, k, v + 1, h) - face_coord(i, j, k, v, h)
            WJ = J
            x = CT(cent_coord(i, j, k, v, h))
            ∂x∂ξ = Geometry.AxisTensor(∂x∂ξ_axes, SMatrix{1, 1}(J))
            center_local_geometry[CartesianIndex(i, j, k, v, h)] =
                Geometry.LocalGeometry(x, J, WJ, ∂x∂ξ)
        end
        for v in 1:Nv_face
            if periodic && v == 1
                # periodic boundary face
                J⁺ = face_coord(i, j, k, 2, h) - face_coord(i, j, k, 1, h)
                J⁻ = face_coord(i, j, k, Nv, h) - face_coord(i, j, k, Nv - 1, h)
                J = (J⁺ + J⁻) / 2
                WJ = J
            elseif !periodic && v == 1
                # bottom face
                J = face_coord(i, j, k, 2, h) - face_coord(i, j, k, 1, h)
                WJ = J / 2
            elseif v == Nv
                # top face
                @assert !periodic
                J = face_coord(i, j, k, Nv, h) - face_coord(i, j, k, Nv - 1, h)
                WJ = J / 2
            else
                J = cent_coord(i, j, k, v, h) - cent_coord(i, j, k, v - 1, h)
                WJ = J
            end
            x = CT(face_coord(i, j, k, v, h))
            ∂x∂ξ = Geometry.AxisTensor(∂x∂ξ_axes, SMatrix{1, 1}(J))
            face_local_geometry[CartesianIndex(i, j, k, v, h)] =
                Geometry.LocalGeometry(x, J, WJ, ∂x∂ξ)
        end
    end
    return (center_local_geometry, face_local_geometry)
end

function fd_geometry_data(
    center_coordinates::DataLayouts.AbstractData{Geometry.PPoint{FT}},
    face_coordinates::DataLayouts.AbstractData{Geometry.PPoint{FT}},
    ::Val{periodic},
) where {FT, periodic}
    CT = Geometry.PPoint{FT}

    LG = Geometry.CoordinateOnlyGeometry{CT}
    (Ni, Nj, Nk, Nv, Nh) = size(face_coordinates)
    Nv_face = Nv - periodic
    Nv_cent = Nv - 1
    center_local_geometry = similar(center_coordinates, LG, Val(Nv_cent))
    face_local_geometry = similar(face_coordinates, LG, Val(Nv_face))
    cent_coord(args...) =
        Geometry.component(center_coordinates[CartesianIndex(args...)], 1)
    face_coord(args...) =
        Geometry.component(face_coordinates[CartesianIndex(args...)], 1)
    for h in 1:Nh, k in 1:Nk, j in 1:Nj, i in 1:Ni
        for v in 1:Nv_cent
            x = CT(cent_coord(i, j, k, v, h))
            center_local_geometry[CartesianIndex(i, j, k, v, h)] =
                Geometry.CoordinateOnlyGeometry(x)
        end
        for v in 1:Nv_face
            x = CT(face_coord(i, j, k, v, h))
            face_local_geometry[CartesianIndex(i, j, k, v, h)] =
                Geometry.CoordinateOnlyGeometry(x)
        end
    end
    return (center_local_geometry, face_local_geometry)
end

FiniteDifferenceGrid(
    device::ClimaComms.AbstractDevice,
    mesh::Meshes.IntervalMesh,
) = FiniteDifferenceGrid(Topologies.IntervalTopology(device, mesh))

# accessors
topology(grid::FiniteDifferenceGrid) = grid.topology
vertical_topology(grid::FiniteDifferenceGrid) = grid.topology

local_geometry_type(
    ::Type{FiniteDifferenceGrid{T, GG, CLG, FLG}},
) where {T, GG, CLG, FLG} = eltype(CLG) # calls eltype from DataLayouts

local_geometry_data(grid::FiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::FiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::FiniteDifferenceGrid) = grid.global_geometry

## GPU compatibility
struct DeviceFiniteDifferenceGrid{T, GG, CLG, FLG} <:
       AbstractFiniteDifferenceGrid
    topology::T
    global_geometry::GG
    center_local_geometry::CLG
    face_local_geometry::FLG
end

local_geometry_type(
    ::Type{DeviceFiniteDifferenceGrid{T, GG, CLG, FLG}},
) where {T, GG, CLG, FLG} = eltype(CLG) # calls eltype from DataLayouts

topology(grid::DeviceFiniteDifferenceGrid) = grid.topology
vertical_topology(grid::DeviceFiniteDifferenceGrid) = grid.topology

local_geometry_data(grid::DeviceFiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::DeviceFiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::DeviceFiniteDifferenceGrid) = grid.global_geometry
