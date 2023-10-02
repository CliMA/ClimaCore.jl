abstract type AbstractFiniteDifferenceGrid end

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

topology(grid::FiniteDifferenceGrid) = grid.topology

ClimaComms.context(grid::FiniteDifferenceGrid) =
    ClimaComms.context(topology(grid))
ClimaComms.device(grid::FiniteDifferenceGrid) =
    ClimaComms.device(topology(grid))


@memoize WeakValueDict function FiniteDifferenceGrid(
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

abstract type AbstractFiniteDifferenceSpace<: AbstractSpace end

"""
    FiniteDifferenceSpace(staggering::Staggering, grid::FiniteDifferenceGrid)

"""
struct FiniteDifferenceSpace{
    S <: Staggering,
    G <: AbstractFiniteDifferenceGrid
} <: AbstractFiniteDifferenceSpace
    staggering::S
    grid::G
end

const FaceFiniteDifferenceSpace = FiniteDifferenceSpace{CellFace}
const CenterFiniteDifferenceSpace = FiniteDifferenceSpace{CellCenter}

function Base.show(io::IO, space::FiniteDifferenceSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(
        io,
        space isa CenterFiniteDifferenceSpace ? "CenterFiniteDifferenceSpace" :
        "FaceFiniteDifferenceSpace",
        ":",
    )
    print(iio, " "^(indent + 2), "context: ")
    Topologies.print_context(iio, ClimaComms.context(space))
    println(iio)
    print(iio, " "^(indent + 2), "mesh: ", topology(space).mesh)
end



FiniteDifferenceSpace{S}(grid::FiniteDifferenceGrid) where {S<:Staggering}=
    FiniteDifferenceSpace(S(), grid)
FiniteDifferenceSpace{S}(space::FiniteDifferenceSpace) where {S<:Staggering}=
    FiniteDifferenceSpace(S(), space.grid)
FiniteDifferenceSpace{S}(topology::Topologies.IntervalTopology) where {S<:Staggering}=
    FiniteDifferenceSpace{S}(FiniteDifferenceGrid(topology))
FiniteDifferenceSpace{S}(mesh::Meshes.IntervalMesh) where {S<:Staggering}=
    FiniteDifferenceSpace{S}(FiniteDifferenceGrid(mesh))

face_space(space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace{CellFace}(space)
center_space(space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace{CellCenter}(space)

Adapt.adapt_structure(to, space::FiniteDifferenceSpace) = FiniteDifferenceSpace(
    space.staggering,
    Adapt.adapt(to, space.topology),
    Adapt.adapt(to, space.global_geometry),
    Adapt.adapt(to, space.center_local_geometry),
    Adapt.adapt(to, space.face_local_geometry),
)

nlevels(space::FiniteDifferenceSpace) = length(space)
# TODO: deprecate?
Base.length(space::FiniteDifferenceSpace) = length(coordinates_data(space))


topology(space::FiniteDifferenceSpace) = topology(space.grid)
vertical_topology(space::FiniteDifferenceSpace) = topology(space.grid)



Base.@deprecate z_component(::Type{T}) where {T} Δz_metric_component(T) false

"""
    Δz_metric_component(::Type{<:Goemetry.AbstractPoint})

The index of the z-component of an abstract point
in an `AxisTensor`.
"""
Δz_metric_component(::Type{<:Geometry.LatLongZPoint}) = 9
Δz_metric_component(::Type{<:Geometry.Cartesian3Point}) = 1
Δz_metric_component(::Type{<:Geometry.Cartesian13Point}) = 4
Δz_metric_component(::Type{<:Geometry.Cartesian123Point}) = 9
Δz_metric_component(::Type{<:Geometry.XYZPoint}) = 9
Δz_metric_component(::Type{<:Geometry.ZPoint}) = 1
Δz_metric_component(::Type{<:Geometry.XZPoint}) = 4

Base.@deprecate dz_data(space::AbstractSpace) Δz_data(space) false

"""
    Δz_data(space::AbstractSpace)

A DataLayout containing the `Δz` on a given space `space`.
"""
function Δz_data(space::AbstractSpace)
    lg = local_geometry_data(space)
    data_layout_type = eltype(lg.coordinates)
    return getproperty(
        lg.∂x∂ξ.components.data,
        Δz_metric_component(data_layout_type),
    )
end

function left_boundary_name(space::AbstractSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
    propertynames(boundaries)[1]
end

function right_boundary_name(space::AbstractSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
    propertynames(boundaries)[2]
end

Base.@propagate_inbounds function level(
    space::FaceFiniteDifferenceSpace,
    v::PlusHalf,
)
    @inbounds local_geometry = level(local_geometry_data(space), v.i + 1)
    PointSpace(local_geometry)
end
Base.@propagate_inbounds function level(
    space::CenterFiniteDifferenceSpace,
    v::Int,
)
    local_geometry = level(local_geometry_data(space), v)
    PointSpace(local_geometry)
end
