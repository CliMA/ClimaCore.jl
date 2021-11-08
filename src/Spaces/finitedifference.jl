abstract type AbstractFiniteDifferenceSpace <: AbstractSpace end

abstract type Staggering end

""" Cell center location """
struct CellCenter <: Staggering end

""" Cell face location """
struct CellFace <: Staggering end

struct FiniteDifferenceSpace{
    S <: Staggering,
    T <: Topologies.IntervalTopology,
    GG,
    LG,
} <: AbstractFiniteDifferenceSpace
    staggering::S
    topology::T
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end

function FiniteDifferenceSpace{S}(
    topology::Topologies.IntervalTopology,
) where {S <: Staggering}
    global_geometry = Geometry.CartesianGlobalGeometry()
    mesh = topology.mesh
    CT = Meshes.coordinate_type(mesh)
    AIdx = Geometry.coordinate_axis(CT)
    # TODO: FD operators  hardcoded to work over the 3-axis, need to generalize
    # similar to spectral operators
    @assert AIdx == (3,) "FiniteDifference operations only work over the 3-axis (ZPoint) domain"
    FT = eltype(CT)
    face_coordinates = collect(mesh.faces)
    LG = Geometry.LocalGeometry{AIdx, CT, FT, SMatrix{1, 1, FT, 1}}
    nface = length(face_coordinates)
    ncent = nface - 1
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
            coord⁺ = Geometry.component(face_coordinates[2], 1)
            J = coord⁺ - coord
            WJ = J / 2
        elseif i == nface
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
    return FiniteDifferenceSpace(
        S(),
        topology,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
end

FiniteDifferenceSpace{S}(mesh::Meshes.IntervalMesh) where {S <: Staggering} =
    FiniteDifferenceSpace{S}(Topologies.IntervalTopology(mesh))


const CenterFiniteDifferenceSpace = FiniteDifferenceSpace{CellCenter}
const FaceFiniteDifferenceSpace = FiniteDifferenceSpace{CellFace}

function FiniteDifferenceSpace{S}(
    space::FiniteDifferenceSpace,
) where {S <: Staggering}
    FiniteDifferenceSpace(
        S(),
        space.topology,
        space.global_geometry,
        space.center_local_geometry,
        space.face_local_geometry,
    )
end

Base.length(space::FiniteDifferenceSpace) = length(coordinates_data(space))

nlevels(space::FiniteDifferenceSpace) = length(space)

local_geometry_data(space::CenterFiniteDifferenceSpace) =
    space.center_local_geometry

local_geometry_data(space::FaceFiniteDifferenceSpace) =
    space.face_local_geometry

left_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.topology.boundaries)[1]

right_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.topology.boundaries)[2]
