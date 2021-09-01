abstract type AbstractFiniteDifferenceSpace <: AbstractSpace end

abstract type Staggering end

""" Cell center location """
struct CellCenter <: Staggering end

""" Cell face location """
struct CellFace <: Staggering end

struct FiniteDifferenceSpace{S <: Staggering, M <: Meshes.IntervalMesh, G} <:
       AbstractFiniteDifferenceSpace
    staggering::S
    mesh::M
    center_local_geometry::G
    face_local_geometry::G
end

function FiniteDifferenceSpace{S}(
    mesh::Meshes.IntervalMesh,
) where {S <: Staggering}
    FT = eltype(mesh)
    face_coordinates = collect(mesh.faces)
    Δh_f2f = diff(face_coordinates)
    CT = eltype(face_coordinates)
    FT = eltype(CT)

    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian3Axis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    }
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Contravariant3Axis, Geometry.Cartesian3Axis},
        SMatrix{1, 1, FT, 1},
    }
    LG = Geometry.LocalGeometry{CT, FT, Mxξ, Mξx}
    nface = length(face_coordinates)
    ncent = nface - 1
    center_local_geometry = DataLayouts.VF{LG}(Array{FT}, ncent)
    face_local_geometry = DataLayouts.VF{LG}(Array{FT}, nface)

    for i in 1:ncent
        # centers
        z⁻ = face_coordinates[i]
        z⁺ = face_coordinates[i + 1]
        # at the moment we use a "discrete Jacobian"
        # ideally we should use the continuous quantity via the derivative of the warp function
        # could we just define this then as deriv on the mesh element coordinates?
        z = (z⁺ + z⁻) / 2
        Δz = z⁺ - z⁻
        J = Δz
        WJ = Δz
        ∂x∂ξ = SMatrix{1, 1}(J)
        ∂ξ∂x = SMatrix{1, 1}(inv(J))
        center_local_geometry[i] = Geometry.LocalGeometry(
            z,
            J,
            WJ,
            Geometry.AxisTensor(
                (Geometry.Cartesian3Axis(), Geometry.Covariant3Axis()),
                ∂x∂ξ,
            ),
            Geometry.AxisTensor(
                (Geometry.Contravariant3Axis(), Geometry.Cartesian3Axis()),
                ∂ξ∂x,
            ),
        )
    end

    for i in 1:nface
        z = face_coordinates[i]
        if i == 1
            # bottom face
            J = face_coordinates[2] - z
            WJ = J / 2
        elseif i == nface
            # top face
            J = z - face_coordinates[i - 1]
            WJ = J / 2
        else
            J = (face_coordinates[i + 1] - face_coordinates[i - 1]) / 2
            WJ = J
        end
        ∂x∂ξ = SMatrix{1, 1}(J)
        ∂ξ∂x = SMatrix{1, 1}(inv(J))
        face_local_geometry[i] = Geometry.LocalGeometry(
            z,
            J,
            WJ,
            Geometry.AxisTensor(
                (Geometry.Cartesian3Axis(), Geometry.Covariant3Axis()),
                ∂x∂ξ,
            ),
            Geometry.AxisTensor(
                (Geometry.Contravariant3Axis(), Geometry.Cartesian3Axis()),
                ∂ξ∂x,
            ),
        )
    end

    return FiniteDifferenceSpace(
        S(),
        mesh,
        center_local_geometry,
        face_local_geometry,
    )
end

const CenterFiniteDifferenceSpace = FiniteDifferenceSpace{CellCenter}
const FaceFiniteDifferenceSpace = FiniteDifferenceSpace{CellFace}

function FiniteDifferenceSpace{S}(
    space::FiniteDifferenceSpace,
) where {S <: Staggering}
    FiniteDifferenceSpace(
        S(),
        space.mesh,
        space.center_local_geometry,
        space.face_local_geometry,
    )
end

Base.length(space::FiniteDifferenceSpace) = length(coordinate_data(space))

nlevels(space::FiniteDifferenceSpace) = length(space)

local_geometry_data(space::CenterFiniteDifferenceSpace) =
    space.center_local_geometry

local_geometry_data(space::FaceFiniteDifferenceSpace) =
    space.face_local_geometry

left_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.mesh.boundaries)[1]

right_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.mesh.boundaries)[2]
