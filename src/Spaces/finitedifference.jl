abstract type Staggering end

""" Cell center location """
struct CellCenter <: Staggering end

""" Cell face location """
struct CellFace <: Staggering end

struct FiniteDifferenceSpace{S <: Staggering, M <: Meshes.IntervalMesh, C, H} <:
       AbstractSpace
    staggering::S
    mesh::M
    nhalo::Int
    center_coordinates::C
    face_coordinates::C
    Δh_f2f::H # lives at cell center coordinates
    Δh_c2c::H # lives at cell face coordinates
end

function FiniteDifferenceSpace{S}(
    mesh::Meshes.IntervalMesh,
    nhalo::Integer = 0,
) where {S <: Staggering}
    if nhalo < 0
        throw(ArgumentError("nhalo must be ≥ 0"))
    end
    FT = eltype(mesh)
    face_coordinates = collect(mesh.faces)
    face_Δh_left = face_coordinates[2] - face_coordinates[1]
    face_Δh_right = face_coordinates[end] - face_coordinates[end - 1]
    for _ in 1:nhalo
        pushfirst!(face_coordinates, face_coordinates[1] - face_Δh_left)
        push!(face_coordinates, face_coordinates[end] - face_Δh_right)
    end
    Δh_f2f = diff(face_coordinates)
    center_coordinates = [
        (face_coordinates[i] + face_coordinates[i + 1]) / 2 for
        i in eachindex(Δh_f2f)
    ]
    Δh_c2c = diff(center_coordinates)
    pushfirst!(Δh_c2c, center_coordinates[1] - face_coordinates[1])
    push!(Δh_c2c, face_coordinates[end] - center_coordinates[end])
    return FiniteDifferenceSpace(
        S(),
        mesh,
        nhalo,
        DataLayouts.VF{FT}(center_coordinates),
        DataLayouts.VF{FT}(face_coordinates),
        DataLayouts.VF{FT}(Δh_f2f),
        DataLayouts.VF{FT}(Δh_c2c),
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
        space.nhalo,
        space.center_coordinates,
        space.face_coordinates,
        space.Δh_f2f,
        space.Δh_c2c,
    )
end

Base.length(space::FiniteDifferenceSpace) = length(coordinates(space))

coordinates(space::CenterFiniteDifferenceSpace) = space.center_coordinates
coordinates(space::FaceFiniteDifferenceSpace) = space.face_coordinates

coordinate(space::CenterFiniteDifferenceSpace, idx) =
    space.center_coordinates[idx]

coordinate(space::FaceFiniteDifferenceSpace, idx) = space.face_coordinates[idx]

Δcoordinate(space::CenterFiniteDifferenceSpace, idx) = space.Δh_c2c[idx]
Δcoordinate(space::FaceFiniteDifferenceSpace, idx) = space.Δh_f2f[idx]

real_indices(space::CenterFiniteDifferenceSpace) = range(
    space.nhalo + 1,
    length(space.center_coordinates) - space.nhalo,
    step = 1,
)

real_indices(space::FaceFiniteDifferenceSpace) = range(
    space.nhalo + 1,
    length(space.face_coordinates) - space.nhalo,
    step = 1,
)

interior_indices(space::CenterFiniteDifferenceSpace) = real_indices(space)

interior_indices(space::FaceFiniteDifferenceSpace) =
    real_indices(space)[2:(end - 1)]

@inline left_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.mesh.boundaries)[1]
@inline right_boundary_name(space::FiniteDifferenceSpace) =
    propertynames(space.mesh.boundaries)[2]
