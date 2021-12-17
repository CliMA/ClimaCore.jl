"""
    RectilinearMesh <: AbstractMesh2D

# Constructors

    RectilinearMesh(domain::RectangleDomain, n1, n2)

Construct a `RectilinearMesh` of equally-spaced `n1` by `n2` elements on `domain`.

    RectilinearMesh(intervalmesh1::IntervalMesh1, intervalmesh2::IntervalMesh2)

Construct the product mesh of `intervalmesh1` and `intervalmesh2`.
"""
struct RectilinearMesh{I1 <: IntervalMesh, I2 <: IntervalMesh} <: AbstractMesh2D
    intervalmesh1::I1
    intervalmesh2::I2
end
RectilinearMesh(domain::RectangleDomain, n1, n2) = RectilinearMesh(
    IntervalMesh(domain.interval1; nelems = n1),
    IntervalMesh(domain.interval2; nelems = n2),
)

function Base.show(io::IO, mesh::RectilinearMesh)
    n1 = nelements(mesh.intervalmesh1)
    n2 = nelements(mesh.intervalmesh2)
    print(io, n1, "×", n2, "-element RectilinearMesh of ", domain(mesh))
end
domain(mesh::RectilinearMesh) =
    RectangleDomain(domain(mesh.intervalmesh1), domain(mesh.intervalmesh2))
nelements(mesh::RectilinearMesh) =
    nelements(mesh.intervalmesh1) * nelements(mesh.intervalmesh2)
function elements(mesh::RectilinearMesh)
    # we use the Base Julia CartesianIndices object to index elements in the mesh
    CartesianIndices((
        elements(mesh.intervalmesh1),
        elements(mesh.intervalmesh2),
    ))
end
function is_boundary_face(mesh::RectilinearMesh, elem::CartesianIndex{2}, face)
    @assert 1 <= face <= 4
    x1, x2 = elem.I
    if face == 1
        return is_boundary_face(mesh.intervalmesh2, x2, 1)
    elseif face == 2
        return is_boundary_face(mesh.intervalmesh1, x1, 2)
    elseif face == 3
        return is_boundary_face(mesh.intervalmesh2, x2, 2)
    else
        return is_boundary_face(mesh.intervalmesh1, x1, 1)
    end
end
function boundary_face_name(
    mesh::RectilinearMesh,
    elem::CartesianIndex{2},
    face,
)
    @assert 1 <= face <= 4
    x1, x2 = elem.I
    if face == 1
        return boundary_face_name(mesh.intervalmesh2, x2, 1)
    elseif face == 2
        return boundary_face_name(mesh.intervalmesh1, x1, 2)
    elseif face == 3
        return boundary_face_name(mesh.intervalmesh2, x2, 2)
    else
        return boundary_face_name(mesh.intervalmesh1, x1, 1)
    end
end

function opposing_face(
    mesh::RectilinearMesh,
    elem::CartesianIndex{2},
    face::Int,
)
    @assert 1 <= face <= 4
    x1, x2 = elem.I
    n1, n2 = size(elements(mesh))
    if face == 1
        return CartesianIndex(x1, mod1(x2 - 1, n2)), 3, true
    elseif face == 2
        return CartesianIndex(mod1(x1 + 1, n1), x2), 4, true
    elseif face == 3
        return CartesianIndex(x1, mod1(x2 + 1, n2)), 1, true
    else
        return CartesianIndex(mod1(x1 - 1, n1), x2), 2, true
    end
end


function coordinates(mesh::RectilinearMesh, elem::CartesianIndex{2}, vert::Int)
    x1, x2 = elem.I
    coord1 = coordinates(mesh.intervalmesh1, x1, vert == 1 || vert == 4 ? 1 : 2)
    coord2 = coordinates(mesh.intervalmesh2, x2, vert == 1 || vert == 2 ? 1 : 2)
    return Geometry.product_coordinates(coord1, coord2)
end
function coordinates(
    mesh::RectilinearMesh,
    elem::CartesianIndex{2},
    (ξ1, ξ2)::NTuple{2},
)
    x1, x2 = elem.I
    coord1 = coordinates(mesh.intervalmesh1, x1, (ξ1,))
    coord2 = coordinates(mesh.intervalmesh2, x2, (ξ2,))
    return Geometry.product_coordinates(coord1, coord2)
end

function containing_element(mesh::RectilinearMesh, coord)
    x1, (ξ1,) =
        containing_element(mesh.intervalmesh1, Geometry.coordinate(coord, 1))
    x2, (ξ2,) =
        containing_element(mesh.intervalmesh2, Geometry.coordinate(coord, 2))
    return CartesianIndex(x1, x2), (ξ1, ξ2)
end
