abstract type VectorValue{FT} end


struct CartesianVector{FT} <: VectorValue{FT}
    u1::FT
    u2::FT
    u3::FT
end



"""
    SphericalCartesianVector

Representation of a vector in spherical cartesian coordinates.
"""
struct SphericalCartesianVector{FT <: Number}
    "zonal (eastward) component"
    u::FT
    "meridional (northward) component"
    v::FT
    "radial (upward) component"
    w::FT
end


function spherical_cartesian_basis(geom::LocalGeometry)
    x = geom.x

    r12 = hypot(x[2], x[1])
    r = hypot(x[3], r12)

    (
        û = SVector(-x[2] / r12, x[1] / r12, 0),
        v̂ = SVector(
            -(x[3] / r) * (x[1] / r12),
            -(x[3] / r) * (x[2] / r12),
            r12 / r,
        ),
        ŵ = x ./ r,
    )
end

function SphericalCartesianVector(v::CartesianVector, geom::LocalGeometry)
    b = spherical_cartesian_basis(geom)
    SphericalCartesianVector(dot(b.û, v), dot(b.v̂, v), dot(b.ŵ, v))
end

function CartesianVector(s::SphericalCartesianVector, geom::LocalGeometry)
    b = spherical_cartesian_basis(geom)
    return s.u .* b.û .+ s.v .* b.v̂ .+ s.w .* b.ŵ
end

struct Contravariant{FT}
    components::SVector{3, FT}
end
function CartesianVector(vⁱ::Contravariant, geom::LocalGeometry)
    CartesianVector((geom.∂ξ∂x \ vⁱ.components)...)
end
function Contravariant(v::CartesianVector, geom::LocalGeometry)
    Contravariant((geom.∂ξ∂x * v.components)...)
end
