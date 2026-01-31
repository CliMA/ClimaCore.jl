import LinearAlgebra: issymmetric

isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, ∂x∂ξT, ∂ξ∂xT, gⁱʲT, gᵢⱼT}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "inverse Jacobian"
    invJ::FT
    "Partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ"
    ∂x∂ξ::∂x∂ξT #::Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S}
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::∂ξ∂xT #::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, LocalAxis{I}}, S}
    "Contravariant metric tensor (inverse of gᵢⱼ), transforms covariant to contravariant vector components"
    gⁱʲ::gⁱʲT #::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, ContravariantAxis{I}}, S}
    "Covariant metric tensor (gᵢⱼ), transforms contravariant to covariant vector components"
    gᵢⱼ::gᵢⱼT #::Axis2Tensor{FT, Tuple{CovariantAxis{I}, CovariantAxis{I}}, S}
end

const FullLocalGeometry{I, C, FT, S} = LocalGeometry{
    I,
    C,
    FT,
    Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S},
    Axis2Tensor{FT, Tuple{ContravariantAxis{I}, LocalAxis{I}}, S},
    Axis2Tensor{FT, Tuple{ContravariantAxis{I}, ContravariantAxis{I}}, S},
    Axis2Tensor{FT, Tuple{CovariantAxis{I}, CovariantAxis{I}}, S},
}

@inline function LocalGeometry(coordinates, J, WJ,
    ∂x∂ξ::Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S},
) where {FT, I, S}
    ∂ξ∂x = inv(∂x∂ξ)
    C = typeof(coordinates)
    Jinv = inv(J)
    gⁱʲ = ∂ξ∂x * ∂ξ∂x'
    gᵢⱼ = ∂x∂ξ' * ∂x∂ξ
    isapproxsymmetric(components(gⁱʲ)) || error("gⁱʲ is not symmetric.")
    isapproxsymmetric(components(gᵢⱼ)) || error("gᵢⱼ is not symmetric.")
    return FullLocalGeometry{I, C, FT, S}(coordinates, J, WJ, Jinv, ∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)
end


"""
    SurfaceGeometry

The necessary local metric information defined at each node on each surface.
"""
struct SurfaceGeometry{FT, N}
    "surface Jacobian determinant, multiplied by the surface quadrature weight"
    sWJ::FT
    "surface outward pointing normal vector"
    normal::N
end

"""
    CoordinateOnlyGeometry

The necessary coordinates information defined at each node.

This is currently used for constructing spaces with pressure as the vertical
coordinate.
"""
struct CoordinateOnlyGeometry{C <: AbstractPoint}
    "Coordinates of the current point"
    coordinates::C
end

undertype(::Type{<:LocalGeometry{I, C, FT}}) where {I, C, FT} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
undertype(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = eltype(C)

"""
    blockmat(m11, m22[, m12])

Construct an `Axis2Tensor` from sub-blocks
"""
function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UAxis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Nothing = nothing,
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.UWAxis(), Geometry.Covariant13Axis()),
        @SMatrix [
            A[1, 1] zero(FT)
            zero(FT) B[1, 1]
        ]
    )
end
function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UAxis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    C = Geometry.components(c)
    Geometry.AxisTensor(
        (Geometry.UWAxis(), Geometry.Covariant13Axis()),
        @SMatrix [
            A[1, 1] zero(FT)
            C[1, 1] B[1, 1]
        ]
    )
end

function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.VAxis, Geometry.Covariant2Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Nothing = nothing,
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.VWAxis(), Geometry.Covariant23Axis()),
        @SMatrix [
            A[1, 1] zero(FT)
            zero(FT) B[1, 1]
        ]
    )
end
function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.VAxis, Geometry.Covariant2Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant2Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    C = Geometry.components(c)
    Geometry.AxisTensor(
        (Geometry.VWAxis(), Geometry.Covariant23Axis()),
        @SMatrix [
            A[1, 1] zero(FT)
            C[1, 1] B[1, 1]
        ]
    )
end
function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UVAxis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Nothing = nothing,
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.UVWAxis(), Geometry.Covariant123Axis()),
        @SMatrix [
            A[1, 1] A[1, 2] zero(FT)
            A[2, 1] A[2, 2] zero(FT)
            zero(FT) zero(FT) B[1, 1]
        ]
    )
end
function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UVAxis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
    c::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant12Axis},
        SMatrix{1, 2, FT, 2},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    C = Geometry.components(c)
    Geometry.AxisTensor(
        (Geometry.UVWAxis(), Geometry.Covariant123Axis()),
        @SMatrix [
            A[1, 1] A[1, 2] zero(FT)
            A[2, 1] A[2, 2] zero(FT)
            C[1, 1] C[1, 2] B[1, 1]
        ]
    )
end
