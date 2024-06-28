import LinearAlgebra: issymmetric

exactly_symmetric(A::AbstractMatrix{T}) where {T} = A == A'

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, S}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "inverse Jacobian"
    invJ::FT
    "Partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ"
    ∂x∂ξ::Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S}
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, LocalAxis{I}}, S}
    "Contravariant metric tensor (inverse of gᵢⱼ), transforms covariant to contravariant vector components"
    gⁱʲ::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, ContravariantAxis{I}}, S}
    "Covariant metric tensor (gᵢⱼ), transforms contravariant to covariant vector components"
    gᵢⱼ::Axis2Tensor{FT, Tuple{CovariantAxis{I}, CovariantAxis{I}}, S}
    @inline function LocalGeometry(
        coordinates,
        J,
        WJ,
        ∂x∂ξ::Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S},
    ) where {FT, I, S}
        ∂ξ∂x = inv(∂x∂ξ)
        C = typeof(coordinates)
        Jinv = inv(J)
        gⁱʲ = ∂ξ∂x * ∂ξ∂x'
        gᵢⱼ = ∂x∂ξ' * ∂x∂ξ
        if !exactly_symmetric(components(gⁱʲ))
            @show coordinates
            @show J
            @show WJ
            @show ∂x∂ξ
            error("gⁱʲ is not symmetric.")
        end
        if !exactly_symmetric(components(gᵢⱼ))
            @show coordinates
            @show J
            @show WJ
            @show ∂x∂ξ
            error("gᵢⱼ is not symmetric.")
        end
        return new{I, C, FT, S}(coordinates, J, WJ, Jinv, ∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)
    end
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

undertype(::Type{LocalGeometry{I, C, FT, S}}) where {I, C, FT, S} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT

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
