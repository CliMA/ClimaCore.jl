@inline function ref_transform(
    ::Basis{Tto, I},
    x::Tensor{1, T, Tuple{Basis{Tfrom, I}}, SVector{N, T}},
) where {Tto <: BasisType, Tfrom <: BasisType, I, T, N}
    x
end

@inline function ref_project(
    ::Basis{Tto, I},
    x::Tensor{1, T, Tuple{Basis{Tfrom, I}}, SVector{N, T}},
) where {Tto <: BasisType, Tfrom <: BasisType, I, T, N}
    x
end

@generated function ref_transform(
    ato::Basis{Tto, Ito},
    x::Tensor{1, T, Tuple{Basis{Tfrom, Ifrom}}, SVector{N, T}},
) where {Tto <: BasisType, Ito, Tfrom <: BasisType, Ifrom, T, N}
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            errcond = :($errcond || x[$n] != zero(T))
        end
    end
    vals = []
    for i in Ito
        val = :(zero(T))
        for n in 1:N
            if i == Ifrom[n]
                val = :(x[$n])
                break
            end
        end
        push!(vals, val)
    end
    Nto = length(Ito)
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, typeof(ato), x))
        end
        @inbounds Tensor(SVector{$Nto, T}($(vals...)), (ato,))
    end
end

@generated function ref_project(
    ato::Basis{Tto, Ito},
    x::Tensor{1, T, Tuple{Basis{Tfrom, Ifrom}}, SVector{N, T}},
) where {Tto <: BasisType, Ito, Tfrom <: BasisType, Ifrom, T, N}
    vals = []
    for i in Ito
        val = :(zero(T))
        for n in 1:N
            if i == Ifrom[n]
                val = :(x[$n])
                break
            end
        end
        push!(vals, val)
    end
    Nto = length(Ito)
    return :(@inbounds Tensor(SVector{$Nto, T}($(vals...)), (ato,)))
end

@inline function ref_transform(
    ::Basis{Tto, I},
    x::Tensor{2, T, Tuple{Basis{Tfrom, I}, Basis{T2, J}}},
) where {Tto <: BasisType, Tfrom <: BasisType, T2 <: BasisType, I, J, T}
    x
end

@inline function ref_project(
    ::Basis{Tto, I},
    x::Tensor{2, T, Tuple{Basis{Tfrom, I}, Basis{T2, J}}},
) where {Tto <: BasisType, Tfrom <: BasisType, T2 <: BasisType, I, J, T}
    x
end

@generated function ref_transform(
    ato::Basis{Tto, Ito},
    x::Tensor{2, T, Tuple{Basis{Tfrom, Ifrom}, Basis{T2, J}}},
) where {Tto <: BasisType, Ito, Tfrom <: BasisType, Ifrom, T2 <: BasisType, J, T}
    N = length(Ifrom)
    M = length(J)
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            for m in 1:M
                errcond = :($errcond || x[$n, $m] != zero(T))
            end
        end
    end
    vals = []
    for m in 1:M
        for i in Ito
            val = :(zero(T))
            for n in 1:N
                if i == Ifrom[n]
                    val = :(x[$n, $m])
                    break
                end
            end
            push!(vals, val)
        end
    end
    Nto = length(Ito)
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, typeof(ato), x))
        end
        @inbounds Tensor(SMatrix{$Nto, $M, T}($(vals...)), (ato, axes(x, 2)))
    end
end

@generated function ref_project(
    ato::Basis{Tto, Ito},
    x::Tensor{2, T, Tuple{Basis{Tfrom, Ifrom}, Basis{T2, J}}},
) where {Tto <: BasisType, Ito, Tfrom <: BasisType, Ifrom, T2 <: BasisType, J, T}
    N = length(Ifrom)
    M = length(J)
    vals = []
    for m in 1:M
        for i in Ito
            val = :(zero(T))
            for n in 1:N
                if i == Ifrom[n]
                    val = :(x[$n, $m])
                    break
                end
            end
            push!(vals, val)
        end
    end
    Nto = length(Ito)
    return :(@inbounds Tensor(SMatrix{$Nto, $M, T}($(vals...)), (ato, axes(x, 2))))
end


for op in (:ref_transform, :ref_project)
    @eval begin
        # Orthonormal <-> Covariant
        @inline $op(
            ax::Basis{Orthonormal},
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::Basis{Covariant},
            v::OrthonormalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )

        # Contravariant <-> Orthonormal
        @inline $op(
            ax::Basis{Contravariant},
            v::OrthonormalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )
        @inline $op(
            ax::Basis{Orthonormal},
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        # Covariant <-> Contravariant
        @inline $op(
            ax::Basis{Contravariant},
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            local_geometry.∂ξ∂x' *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::Basis{Covariant},
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            local_geometry.∂x∂ξ *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        @inline $op(ato::Basis{Covariant}, v::CovariantTensor, ::LocalGeometry) =
            $op(ato, v)
        @inline $op(
            ato::Basis{Contravariant},
            v::ContravariantTensor,
            ::LocalGeometry,
        ) = $op(ato, v)
        @inline $op(ato::Basis{Orthonormal}, v::OrthonormalTensor, ::LocalGeometry) =
            $op(ato, v)
    end
end

@inline ref_contravariant1(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1]
@inline ref_contravariant2(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1]
@inline ref_contravariant3(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1]

@inline ref_contravariant1(u::AbstractTensor{2}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1, :]
@inline ref_contravariant2(u::AbstractTensor{2}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1, :]
@inline ref_contravariant3(u::AbstractTensor{2}, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1, :]

@inline ref_covariant1(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₁
@inline ref_covariant2(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₂
@inline ref_covariant3(u::AbstractTensor{1}, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₃

Base.@propagate_inbounds ref_Jcontravariant3(
    u::AbstractTensor,
    local_geometry::LocalGeometry,
) = local_geometry.J * ref_contravariant3(u, local_geometry)
