@inline function ref_transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end

@inline function ref_project(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end

@generated function ref_transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
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
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        @inbounds AxisVector(ato, SVector($(vals...)))
    end
end

@generated function ref_project(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
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
    return :(@inbounds AxisVector(ato, SVector($(vals...))))
end

@inline function ref_transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@inline function ref_project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@generated function ref_transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
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
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        @inbounds Axis2Tensor(
            (ato, axes(x, 2)),
            SMatrix{$(length(Ito)), $M}($(vals...)),
        )
    end
end

@generated function ref_project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
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
    return :(@inbounds Axis2Tensor(
        (ato, axes(x, 2)),
        SMatrix{$(length(Ito)), $M}($(vals...)),
    ))
end


for op in (:ref_transform, :ref_project)
    @eval begin
        # Covariant <-> Cartesian
        @inline $op(
            ax::CartesianAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )
        @inline $op(
            ax::LocalAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::LocalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )

        # Contravariant <-> Cartesian
        @inline $op(
            ax::ContravariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )
        @inline $op(
            ax::CartesianAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )
        @inline $op(
            ax::ContravariantAxis,
            v::LocalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )

        @inline $op(
            ax::LocalAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        # Covariant <-> Contravariant
        @inline $op(
            ax::ContravariantAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            local_geometry.∂ξ∂x' *
            $op(Geometry.dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            local_geometry.∂x∂ξ *
            $op(Geometry.dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        @inline $op(ato::CovariantAxis, v::CovariantTensor, ::LocalGeometry) =
            $op(ato, v)
        @inline $op(
            ato::ContravariantAxis,
            v::ContravariantTensor,
            ::LocalGeometry,
        ) = $op(ato, v)
        @inline $op(ato::CartesianAxis, v::CartesianTensor, ::LocalGeometry) =
            $op(ato, v)
        @inline $op(ato::LocalAxis, v::LocalTensor, ::LocalGeometry) =
            $op(ato, v)
    end
end

@inline ref_contravariant1(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1]
@inline ref_contravariant2(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1]
@inline ref_contravariant3(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1]

@inline ref_contravariant1(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1, :]
@inline ref_contravariant2(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1, :]
@inline ref_contravariant3(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1, :]

@inline ref_covariant1(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₁
@inline ref_covariant2(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₂
@inline ref_covariant3(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₃

Base.@propagate_inbounds ref_Jcontravariant3(
    u::AxisTensor,
    local_geometry::LocalGeometry,
) = local_geometry.J * ref_contravariant3(u, local_geometry)
