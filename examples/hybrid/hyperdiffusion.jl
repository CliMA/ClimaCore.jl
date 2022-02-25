hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(1e17)) = (;
    ᶜχ = similar(ᶜlocal_geometry, FT),
    ᶜχuₕ = similar(ᶜlocal_geometry, Geometry.Covariant12Vector{FT}),
    κ₄,
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ, κ₄) = p # assume that ᶜp has been updated
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    if :ρθ in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ(Y.c.ρθ / ᶜρ)) # ᶜχθ
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρθ -= κ₄ * W∇◦ₕ(ᶜρ * ∇ₕ(ᶜχ))
    elseif :ρe in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ((Y.c.ρe + ᶜp) / ᶜρ)) # ᶜχe
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρe -= κ₄ * W∇◦ₕ(ᶜρ * ∇ₕ(ᶜχ))
    elseif :ρe_int in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ((Y.c.ρe_int + ᶜp) / ᶜρ)) # ᶜχe_int
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρe_int -= κ₄ * W∇◦ₕ(ᶜρ * ∇ₕ(ᶜχ))
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            W∇ₕ(∇◦ₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                W∇⨉ₕ(Geometry.Covariant3Vector(∇⨉ₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss!(ᶜχuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                W∇ₕ(∇◦ₕ(ᶜχuₕ)) - Geometry.Covariant12Vector(
                    W∇⨉ₕ(Geometry.Covariant3Vector(∇⨉ₕ(ᶜχuₕ))),
                )
            )
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(W∇ₕ(∇◦ₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ)
        @. Yₜ.c.uₕ -= κ₄ * Geometry.Covariant12Vector(W∇ₕ(∇◦ₕ(ᶜχuₕ)))
    end
end

function tempest_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜp, ᶜχ, ᶜχuₕ, κ₄) = p # assume that ᶜp has been updated
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    @. ᶜχ = W∇◦ₕ(∇ₕ(ᶜρ))
    Spaces.weighted_dss!(ᶜχ)
    @. Yₜ.c.ρ -= κ₄ * W∇◦ₕ(∇ₕ(ᶜχ))

    if :ρθ in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ(Y.c.ρθ)) # ᶜχρθ
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρθ -= κ₄ * W∇◦ₕ(∇ₕ(ᶜχ))
    elseif :ρe in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ(Y.c.ρe + ᶜp)) # ᶜχρe
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρe -= κ₄ * W∇◦ₕ(∇ₕ(ᶜχ))
    elseif :ρe_int in propertynames(Y.c)
        @. ᶜχ = W∇◦ₕ(∇ₕ(Y.c.ρe_int + ᶜp)) # ᶜχρe_int
        Spaces.weighted_dss!(ᶜχ)
        @. Yₜ.c.ρe_int -= κ₄ * W∇◦ₕ(∇ₕ(ᶜχ))
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            W∇ₕ(∇◦ₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                W∇⨉ₕ(Geometry.Covariant3Vector(∇⨉ₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss!(ᶜχuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                W∇ₕ(∇◦ₕ(ᶜχuₕ)) - Geometry.Covariant12Vector(
                    W∇⨉ₕ(Geometry.Covariant3Vector(∇⨉ₕ(ᶜχuₕ))),
                )
            )
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(W∇ₕ(∇◦ₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ)
        @. Yₜ.c.uₕ -= κ₄ * Geometry.Covariant12Vector(W∇ₕ(∇◦ₕ(ᶜχuₕ)))
    end

    # treat w as a scalar
    @. ᶜχ = W∇◦ₕ(∇ₕ(ᶠw.components.data.:1))
    Spaces.weighted_dss!(ᶜχ)
    @. Yₜ.f.w.components.data.:1 -= κ₄ * W∇◦ₕ(∇ₕ(ᶜχ))
end
