
# set initial condition for steady-state test
function set_initial_condition(space)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        h = 1.0
        return h
    end
    return Y
end

# set simple field
function set_simple_field(space)
    α0 = 45.0
    h0 = 1.0
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long
        z = coord.z
        h = h0 * z * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        return h
    end
    return Y
end

function set_elevation(space, h₀)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long
        FT = eltype(λ)
        ϕₘ = FT(0) # degrees (equator)
        λₘ = FT(3 / 2 * 180)  # degrees
        rₘ =
            FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
        Rₘ = FT(3π / 4) # Moutain radius
        ζₘ = FT(π / 16) # Mountain oscillation half-width
        zₛ = ifelse(
            rₘ < Rₘ,
            FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
            FT(0),
        )
        return zₛ
    end
    return Y
end
