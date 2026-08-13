# Real vector spherical harmonics, and the inter-element jump of a pre-DSS
# field, shared by `unit_sphere_vsh_divergence.jl` and
# `conv_sphere_divergence_jump.jl`.
#
# With Y = λ_lm(sin φ) cos(m λ) — a real spherical harmonic; λ_lm is the
# spherical-harmonic-normalized associated Legendre function — the tangent
# fields on a sphere of radius a
#
#     S = ∇Y        (spheroidal),    div S = -l(l+1)/a² Y
#     T = r̂ × ∇Y    (toroidal),      div T = 0
#
# give the divergence a pointwise-exact answer at every wavenumber l.

using AssociatedLegendrePolynomials: λlm
import ClimaCore: Fields, Geometry

Ylm(l, m, lat, long) = λlm(l, m, sind(lat)) * cosd(m * long)

# dλ_lm(sin φ)/dφ (φ in radians, `lat` in degrees), from the recurrence
#     (1 - x²) dλ_lm/dx = -l x λ_lm(x) + a_lm λ_(l-1)m(x),
#     a_lm = √((2l + 1)(l² - m²) / (2l - 1)),
# with x = sin φ, dx = cos φ dφ and 1 - x² = cos²φ.
function dλlm_dlat(l, m, lat)
    x = sind(lat)
    lower =
        l == m ? zero(x) :
        sqrt((2l + 1) * (l^2 - m^2) / (2l - 1)) * λlm(l - 1, m, x)
    return (-(l * x * λlm(l, m, x)) + lower) / cosd(lat)
end

# Eastward and northward components of S and T. The (u, v) representation of a
# VSH is singular at the poles; keep nodes off them (odd Ne, even Nq).
function spheroidal_uv(l, m, lat, long, radius)
    uE = -m * λlm(l, m, sind(lat)) * sind(m * long) / (radius * cosd(lat))
    uN = dλlm_dlat(l, m, lat) * cosd(m * long) / radius
    return (uE, uN)
end

function toroidal_uv(l, m, lat, long, radius)
    (uE, uN) = spheroidal_uv(l, m, lat, long, radius)
    return (-uN, uE)
end

# VSH field on `space` with components given by `uv`, as local (E, N) vectors
# (the form the DG interface flux takes) or as a contravariant field (the form
# the CG divergence operators take).
function vsh_uv_field(uv::UV, l, m, space, radius) where {UV}
    return map(Fields.coordinate_field(space)) do coord
        (uE, uN) = uv(l, m, coord.lat, coord.long, radius)
        Geometry.UVVector(oftype(coord.lat, uE), oftype(coord.lat, uN))
    end
end

vsh_field(uv::UV, l, m, space, radius) where {UV} = Geometry.transform.(
    Ref(Geometry.Contravariant12Axis()),
    vsh_uv_field(uv, l, m, space, radius),
)

ylm_field(l, m, space) = map(
    coord -> oftype(coord.lat, Ylm(l, m, coord.lat, coord.long)),
    Fields.coordinate_field(space),
)

# ∇Y_(lψ mψ) · u as a scalar field, with u the (l, m) VSH given by `uv`; both
# factors are analytic in the (E, N) orthonormal basis. With ψ div(u) added,
# this is the exact div(ψ u) that the split divergence approximates.
function vsh_grad_dot_field(uv::UV, l, m, lψ, mψ, space, radius) where {UV}
    return map(Fields.coordinate_field(space)) do coord
        (uE, uN) = uv(l, m, coord.lat, coord.long, radius)
        (gE, gN) = spheroidal_uv(lψ, mψ, coord.lat, coord.long, radius)
        oftype(coord.lat, gE * uE + gN * uN)
    end
end

# Coincident element-boundary nodes, grouped by rounded position on the unit
# sphere ((lat, long) is not a usable key: longitude wraps at a panel seam).
# The same grouping as `test/Spaces/unit_dss_reference.jl`.
function unit_sphere_positions(space)
    coords = Fields.coordinate_field(space)
    return map(vec(parent(coords.lat)), vec(parent(coords.long))) do lat, long
        (φ, λ) = (Float64(lat), Float64(long))
        round.((cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ)), sigdigits = 8)
    end
end

# Largest max - min spread of `field` over any set of coincident nodes: the
# inter-element discontinuity that `weighted_dss!` averages away.
function max_interelement_jump(field, positions)
    vals = vec(parent(field))
    lo = Dict{eltype(positions), eltype(vals)}()
    hi = Dict{eltype(positions), eltype(vals)}()
    for (p, v) in zip(positions, vals)
        lo[p] = min(v, get(lo, p, v))
        hi[p] = max(v, get(hi, p, v))
    end
    return maximum(p -> hi[p] - lo[p], keys(hi))
end
