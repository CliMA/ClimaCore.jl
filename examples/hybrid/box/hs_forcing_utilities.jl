# Constants required by "staggered_nonhydrostatic_model.jl"
# const FT = ? # specified in each test file
const p_0 = FT(1.0e5)
const R_d = FT(287.0)
const κ = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.80616)
const Ω = FT(0.0)
const f = FT(0.0)
include("../staggered_nonhydrostatic_model.jl")

# Constants required for balanced flow and baroclinic wave initial conditions
const R = FT(6.371229e6)
const k = 3
const T_e = FT(310) # temperature at the equator
const T_p = FT(240) # temperature at the pole
const T_0 = FT(0.5) * (T_e + T_p)
const Γ = FT(0.005)
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = FT(15e3)
const λ_c = FT(20)
const ϕ_c = FT(40)
const d_0 = R / 6
const V_p = FT(1)

# Constants required for Rayleigh sponge layer
const z_D = FT(15e3)

# Constants required for Held-Suarez forcing
const day = FT(3600 * 24)
const k_a = 1 / (40 * day)
const k_f = 1 / day
const k_s = 1 / (4 * day)
const ΔT_y = FT(0)
const Δθ_z = FT(-5)
const T_equator = FT(315)
const T_min = FT(200)
const σ_b = FT(7 / 10)

##
## Initial conditions
##

const T_init = 315
const scale_height = R_d * T_init / grav
const lapse_rate = FT(-0.008)
temp(z) = T_init + lapse_rate * z + rand(FT) * FT(0.1) * (z < 5000)
pres(z) = p_0 * (1 + lapse_rate / T_init * z)^(-grav / R_d / lapse_rate)
θ(z) = temp(z) * (p_0 / pres(z))^κ
u(z) = 0.0
v(z) = 0.0

function center_initial_condition(
    local_geometry,
    ᶜ𝔼_name
)
    (; x, y, z) = local_geometry.coordinates
    ρ = pres(z) / R_d / temp(z)
    uₕ = Geometry.Covariant12Vector(Geometry.UVVector(u(z), v(z)), local_geometry)
    if ᶜ𝔼_name === Val(:ρθ)
        ρθ = ρ * θ(z)
        return (; ρ, ρθ, uₕ)
    elseif ᶜ𝔼_name === Val(:ρe)
        ρe =
            ρ *
            (cv_d * (temp(z) - T_tri) + norm_sqr(uₕ) / 2 + grav * z)
        return (; ρ, ρe, uₕ)
    elseif ᶜ𝔼_name === Val(:ρe_int)
        ρe_int = ρ * cv_d * (temp(z) - T_tri)
        return (; ρ, ρe_int, uₕ)
    end
end
face_initial_condition(local_geometry) =
    (; w = Geometry.Covariant3Vector(FT(0)))

##
## Additional tendencies
##

function rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt)
    ᶜz = ᶜlocal_geometry.coordinates.z
    ᶠz = ᶠlocal_geometry.coordinates.z
    ᶜαₘ = @. ifelse(ᶜz > z_D, 1 / (20 * dt), FT(0))
    ᶠαₘ = @. ifelse(ᶠz > z_D, 1 / (20 * dt), FT(0))
    zmax = maximum(ᶠz)
    ᶜβ = @. ᶜαₘ * sin(π / 2 * (ᶜz - z_D) / (zmax - z_D))^2
    ᶠβ = @. ᶠαₘ * sin(π / 2 * (ᶠz - z_D) / (zmax - z_D))^2
    return (; ᶜβ, ᶠβ)
end

function rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    (; ᶜβ, ᶠβ) = p
    @. Yₜ.c.uₕ -= ᶜβ * Y.c.uₕ
    @. Yₜ.f.w -= ᶠβ * Y.f.w
end

held_suarez_cache(ᶜlocal_geometry) = (;
    ᶜσ = similar(ᶜlocal_geometry, FT),
    ᶜheight_factor = similar(ᶜlocal_geometry, FT),
    ᶜΔρT = similar(ᶜlocal_geometry, FT),
    ᶜφ = deg2rad.(ᶜlocal_geometry.coordinates.y),
)

function held_suarez_tendency!(Yₜ, Y, p, t)
    (; ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ) = p # assume that ᶜp has been updated

    @. ᶜσ = ᶜp / p_0
    @. ᶜheight_factor = max(0, (ᶜσ - σ_b) / (1 - σ_b))
    @. ᶜΔρT =
        (k_a + (k_s - k_a) * ᶜheight_factor) *
        Y.c.ρ *
        ( # ᶜT - ᶜT_equil
            ᶜp / (Y.c.ρ * R_d) - max(
                T_min,
                (T_equator - ΔT_y - Δθ_z * log(ᶜσ)) *
                ᶜσ^(R_d / cp_d),
            )
        )

    @. Yₜ.c.uₕ -= (k_f * ᶜheight_factor) * Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= ᶜΔρT * (p_0 / ᶜp)^κ
    elseif :ρe in propertynames(Y.c)
        @. Yₜ.c.ρe -= ᶜΔρT * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int -= ᶜΔρT * cv_d
    end
end
