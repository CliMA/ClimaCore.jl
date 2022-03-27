# Constants required by "staggered_nonhydrostatic_model.jl"
# const FT = ? # specified in each test file
const p_0 = FT(1.0e5)
const R_d = FT(287.0)
const κ = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.80616)
const Ω = FT(0.0)
const f = FT(0.0)
const flux_form = true
include("../staggered_nonhydrostatic_model.jl")

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
temp(x, y, z) = T_init + lapse_rate * z + rand(FT) * FT(0.1) * (z < 5000)
pres(z) = p_0 * (1 + lapse_rate / T_init * z)^(-grav / R_d / lapse_rate)
θ(x, y, z) = temp(x, y, z) * (p_0 / pres(z))^κ
u(z) = 0.0
v(z) = 0.0

function center_initial_condition(
    local_geometry,
    ᶜ𝔼_name
)
  if flux_form 
      (; x, y, z) = local_geometry.coordinates
      ρ = pres(z) / R_d / temp(x, y, z)
      #ρuₕ = @. ρ * Geometry.Covariant12Vector(Geometry.UVVector(u(z), v(z)), local_geometry)
      ρuₕ = @. ρ * Geometry.UVVector(u(z), v(z))
      if ᶜ𝔼_name === Val(:ρθ)
          ρθ = ρ * θ(x, y, z)
          return (; ρ, ρθ, ρuₕ)
     # elseif ᶜ𝔼_name === Val(:ρe)
     #     ρe =
     #         ρ *
     #         (cv_d * (temp(x, y, z) - T_tri) + norm_sqr(uₕ) / 2 + grav * z)
     #     return (; ρ, ρe, ρuₕ)
     # elseif ᶜ𝔼_name === Val(:ρe_int)
     #     ρe_int = ρ * cv_d * (temp(x, y, z) - T_tri)
     #     return (; ρ, ρe_int, ρuₕ)
      end
  else
      (; x, y, z) = local_geometry.coordinates
      ρ = pres(z) / R_d / temp(x, y, z)
      uₕ = Geometry.Covariant12Vector(Geometry.UVVector(u(z), v(z)), local_geometry)
      if ᶜ𝔼_name === Val(:ρθ)
          ρθ = ρ * θ(x, y, z)
          return (; ρ, ρθ, uₕ)
      elseif ᶜ𝔼_name === Val(:ρe)
          ρe =
              ρ *
              (cv_d * (temp(x, y, z) - T_tri) + norm_sqr(uₕ) / 2 + grav * z)
          return (; ρ, ρe, uₕ)
      elseif ᶜ𝔼_name === Val(:ρe_int)
          ρe_int = ρ * cv_d * (temp(x, y, z) - T_tri)
          return (; ρ, ρe_int, uₕ)
      end
    end
end

if flux_form 
    face_initial_condition(local_geometry) =
        (; ρw = Geometry.WVector(FT(0)))
else
    face_initial_condition(local_geometry) =
        (; w = Geometry.Covariant3Vector(FT(0)))
end

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
    if flux_form 
        @. Yₜ.c.ρuₕ -= ᶜβ * Y.c.ρuₕ
        @. Yₜ.f.ρw -= ᶠβ * Y.f.ρw
    else
        @. Yₜ.c.uₕ -= ᶜβ * Y.c.uₕ
        @. Yₜ.f.w -= ᶠβ * Y.f.w
    end
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
    if flux_form 
        @. Yₜ.c.ρuₕ -= Y.c.ρ * (k_f * ᶜheight_factor) * Y.c.uₕ
    else
        @. Yₜ.c.uₕ -= (k_f * ᶜheight_factor) * Y.c.uₕ
    end
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= ᶜΔρT * (p_0 / ᶜp)^κ
    elseif :ρe in propertynames(Y.c)
        @. Yₜ.c.ρe -= ᶜΔρT * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int -= ᶜΔρT * cv_d
    end
end
