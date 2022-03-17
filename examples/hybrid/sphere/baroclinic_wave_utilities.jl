using LinearAlgebra: ×, norm

# General parameters
const k = 3
const T_e = 310 # temperature at the equator
const T_p = 240 # temperature at the pole
const T_0 = 0.5 * (T_e + T_p)
const Γ = 0.005
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = 0.5 * (k + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = 15.0e3
const λ_c = 20.0
const ϕ_c = 40.0
const d_0 = R / 6
const V_p = 1.0

# Held-Suarez parameters
const day = FT(3600 * 24)
const k_a = FT(1 / (40 * day))
const k_f = FT(1 / day)
const k_s = FT(1 / (4 * day))
const ΔT_y = FT(60)
const Δθ_z = FT(10)
const T_equator = FT(315)
const T_min = FT(200)
const σ_b = FT(7 / 10)
const z_D = FT(15.0e3)

# Helper functions
τ_z_1(z) = exp(Γ * z / T_0)
τ_z_2(z) = 1 - 2 * (z / b / H)^2
τ_z_3(z) = exp(-(z / b / H)^2)
τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
τ_int_2(z) = C * z * τ_z_3(z)
F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
I_T(ϕ) = cosd(ϕ)^k - k / (k + 2) * (cosd(ϕ))^(k + 2)
temp(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
pres(ϕ, z) = p_0 * exp(-grav / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
θ(ϕ, z) = temp(ϕ, z) * (p_0 / pres(ϕ, z))^κ
r(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
U(ϕ, z) =
    grav * k / R * τ_int_2(z) * temp(ϕ, z) * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
u(ϕ, z) = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
v(ϕ, z) = 0.0
c3(λ, ϕ) = cos(π * r(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(π * r(λ, ϕ) / 2 / d_0)
cond(λ, ϕ) = (0 < r(λ, ϕ) < d_0) * (r(λ, ϕ) != R * pi)
δu(λ, ϕ, z) =
    -16 * V_p / 3 / sqrt(3) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
    sin(r(λ, ϕ) / R) * cond(λ, ϕ)
δv(λ, ϕ, z) =
    16 * V_p / 3 / sqrt(3) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    cosd(ϕ_c) *
    sind(λ - λ_c) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)

# Initial conditions
function initial_condition_uv(local_geometry; is_balanced_flow)
    @unpack lat, long, z = local_geometry.coordinates
    u₀ = u(lat, z)
    v₀ = v(lat, z)
    if !is_balanced_flow
        u₀ += δu(long, lat, z)
        v₀ += δv(long, lat, z)
    end
    return Geometry.UVVector(u₀, v₀)
end
function initial_condition_ρθ(local_geometry)
    @unpack lat, z = local_geometry.coordinates
    ρ = pres(lat, z) / R_d / temp(lat, z)
    ρθ = ρ * θ(lat, z)
    return (; ρ, ρθ)
end
function initial_condition_ρe(local_geometry; is_balanced_flow)
    @unpack lat, long, z = local_geometry.coordinates
    ρ = pres(lat, z) / R_d / temp(lat, z)
    uₕ = initial_condition_uv(local_geometry; is_balanced_flow)
    ρe = ρ * (cv_d * (temp(lat, z) - T_tri) + norm_sqr(uₕ) / 2 + grav * z)
    return (; ρ, ρe)
end
function initial_condition_velocity(local_geometry; is_balanced_flow)
    uₕ = initial_condition_uv(local_geometry; is_balanced_flow)
    return Geometry.Covariant12Vector(uₕ, local_geometry)
end

# Operators
const hdiv = Operators.Divergence()
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const hwgrad = Operators.WeakGradient()
const hcurl = Operators.Curl()
const hwcurl = Operators.WeakCurl()
const If2c = Operators.InterpolateF2C()
const Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const vdivf2c = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
)
const vcurlc2f = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
)
const vgradc2f = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
)
const fcc = Operators.FluxCorrectionC2C(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

# Parameters available in the tendency functions
function baroclinic_wave_cache_values(Y, dt)
    FT = eltype(Y)
    center_space = axes(Y.Yc.ρ)
    face_space = axes(Y.w)
    center_coordinates = Fields.local_geometry_field(center_space).coordinates
    cf = @. Geometry.Contravariant3Vector(
        Geometry.WVector(2 * Ω * sind(center_coordinates.lat)),
    )
    cχ_energy_named_tuple =
        :ρθ in propertynames(Y.Yc) ? (; cχθ = Fields.Field(FT, center_space)) :
        (; cχe = Fields.Field(FT, center_space))
    return (;
        cχ_energy_named_tuple...,
        cχρ = Fields.Field(FT, center_space),
        cχuₕ = Fields.Field(Geometry.Covariant12Vector{FT}, center_space),
        cuvw = Fields.Field(Geometry.Covariant123Vector{FT}, center_space),
        cω³ = Fields.Field(Geometry.Contravariant3Vector{FT}, center_space),
        fω¹² = Fields.Field(Geometry.Contravariant12Vector{FT}, face_space),
        fu¹² = Fields.Field(Geometry.Contravariant12Vector{FT}, face_space),
        fu³ = Fields.Field(Geometry.Contravariant3Vector{FT}, face_space),
        cf,
        cK = Fields.Field(FT, center_space),
        cp = Fields.Field(FT, center_space),
        cΦ = grav .* center_coordinates.z,
    )
end
function held_suarez_cache_values(Y, dt)
    FT = eltype(Y)
    center_space = axes(Y.Yc.ρ)
    center_coordinates = Fields.local_geometry_field(center_space).coordinates
    return (;
        cσ = Fields.Field(FT, center_space),
        c_height_factor = Fields.Field(FT, center_space),
        cΔρT = Fields.Field(FT, center_space),
        cφ = deg2rad.(center_coordinates.lat),
    )
end
function final_adjustments_cache_values(Y, dt; use_rayleigh_sponge)
    FT = eltype(Y)
    center_space = axes(Y.Yc.ρ)
    face_space = axes(Y.w)
    center_coordinates = Fields.local_geometry_field(center_space).coordinates
    if use_rayleigh_sponge
        cαₘ = @. ifelse(center_coordinates.z > z_D, 1 / (20 * dt), FT(0.0))
        zmax = maximum(Fields.local_geometry_field(face_space).coordinates.z)
        cβ = @. cαₘ * sin(π / 2 * (center_coordinates.z - z_D) / (zmax - z_D))^2
        return (; cβ)
    else
        return (;)
    end
end

##
## Tendency functions
##

function baroclinic_wave_ρθ_remaining_tendency!(dY, Y, p, t; κ₄)
    @unpack cχθ, cχuₕ, cuvw, cω³, fω¹², fu¹², fu³, cf, cK, cp, cΦ = p

    cρ = Y.Yc.ρ # density on centers
    cρθ = Y.Yc.ρθ # potential temperature on centers
    cuₕ = Y.uₕ # Covariant12Vector on centers
    fw = Y.w # Covariant3Vector on faces

    # Update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    # Hyperdiffusion

    @. cχθ = hwdiv(hgrad(cρθ / cρ))
    @. cχuₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(cχθ)
    Spaces.weighted_dss!(cχuₕ)

    @. dY.Yc.ρθ -= κ₄ * hwdiv(cρ * hgrad(cχθ))
    @. dY.uₕ -=
        κ₄ * (
            hwgrad(hdiv(cχuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(cχuₕ))),
            )
        )

    # Mass conservation

    @. cuvw =
        Geometry.Covariant123Vector(cuₕ) + Geometry.Covariant123Vector(If2c(fw))

    @. dY.Yc.ρ -= hdiv(cρ * cuvw)
    @. dY.Yc.ρ -= vdivf2c(Ic2f(cρ * cuₕ))

    # Momentum conservation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    @. cω³ = hcurl(cuₕ) # Contravariant3Vector
    @. fω¹² = hcurl(fw) # Contravariant12Vector
    @. fω¹² += vcurlc2f(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. fu¹² =
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(cuₕ)))
    @. fu³ = Geometry.Contravariant3Vector(Geometry.Covariant123Vector(fw))

    @. dY.uₕ -= If2c(fω¹² × fu³)
    # Needed for 3D:
    @. dY.uₕ -=
        (cf + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    @. cK = norm_sqr(cuvw) / 2
    @. cp = pressure(cρθ)

    @. dY.uₕ -= hgrad(cp) / cρ
    @. dY.uₕ -= hgrad(cK + cΦ)

    @. dY.w -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dY.w -= vgradc2f(cK)

    # Energy conservation

    @. dY.Yc.ρθ -= hdiv(cuvw * cρθ)
    @. dY.Yc.ρθ -= vdivf2c(Ic2f(cuₕ * cρθ))
end

function held_suarez_ρθ_tempest_remaining_tendency!(dY, Y, p, t; κ₄)
    @unpack cχρ, cχθ, cχuₕ, cuvw, cω³, fω¹², fu¹², fu³, cf, cK, cp, cΦ = p

    cρ = Y.Yc.ρ # density on centers
    cρθ = Y.Yc.ρθ # potential temperature on centers
    cuₕ = Y.uₕ # Covariant12Vector on centers
    fw = Y.w # Covariant3Vector on faces

    # Update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    # Hyperdiffusion: ρ, ρθ, uₕ
    @. cχθ = hwdiv(hgrad(cρθ))
    @. cχρ = hwdiv(hgrad(cρ))
    @. cχuₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )
    # treat w as a scalar for the purposes of applying Hyperdiffusion
    fws = fw.components.data.:1
    fχw = @. hwdiv(hgrad(fws))

    Spaces.weighted_dss!(cχθ)
    Spaces.weighted_dss!(cχuₕ)
    Spaces.weighted_dss!(cχρ)
    Spaces.weighted_dss!(fχw)

    @. dY.Yc.ρθ -= κ₄ * hwdiv(hgrad(cχθ))
    @. dY.Yc.ρ -= κ₄ * hwdiv(hgrad(cχρ))
    @. dY.uₕ -=
        κ₄ * (
            hwgrad(hdiv(cχuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(cχuₕ))),
            )
        )
    dfws = dY.w.components.data.:1
    @. dfws -= κ₄ * hwdiv(hgrad(fχw))

    # Mass conservation

    @. cuvw =
        Geometry.Covariant123Vector(cuₕ) + Geometry.Covariant123Vector(If2c(fw))

    @. dY.Yc.ρ -= hdiv(cρ * cuvw)
    @. dY.Yc.ρ -= vdivf2c(Ic2f(cρ * cuₕ))

    # Momentum conservation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    @. cω³ = hcurl(cuₕ) # Contravariant3Vector
    @. fω¹² = hcurl(fw) # Contravariant12Vector
    @. fω¹² += vcurlc2f(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. fu¹² =
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(cuₕ)))
    @. fu³ = Geometry.Contravariant3Vector(Geometry.Covariant123Vector(fw))

    @. dY.uₕ -= If2c(fω¹² × fu³)
    # Needed for 3D:
    @. dY.uₕ -=
        (cf + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    @. cK = norm_sqr(cuvw) / 2
    @. cp = pressure(cρθ)

    @. dY.uₕ -= hgrad(cp) / cρ
    @. dY.uₕ -= hgrad(cK + cΦ)

    @. dY.w -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dY.w -= vgradc2f(cK)

    # Energy conservation

    @. dY.Yc.ρθ -= hdiv(cuvw * cρθ)
    @. dY.Yc.ρθ -= vdivf2c(Ic2f(cuₕ * cρθ))
end

function baroclinic_wave_ρe_remaining_tendency!(dY, Y, p, t; κ₄)
    @unpack cχe, cχuₕ, cuvw, cω³, fω¹², fu¹², fu³, cf, cK, cp, cΦ = p

    cρ = Y.Yc.ρ # density on centers
    cρe = Y.Yc.ρe # total energy on centers
    cuₕ = Y.uₕ # Covariant12Vector on centers
    fw = Y.w # Covariant3Vector on faces

    @. cuvw =
        Geometry.Covariant123Vector(cuₕ) + Geometry.Covariant123Vector(If2c(fw))

    @. cK = norm_sqr(cuvw) / 2
    @. cp = pressure(cρ, cρe / cρ, cK, cΦ)

    # Update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    # Hyperdiffusion
    ce = @. cρe / cρ
    ch_tot = @. ce + cp / cρ

    # Total enthalpy

    @. cχe = hwdiv(hgrad(ch_tot))
    @. cχuₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(cχe)
    Spaces.weighted_dss!(cχuₕ)

    @. dY.Yc.ρe -= κ₄ * hwdiv(cρ * hgrad(cχe))
    @. dY.uₕ -=
        κ₄ * (
            hwgrad(hdiv(cχuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(cχuₕ))),
            )
        )

    # Mass conservation

    @. dY.Yc.ρ -= hdiv(cρ * cuvw)
    @. dY.Yc.ρ -= vdivf2c(Ic2f(cρ * cuₕ))

    # Momentum conservation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    @. cω³ = hcurl(cuₕ) # Contravariant3Vector
    @. fω¹² = hcurl(fw) # Contravariant12Vector
    @. fω¹² += vcurlc2f(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. fu¹² =
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(cuₕ)))
    @. fu³ = Geometry.Contravariant3Vector(Geometry.Covariant123Vector(fw))

    @. dY.uₕ -= If2c(fω¹² × fu³)
    # Needed for 3D:
    @. dY.uₕ -=
        (cf + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))


    @. dY.uₕ -= hgrad(cp) / cρ
    @. dY.uₕ -= hgrad(cK + cΦ)

    @. dY.w -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dY.w -= vgradc2f(cK)

    # Energy conservation

    @. dY.Yc.ρe -= hdiv(cuvw * (cρe + cp))
    @. dY.Yc.ρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))
end

function held_suarez_forcing!(dY, Y, p, t)
    @unpack cp, cσ, c_height_factor, cΔρT, cφ = p

    @. cσ = cp / p_0 # assumes that cp has already been updated
    @. c_height_factor = max(0, (cσ - σ_b) / (1 - σ_b))
    @. cΔρT =
        (k_a + (k_s - k_a) * c_height_factor * cos(cφ)^4) *
        Y.Yc.ρ *
        (
            cp / (Y.Yc.ρ * R_d) - # cT
            max(
                T_min,
                (T_equator - ΔT_y * sin(cφ)^2 - Δθ_z * log(cσ) * cos(cφ)^2) *
                cσ^(R_d / cp_d),
            ) # cT_equil
        )

    @. dY.uₕ -= (k_f * c_height_factor) * Y.uₕ
    if :ρθ in propertynames(Y.Yc)
        @. dY.Yc.ρθ -= cΔρT * (p_0 / cp)^κ
    elseif :ρe in propertynames(Y.Yc)
        @. dY.Yc.ρe -= cΔρT * cv_d
    end
end

function final_adjustments!(
    dY,
    Y,
    p,
    t;
    use_flux_correction,
    use_rayleigh_sponge,
)
    if use_flux_correction
        @. dY.Yc.ρ += fcc(Y.w, Y.Yc.ρ)
        if :ρθ in propertynames(Y.Yc)
            @. dY.Yc.ρθ += fcc(Y.w, Y.Yc.ρθ)
        elseif :ρe in propertynames(Y.Yc)
            @. dY.Yc.ρe += fcc(Y.w, Y.Yc.ρe)
        end
    end

    if use_rayleigh_sponge
        @unpack cβ = p
        @. dY.uₕ -= cβ * Y.uₕ
        @. dY.w -= cβ * Y.w
    end

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)
end
