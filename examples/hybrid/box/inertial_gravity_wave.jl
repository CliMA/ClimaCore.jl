using Printf
using ProgressLogging
using ClimaCorePlots, Plots

# Reference paper: https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.2105

const is_small_scale = true
const ᶜ𝔼_name = :ρe
const is_discrete_hydrostatic_balance = true # `false` causes large oscillations

# Constants required for staggered_nonhydrostatic_model
const p_0 = FT(1.0e5)    # reference pressure
const R_d = FT(287.0)    # dry specific gas constant
const κ = FT(2 / 7)      # kappa
const cp_d = R_d / κ     # heat capacity at constant pressure
const cv_d = cp_d - R_d  # heat capacity at constant volume
const γ = cp_d / cv_d    # heat capacity ratio
const T_tri = FT(273.16) # triple point temperature
const grav = FT(9.80616) # Earth's gravitational acceleration
# const Ω = ?            # Earth's rotation rate (not required for flat space)
const f =                # Coriolis frequency
    is_small_scale ? FT(0) : 2 * sin(π / 4) * 2π / FT(86164.09)

# Additional constants required for inertial gravity wave initial condition
const zmax = FT(10e3)
const xmax = is_small_scale ? FT(300e3) : FT(6000e3)
const xmid = is_small_scale ? FT(100e3) : FT(3000e3)
const d = is_small_scale ? FT(5e3) : FT(100e3)
const u₀ = is_small_scale ? FT(20) : FT(0)
const v₀ = FT(0)
const T₀ = FT(250)
const ΔT = FT(0.01)

# Other convenient constants used in reference paper
const δ = grav / (R_d * T₀)        # Bretherton height parameter
const cₛ² = cp_d / cv_d * R_d * T₀ # speed of sound squared
const ρₛ = p_0 / (R_d * T₀)        # air density at surface

# TODO: Loop over all domain setups used in reference paper
const Δx = is_small_scale ? FT(1e3) : FT(20e3)
const Δz = is_small_scale ? Δx / 2 : Δx / 40
zelem = Int(zmax / Δz)
npoly, xelem = 1, Int(xmax / Δx) # max small-scale dt = 1.5
# npoly, xelem = 4, Int(xmax / (Δx * (4 + 1))) # max small-scale dt = 0.8

# Animation-related values
animation_duration = FT(10)
fps = 2

# Values required for driver
space =
    ExtrudedSpace(; zmax, zelem, hspace = PeriodicLine(; xmax, xelem, npoly))
tend = is_small_scale ? FT(60 * 60 * 0.5) : FT(60 * 60 * 8)
dt = is_small_scale ? FT(1.5) : FT(20)
dt_save_to_sol = tend / (animation_duration * fps)
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = ᶜ𝔼_name == :ρe ? :no_∂p∂K : :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
show_progress_bar = true

if is_discrete_hydrostatic_balance
    # Yₜ.f.w = 0 in implicit tendency                                        ==>
    # -(ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) + ᶠ∇ᵥᶜΦ) = 0                                        ==>
    # ᶠ∇ᵥ(ᶜp) = -grav * ᶠI(ᶜρ)                                               ==>
    # (p(z + Δz) - p(z)) / Δz = -grav * (ρ(z + Δz) + ρ(z)) / 2               ==>
    # p(z + Δz) + grav * Δz * ρ(z + Δz) / 2 = p(z) - grav * Δz * ρ(z) / 2    ==>
    # p(z + Δz) * (1 + δ * Δz / 2) = p(z) * (1 - δ * Δz / 2)                 ==>
    # p(z + Δz) / p(z) = (1 - δ * Δz / 2) / (1 + δ * Δz / 2)                 ==>
    # p(z) = p(0) * ((1 - δ * Δz / 2) / (1 + δ * Δz / 2))^(z / Δz)
    p₀(z) = p_0 * ((1 - δ * Δz / 2) / (1 + δ * Δz / 2))^(z / Δz)
else
    p₀(z) = p_0 * exp(-δ * z)
end
Tb_init(x, z) = ΔT * exp(-(x - xmid)^2 / d^2) * sin(π * z / zmax)
T′_init(x, z) = Tb_init(x, z) * exp(δ * z / 2)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    p = p₀(z)
    T = T₀ + T′_init(x, z)
    ρ = p / (R_d * T)
    uₕ_local = Geometry.UVVector(u₀, v₀)
    uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)
    if ᶜ𝔼_name == :ρθ
        ρθ = ρ * T * (p_0 / p)^(R_d / cp_d)
        return (; ρ, ρθ, uₕ)
    elseif ᶜ𝔼_name == :ρe
        ρe = ρ * (cv_d * (T - T_tri) + norm_sqr(uₕ_local) / 2 + grav * z)
        return (; ρ, ρe, uₕ)
    elseif ᶜ𝔼_name == :ρe_int
        ρe_int = ρ * cv_d * (T - T_tri)
        return (; ρ, ρe_int, uₕ)
    end
end
face_initial_condition(local_geometry) =
    (; w = Geometry.Covariant3Vector(FT(0)))

function postprocessing(sol, p, path)
    ᶜlocal_geometry = Fields.local_geometry_field(sol.u[1].c)
    ᶠlocal_geometry = Fields.local_geometry_field(sol.u[1].f)
    lin_cache = linear_solution_cache(ᶜlocal_geometry, ᶠlocal_geometry)
    Y_lin = similar(sol.u[1])

    ρ′ = Y -> @. Y.c.ρ - p₀(ᶜlocal_geometry.coordinates.z) / (R_d * T₀)
    if ᶜ𝔼_name == :ρθ
        T′ =
            Y -> @. Y.c.ρθ / Y.c.ρ * (pressure_ρθ(Y.c.ρθ) / p_0)^(R_d / cp_d) -
               T₀
    elseif ᶜ𝔼_name == :ρe
        T′ = Y -> begin
            @. p.ᶜK = norm_sqr(C123(Y.c.uₕ) + C123(ᶜI(Y.f.w))) / 2
            @. (Y.c.ρe / Y.c.ρ - p.ᶜK - p.ᶜΦ) / cv_d + T_tri - T₀
        end
    elseif ᶜ𝔼_name == :ρe_int
        T′ = Y -> @. Y.c.ρe_int / Y.c.ρ / cv_d + T_tri - T₀
    end
    u′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:1 - u₀
    v′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:2 - v₀
    w′ = Y -> @. Geometry.WVector(Y.f.w).components.data.:1

    for iframe in (1, length(sol.t))
        t = sol.t[iframe]
        Y = sol.u[iframe]
        linear_solution!(Y_lin, lin_cache, t)
        println("Error norms at time t = $t:")
        for (name, f) in ((:ρ′, ρ′), (:T′, T′), (:u′, u′), (:v′, v′), (:w′, w′))
            var = f(Y)
            var_lin = f(Y_lin)
            strings = (
                norm_strings(var, var_lin, 2)...,
                norm_strings(var, var_lin, Inf)...,
            )
            println("ϕ = $name: ", join(strings, ", "))
        end
        println()
    end

    anim_vars = (
        (:Tprime, T′, is_small_scale ? 0.014 : 0.014),
        (:uprime, u′, is_small_scale ? 0.042 : 0.014),
        (:wprime, w′, is_small_scale ? 0.0042 : 0.0014),
    )
    anims = [Animation() for _ in 1:(3 * length(anim_vars))]
    @progress "Animations" for iframe in 1:length(sol.t)
        t = sol.t[iframe]
        Y = sol.u[iframe]
        linear_solution!(Y_lin, lin_cache, t)
        for (ivar, (_, f, var_max)) in enumerate(anim_vars)
            var = f(Y)
            var_lin = f(Y_lin)
            var_rel_err = @. (var - var_lin) / (abs(var_lin) + eps(FT))
            # adding eps(FT) to the denominator prevents divisions by 0
            frame(anims[3 * ivar], plot(var_lin, clim = (-var_max, var_max)))
            frame(anims[3 * ivar - 1], plot(var, clim = (-var_max, var_max)))
            frame(anims[3 * ivar - 2], plot(var_rel_err, clim = (-10, 10)))
        end
    end
    for (ivar, (name, _, _)) in enumerate(anim_vars)
        mp4(anims[3 * ivar], joinpath(path, "$(name)_lin.mp4"); fps)
        mp4(anims[3 * ivar - 1], joinpath(path, "$name.mp4"); fps)
        mp4(anims[3 * ivar - 2], joinpath(path, "$(name)_rel_err.mp4"); fps)
    end
end

function norm_strings(var, var_lin, p)
    norm_err = norm(var .- var_lin, p; normalize = false)
    scaled_norm_err = norm_err / norm(var_lin, p; normalize = false)
    return (
        @sprintf("‖ϕ‖_%d = %-#9.4g", p, norm(var, p; normalize = false)),
        @sprintf("‖ϕ - ϕ_lin‖_%d = %-#9.4g", p, norm_err),
        @sprintf("‖ϕ - ϕ_lin‖_%d/‖ϕ_lin‖_%d = %-#9.4g", p, p, scaled_norm_err),
    )
end

# min_λx = 2 * (xmax / xelem) / upsampling_factor # should this involve npoly?
# min_λz = 2 * (zmax / zelem) / upsampling_factor
# min_λx = 2 * π / max_kx = xmax / max_ikx
# min_λz = 2 * π / max_kz = 2 * zmax / max_ikz
# max_ikx = xmax / min_λx = upsampling_factor * xelem / 2
# max_ikz = 2 * zmax / min_λz = upsampling_factor * zelem
function ρ̂b_init_coefs(
    upsampling_factor = 3,
    max_ikx = upsampling_factor * xelem ÷ 2,
    max_ikz = upsampling_factor * zelem,
)
    # upsampled coordinates (more upsampling gives more accurate coefficients)
    space = ExtrudedSpace(;
        zmax,
        zelem = upsampling_factor * zelem,
        hspace = PeriodicLine(; xmax, xelem = upsampling_factor * xelem, npoly),
    )
    ᶜlocal_geometry, _ = local_geometry_fields(space)
    ᶜx = ᶜlocal_geometry.coordinates.x
    ᶜz = ᶜlocal_geometry.coordinates.z

    # Bretherton transform of initial perturbation
    linearize_density_perturbation = false
    if linearize_density_perturbation
        ᶜρb_init = @. -ρₛ * Tb_init(ᶜx, ᶜz) / T₀
    else
        ᶜp₀ = @. p₀(ᶜz)
        ᶜρ₀ = @. ᶜp₀ / (R_d * T₀)
        ᶜρ′_init = @. ᶜp₀ / (R_d * (T₀ + T′_init(ᶜx, ᶜz))) - ᶜρ₀
        ᶜbretherton_factor_pρ = @. exp(-δ * ᶜz / 2)
        ᶜρb_init = @. ᶜρ′_init / ᶜbretherton_factor_pρ
    end

    # Fourier coefficients of Bretherton transform of initial perturbation
    ρ̂b_init_array = Array{Complex{FT}}(undef, 2 * max_ikx + 1, 2 * max_ikz + 1)
    ᶜfourier_factor = Fields.Field(Complex{FT}, axes(ᶜlocal_geometry))
    ᶜintegrand = Fields.Field(Complex{FT}, axes(ᶜlocal_geometry))
    unit_integral = 2 * sum(one.(ᶜρb_init))
    # Since the coefficients are for a modified domain of height 2 * zmax, the
    # unit integral over the domain must be multiplied by 2 to ensure correct
    # normalization. On the other hand, ᶜρb_init is assumed to be 0 outside of
    # the "true" domain, so the integral of ᶜintegrand should not be modified.
    @progress "ρ̂b_init" for ikx in (-max_ikx):max_ikx,
        ikz in (-max_ikz):max_ikz

        kx = 2 * π / xmax * ikx
        kz = 2 * π / (2 * zmax) * ikz
        @. ᶜfourier_factor = exp(im * (kx * ᶜx + kz * ᶜz))
        @. ᶜintegrand = ᶜρb_init / ᶜfourier_factor
        ρ̂b_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1] =
            sum(ᶜintegrand) / unit_integral
    end
    return ρ̂b_init_array
end

function linear_solution_cache(ᶜlocal_geometry, ᶠlocal_geometry)
    ᶜz = ᶜlocal_geometry.coordinates.z
    ᶠz = ᶠlocal_geometry.coordinates.z
    ᶜp₀ = @. p₀(ᶜz)
    return (;
        # coordinates
        ᶜx = ᶜlocal_geometry.coordinates.x,
        ᶠx = ᶠlocal_geometry.coordinates.x,
        ᶜz,
        ᶠz,

        # background state
        ᶜp₀,
        ᶜρ₀ = (@. ᶜp₀ / (R_d * T₀)),
        ᶜu₀ = map(_ -> u₀, ᶜlocal_geometry),
        ᶜv₀ = map(_ -> v₀, ᶜlocal_geometry),
        ᶠw₀ = map(_ -> FT(0), ᶠlocal_geometry),

        # Bretherton transform factors
        ᶜbretherton_factor_pρ = (@. exp(-δ * ᶜz / 2)),
        ᶜbretherton_factor_uvwT = (@. exp(δ * ᶜz / 2)),
        ᶠbretherton_factor_uvwT = (@. exp(δ * ᶠz / 2)),

        # Fourier coefficients of Bretherton transform of initial perturbation
        ρ̂b_init_array = ρ̂b_init_coefs(),

        # Fourier transform factors
        ᶜfourier_factor = Fields.Field(Complex{FT}, axes(ᶜlocal_geometry)),
        ᶠfourier_factor = Fields.Field(Complex{FT}, axes(ᶠlocal_geometry)),

        # Bretherton transform of final perturbation
        ᶜpb = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜρb = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜub = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜvb = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶠwb = Fields.Field(FT, axes(ᶠlocal_geometry)),

        # final state
        ᶜp = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜρ = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜu = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶜv = Fields.Field(FT, axes(ᶜlocal_geometry)),
        ᶠw = Fields.Field(FT, axes(ᶠlocal_geometry)),

        # final temperature
        ᶜT = Fields.Field(FT, axes(ᶜlocal_geometry)),
    )
end

function linear_solution!(Y, lin_cache, t)
    (;
        ᶜx,
        ᶠx,
        ᶜz,
        ᶠz,
        ᶜp₀,
        ᶜρ₀,
        ᶜu₀,
        ᶜv₀,
        ᶠw₀,
        ᶜbretherton_factor_pρ,
        ᶜbretherton_factor_uvwT,
        ᶠbretherton_factor_uvwT,
        ρ̂b_init_array,
        ᶜfourier_factor,
        ᶠfourier_factor,
        ᶜpb,
        ᶜρb,
        ᶜub,
        ᶜvb,
        ᶠwb,
        ᶜp,
        ᶜρ,
        ᶜu,
        ᶜv,
        ᶠw,
        ᶜT,
    ) = lin_cache

    ᶜpb .= FT(0)
    ᶜρb .= FT(0)
    ᶜub .= FT(0)
    ᶜvb .= FT(0)
    ᶠwb .= FT(0)
    max_ikx, max_ikz = (size(ρ̂b_init_array) .- 1) .÷ 2
    for ikx in (-max_ikx):max_ikx, ikz in (-max_ikz):max_ikz
        kx = 2 * π / xmax * ikx
        kz = 2 * π / (2 * zmax) * ikz

        # Fourier coefficient of ᶜρb_init (for current kx and kz)
        ρ̂b_init = ρ̂b_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1]

        # Fourier factors, shifted by u₀ * t along the x-axis
        @. ᶜfourier_factor = exp(im * (kx * (ᶜx - u₀ * t) + kz * ᶜz))
        @. ᶠfourier_factor = exp(im * (kx * (ᶠx - u₀ * t) + kz * ᶠz))

        # roots of a₁(s)
        p₁ = cₛ² * (kx^2 + kz^2 + δ^2 / 4) + f^2
        q₁ = grav * kx^2 * (cₛ² * δ - grav) + cₛ² * f^2 * (kz^2 + δ^2 / 4)
        α² = p₁ / 2 - sqrt(p₁^2 / 4 - q₁)
        β² = p₁ / 2 + sqrt(p₁^2 / 4 - q₁)
        α = sqrt(α²)
        β = sqrt(β²)

        # inverse Laplace transform of s^p/((s^2 + α^2)(s^2 + β^2)) for p ∈ -1:3
        if α == 0
            L₋₁ = (β² * t^2 / 2 - 1 + cos(β * t)) / β^4
            L₀ = (β * t - sin(β * t)) / β^3
        else
            L₋₁ =
                (-cos(α * t) / α² + cos(β * t) / β²) / (β² - α²) + 1 / (α² * β²)
            L₀ = (sin(α * t) / α - sin(β * t) / β) / (β² - α²)
        end
        L₁ = (cos(α * t) - cos(β * t)) / (β² - α²)
        L₂ = (-sin(α * t) * α + sin(β * t) * β) / (β² - α²)
        L₃ = (-cos(α * t) * α² + cos(β * t) * β²) / (β² - α²)

        # Fourier coefficients of Bretherton transforms of final perturbations
        C₁ = grav * (grav - cₛ² * (im * kz + δ / 2))
        C₂ = grav * (im * kz - δ / 2)
        p̂b = -ρ̂b_init * (L₁ + L₋₁ * f^2) * C₁
        ρ̂b =
            ρ̂b_init *
            (L₃ + L₁ * (p₁ + C₂) + L₋₁ * f^2 * (cₛ² * (kz^2 + δ^2 / 4) + C₂))
        ûb = ρ̂b_init * L₀ * im * kx * C₁ / ρₛ
        v̂b = -ρ̂b_init * L₋₁ * im * kx * f * C₁ / ρₛ
        ŵb = -ρ̂b_init * (L₂ + L₀ * (f^2 + cₛ² * kx^2)) * grav / ρₛ

        # Bretherton transforms of final perturbations
        @. ᶜpb += real(p̂b * ᶜfourier_factor)
        @. ᶜρb += real(ρ̂b * ᶜfourier_factor)
        @. ᶜub += real(ûb * ᶜfourier_factor)
        @. ᶜvb += real(v̂b * ᶜfourier_factor)
        @. ᶠwb += real(ŵb * ᶠfourier_factor)
        # The imaginary components should be 0 (or at least very close to 0).
    end

    # final state
    @. ᶜp = ᶜp₀ + ᶜpb * ᶜbretherton_factor_pρ
    @. ᶜρ = ᶜρ₀ + ᶜρb * ᶜbretherton_factor_pρ
    @. ᶜu = ᶜu₀ + ᶜub * ᶜbretherton_factor_uvwT
    @. ᶜv = ᶜv₀ + ᶜvb * ᶜbretherton_factor_uvwT
    @. ᶠw = ᶠw₀ + ᶠwb * ᶠbretherton_factor_uvwT
    @. ᶜT = ᶜp / (R_d * ᶜρ)

    @. Y.c.ρ = ᶜρ
    if ᶜ𝔼_name == :ρθ
        @. Y.c.ρθ = ᶜρ * ᶜT * (p_0 / ᶜp)^(R_d / cp_d)
    elseif ᶜ𝔼_name == :ρe
        @. Y.c.ρe =
            ᶜρ *
            (cv_d * (ᶜT - T_tri) + (ᶜu^2 + ᶜv^2 + ᶜI(ᶠw)^2) / 2 + grav * ᶜz)
    elseif ᶜ𝔼_name == :ρe_int
        @. Y.c.ρe_int = ᶜρ * cv_d * (ᶜT - T_tri)
    end
    @. Y.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(ᶜu, ᶜv))
    @. Y.f.w = Geometry.Covariant3Vector(Geometry.WVector(ᶠw))
end
