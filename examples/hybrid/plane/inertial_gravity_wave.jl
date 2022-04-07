using Printf
using ProgressLogging
using ClimaCorePlots, Plots

# Reference paper: https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.2105

# Constants for switching between different experiment setups
const is_small_scale = true
const 𝔼_name = :ρe
const is_discrete_hydrostatic_balance = true # `false` causes large oscillations

# Constants required by "staggered_nonhydrostatic_model.jl"
const p_0 = FT(1.0e5)
const R_d = FT(287.0)
const κ = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.80616)
const f = is_small_scale ? FT(0) : 2 * sin(π / 4) * 2π / FT(86164.09)
include("../staggered_nonhydrostatic_model.jl")

# Additional constants required for inertial gravity wave initial condition
const z_top = FT(10e3)
const x_max = is_small_scale ? FT(300e3) : FT(6000e3)
const x_mid = is_small_scale ? FT(100e3) : FT(3000e3)
const d = is_small_scale ? FT(5e3) : FT(100e3)
const u₀ = is_small_scale ? FT(20) : FT(0)
const v₀ = FT(0)
const T₀ = FT(250)
const ΔT = FT(0.01)

# Other convenient constants used in reference paper
const δ = grav / (R_d * T₀)        # Bretherton height parameter
const cₛ² = cp_d / cv_d * R_d * T₀ # speed of sound squared
const ρₛ = p_0 / (R_d * T₀)        # air density at surface

p₀(z) = p_0 * exp(-δ * z)
Tb_init(x, z) = ΔT * exp(-(x - x_mid)^2 / d^2) * sin(π * z / z_top)
T′_init(x, z) = Tb_init(x, z) * exp(δ * z / 2)

function make_center_initial_condition(Δz)
    # Yₜ.f.w = 0 in implicit tendency                                        ==>
    # -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥᶜΦ) = 0                             ==>
    # ᶠgradᵥ(ᶜp) = -grav * ᶠinterp(ᶜρ)                                       ==>
    # (p(z + Δz) - p(z)) / Δz = -grav * (ρ(z + Δz) + ρ(z)) / 2               ==>
    # p(z + Δz) + grav * Δz * ρ(z + Δz) / 2 = p(z) - grav * Δz * ρ(z) / 2    ==>
    # p(z + Δz) * (1 + δ * Δz / 2) = p(z) * (1 - δ * Δz / 2)                 ==>
    # p(z + Δz) / p(z) = (1 - δ * Δz / 2) / (1 + δ * Δz / 2)                 ==>
    # p(z) = p(0) * ((1 - δ * Δz / 2) / (1 + δ * Δz / 2))^(z / Δz)
    p₀_discrete(z) = p_0 * ((1 - δ * Δz / 2) / (1 + δ * Δz / 2))^(z / Δz)
    p₀_func = is_discrete_hydrostatic_balance ? p₀_discrete : p₀

    function center_initial_condition(local_geometry)
        (; x, z) = local_geometry.coordinates
        p = p₀_func(z)
        T = T₀ + T′_init(x, z)
        ρ = p / (R_d * T)
        uₕ_local = Geometry.UVVector(u₀, v₀)
        uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)
        if 𝔼_name == :ρθ
            ρθ = ρ * T * (p_0 / p)^(R_d / cp_d)
            return (; ρ, ρθ, uₕ)
        elseif 𝔼_name == :ρe
            ρe = ρ * (cv_d * (T - T_tri) + norm_sqr(uₕ_local) / 2 + grav * z)
            return (; ρ, ρe, uₕ)
        elseif 𝔼_name == :ρe_int
            ρe_int = ρ * cv_d * (T - T_tri)
            return (; ρ, ρe_int, uₕ)
        end
    end
    return center_initial_condition
end
function make_face_initial_condition()
    face_initial_condition(local_geometry) =
        (; w = Geometry.Covariant3Vector(FT(0)))
    return face_initial_condition
end

# TODO: Use full set of Δxs once the solution can be computed more quickly.
# Δxs = is_small_scale ? FT[1000, 500, 250, 125, 50, 25] :
#     FT[20e3, 10e3, 5e3, 2.5e3]
Δxs = is_small_scale ? FT[1000,] : FT[20e3, 10e3]
Δzs = is_small_scale ? Δxs ./ 2 : Δxs ./ 40
setups = map(Δxs, Δzs) do Δx, Δz
    npoly = 1
    x_elem = Int(x_max / (Δx * npoly))
    t_end = is_small_scale ? FT(60 * 60 * 0.5) : FT(60 * 60 * 8)
    dt_for_first_Δx = is_small_scale ? FT(1.5) : FT(30) # this depends on npoly
    animation_duration = FT(5) # output a 5-second gif
    fps = 2 # play the gif at 2 frames per second
    return HybridDriverSetup(;
        center_initial_condition = make_center_initial_condition(Δz),
        face_initial_condition = make_face_initial_condition(),
        horizontal_mesh = periodic_line_mesh(; x_max, x_elem),
        npoly,
        z_max = z_top,
        z_elem = Int(z_top / Δz),
        t_end,
        dt = dt_for_first_Δx / Δxs[1] * Δx,
        dt_save_to_sol = t_end / (animation_duration * fps),
        ode_algorithm = Rosenbrock23,
        jacobian_flags = (;
            ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = 𝔼_name == :ρe ? :no_∂ᶜp∂ᶜK : :exact,
            ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact,
        ),
    )
end

function postprocessing(sols, output_dir)
    if 𝔼_name == :ρθ
        press = Y -> @. pressure_ρθ(Y.c.ρθ)
        T′ = Y -> begin
            ᶜp = @. pressure_ρθ(Y.c.ρθ)
            @. Y.c.ρθ / Y.c.ρ * (ᶜp / p_0)^(R_d / cp_d) - T₀
        end
    elseif 𝔼_name == :ρe
        press = Y -> begin
            ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
            ᶜΦ = Fields.coordinate_field(Y.c).z .* grav
            @. pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)
        end
        T′ = Y -> begin
            ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
            ᶜΦ = Fields.coordinate_field(Y.c).z .* grav
            @. (Y.c.ρe / Y.c.ρ - ᶜK - ᶜΦ) / cv_d + T_tri - T₀
        end
    elseif 𝔼_name == :ρe_int
        press = Y -> @. pressure_ρe_int(Y.c.ρe_int, Y.c.ρ)
        T′ = Y -> @. Y.c.ρe_int / Y.c.ρ / cv_d + T_tri - T₀
    end
    u′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:1 - u₀
    v′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:2 - v₀
    w′ = Y -> @. Geometry.WVector(Y.f.w).components.data.:1

    ρfb_init_array = ρfb_init_coefs(1, 600, 40)

    for index in 1:length(sols)
        Δx = Δxs[index]
        Δz = Δzs[index]
        sol = sols[index]
        ᶜlocal_geometry = Fields.local_geometry_field(sol.u[1].c)
        ᶠlocal_geometry = Fields.local_geometry_field(sol.u[1].f)
        lin_cache = linear_solution_cache(ᶜlocal_geometry, ᶠlocal_geometry)

        p₀_discrete(z) = p_0 * ((1 - δ * Δz / 2) / (1 + δ * Δz / 2))^(z / Δz)
        p₀_func = is_discrete_hydrostatic_balance ? p₀_discrete : p₀
        p′ = Y -> press(Y) .- p₀_func.(ᶜlocal_geometry.coordinates.z)
        ρ′ = Y -> @. Y.c.ρ - p₀_func(ᶜlocal_geometry.coordinates.z) / (R_d * T₀)

        println("Info for Δx = $Δx:\n")
        for iframe in (1, length(sol.t))
            t = sol.t[iframe]
            Y = sol.u[iframe]
            (; ᶜp′, ᶜρ′, ᶜu′, ᶜv′, ᶠw′, ᶜT′) =
                linear_solution(lin_cache, ρfb_init_array, t)
            println("Error norms at time t = $t:")
            for (name, f, var_lin) in (
                (:p′, p′, ᶜp′),
                (:ρ′, ρ′, ᶜρ′),
                (:u′, u′, ᶜu′),
                (:v′, v′, ᶜv′),
                (:w′, w′, ᶠw′),
                (:T′, T′, ᶜT′),
            )
                var = f(Y)
                strings = (
                    norm_strings(var, var_lin, 2)...,
                    norm_strings(var, var_lin, Inf)...,
                )
                println("ϕ = $name: ", join(strings, ", "))
            end
            println()
        end
    end

    # Animation is very slow, so only do it for the first solution
    sol = sols[1]
    ᶜlocal_geometry = Fields.local_geometry_field(sol.u[1].c)
    ᶠlocal_geometry = Fields.local_geometry_field(sol.u[1].f)
    lin_cache = linear_solution_cache(ᶜlocal_geometry, ᶠlocal_geometry)
    Y_lin = similar(sol.u[1])
    anim_vars = (
        (:Tprime, T′, is_small_scale ? 0.014 : 0.014),
        (:uprime, u′, is_small_scale ? 0.042 : 0.014),
        (:wprime, w′, is_small_scale ? 0.0042 : 0.0014),
    )
    anims = [Animation() for _ in 1:(3 * length(anim_vars))]
    @progress "Animations" for iframe in 1:length(sol.t)
        t = sol.t[iframe]
        Y = sol.u[iframe]
        linear_solution!(Y_lin, lin_cache, ρfb_init_array, t)
        for (ivar, (_, f, lim)) in enumerate(anim_vars)
            var = f(Y)
            var_lin = f(Y_lin)
            var_rel_err = @. (var - var_lin) / (abs(var_lin) + eps(FT))
            # adding eps(FT) to the denominator prevents divisions by 0
            frame(anims[3 * ivar - 2], plot(var_lin, clim = (-lim, lim)))
            frame(anims[3 * ivar - 1], plot(var, clim = (-lim, lim)))
            frame(anims[3 * ivar], plot(var_rel_err, clim = (-10, 10)))
        end
    end
    for (ivar, (name, _, _)) in enumerate(anim_vars)
        mp4(anims[3 * ivar - 2], joinpath(output_dir, "$(name)_lin.mp4"); fps)
        mp4(anims[3 * ivar - 1], joinpath(output_dir, "$name.mp4"); fps)
        mp4(anims[3 * ivar], joinpath(output_dir, "$(name)_rel_err.mp4"); fps)
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

# TODO: Verify that this converges as the resolution increases.
function ρfb_init_coefs(npoly, x_elem, z_elem)
    max_ikx = x_elem * npoly
    max_ikz = 2 * z_elem

    # coordinates
    horizontal_mesh = periodic_line_mesh(; x_max, x_elem)
    h_space = make_horizontal_space(horizontal_mesh, npoly)
    center_space, _ = make_hybrid_spaces(h_space, 2 * z_top, 2 * z_elem)
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
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
    ρfb_init_array = Array{Complex{FT}}(undef, 2 * max_ikx + 1, 2 * max_ikz + 1)
    ᶜfourier_factor = Fields.Field(Complex{FT}, axes(ᶜlocal_geometry))
    ᶜintegrand = Fields.Field(Complex{FT}, axes(ᶜlocal_geometry))
    unit_integral = sum(one.(ᶜρb_init))
    @progress "ρfb t=0" for ikx in (-max_ikx):max_ikx, ikz in (-max_ikz):max_ikz
        kx = 2 * π / x_max * ikx
        kz = 2 * π / (2 * z_top) * ikz
        @. ᶜfourier_factor = exp(im * (kx * ᶜx + kz * ᶜz))
        @. ᶜintegrand = ᶜρb_init / ᶜfourier_factor
        ρfb_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1] =
            sum(ᶜintegrand) / unit_integral
    end
    return ρfb_init_array
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

function linear_solution!(Y, lin_cache, ρfb_init_array, t)
    (; ᶜx, ᶠx, ᶜz, ᶠz, ᶜp₀, ᶜρ₀, ᶜu₀, ᶜv₀, ᶠw₀) = lin_cache
    (; ᶜbretherton_factor_pρ) = lin_cache
    (; ᶜbretherton_factor_uvwT, ᶠbretherton_factor_uvwT) = lin_cache
    (; ᶜfourier_factor, ᶠfourier_factor) = lin_cache
    (; ᶜpb, ᶜρb, ᶜub, ᶜvb, ᶠwb, ᶜp, ᶜρ, ᶜu, ᶜv, ᶠw, ᶜT) = lin_cache

    ᶜpb .= FT(0)
    ᶜρb .= FT(0)
    ᶜub .= FT(0)
    ᶜvb .= FT(0)
    ᶠwb .= FT(0)
    max_ikx, max_ikz = (size(ρfb_init_array) .- 1) .÷ 2
    @progress "Y_lin" for ikx in (-max_ikx):max_ikx, ikz in (-max_ikz):max_ikz
        kx = 2 * π / x_max * ikx
        kz = 2 * π / (2 * z_top) * ikz

        # Fourier coefficient of ᶜρb_init (for current kx and kz)
        ρfb_init = ρfb_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1]

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
        pfb = -ρfb_init * (L₁ + L₋₁ * f^2) * C₁
        ρfb =
            ρfb_init *
            (L₃ + L₁ * (p₁ + C₂) + L₋₁ * f^2 * (cₛ² * (kz^2 + δ^2 / 4) + C₂))
        ufb = ρfb_init * L₀ * im * kx * C₁ / ρₛ
        vfb = -ρfb_init * L₋₁ * im * kx * f * C₁ / ρₛ
        wfb = -ρfb_init * (L₂ + L₀ * (f^2 + cₛ² * kx^2)) * grav / ρₛ

        # Bretherton transforms of final perturbations
        @. ᶜpb += real(pfb * ᶜfourier_factor)
        @. ᶜρb += real(ρfb * ᶜfourier_factor)
        @. ᶜub += real(ufb * ᶜfourier_factor)
        @. ᶜvb += real(vfb * ᶜfourier_factor)
        @. ᶠwb += real(wfb * ᶠfourier_factor)
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
    if 𝔼_name == :ρθ
        @. Y.c.ρθ = ᶜρ * ᶜT * (p_0 / ᶜp)^(R_d / cp_d)
    elseif 𝔼_name == :ρe
        @. Y.c.ρe =
            ᶜρ * (
                cv_d * (ᶜT - T_tri) +
                (ᶜu^2 + ᶜv^2 + ᶜinterp(ᶠw)^2) / 2 +
                grav * ᶜz
            )
    elseif 𝔼_name == :ρe_int
        @. Y.c.ρe_int = ᶜρ * cv_d * (ᶜT - T_tri)
    end
    @. Y.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(ᶜu, ᶜv))
    @. Y.f.w = Geometry.Covariant3Vector(Geometry.WVector(ᶠw))
end

function linear_solution(lin_cache, ρfb_init_array, t)
    (; ᶜx, ᶠx, ᶜz, ᶠz, ᶜp₀, ᶜρ₀, ᶜbretherton_factor_pρ) = lin_cache
    (; ᶜbretherton_factor_uvwT, ᶠbretherton_factor_uvwT) = lin_cache
    (; ᶜfourier_factor, ᶠfourier_factor, ᶜpb, ᶜρb, ᶜub, ᶜvb, ᶠwb) = lin_cache

    ᶜpb .= FT(0)
    ᶜρb .= FT(0)
    ᶜub .= FT(0)
    ᶜvb .= FT(0)
    ᶠwb .= FT(0)
    max_ikx, max_ikz = (size(ρfb_init_array) .- 1) .÷ 2
    @progress "Y_lin" for ikx in (-max_ikx):max_ikx, ikz in (-max_ikz):max_ikz
        kx = 2 * π / x_max * ikx
        kz = 2 * π / (2 * z_top) * ikz

        # Fourier coefficient of ᶜρb_init (for current kx and kz)
        ρfb_init = ρfb_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1]

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
        pfb = -ρfb_init * (L₁ + L₋₁ * f^2) * C₁
        ρfb =
            ρfb_init *
            (L₃ + L₁ * (p₁ + C₂) + L₋₁ * f^2 * (cₛ² * (kz^2 + δ^2 / 4) + C₂))
        ufb = ρfb_init * L₀ * im * kx * C₁ / ρₛ
        vfb = -ρfb_init * L₋₁ * im * kx * f * C₁ / ρₛ
        wfb = -ρfb_init * (L₂ + L₀ * (f^2 + cₛ² * kx^2)) * grav / ρₛ

        # Bretherton transforms of final perturbations
        @. ᶜpb += real(pfb * ᶜfourier_factor)
        @. ᶜρb += real(ρfb * ᶜfourier_factor)
        @. ᶜub += real(ufb * ᶜfourier_factor)
        @. ᶜvb += real(vfb * ᶜfourier_factor)
        @. ᶠwb += real(wfb * ᶠfourier_factor)
        # The imaginary components should be 0 (or at least very close to 0).
    end

    # final state
    ᶜp′ = @. ᶜpb * ᶜbretherton_factor_pρ
    ᶜρ′ = @. ᶜρb * ᶜbretherton_factor_pρ
    ᶜu′ = @. ᶜub * ᶜbretherton_factor_uvwT
    ᶜv′ = @. ᶜvb * ᶜbretherton_factor_uvwT
    ᶠw′ = @. ᶠwb * ᶠbretherton_factor_uvwT
    ᶜT′ = @. (ᶜp₀ + ᶜp′) / (R_d * (ᶜρ₀ + ᶜρ′)) - T₀

    return (; ᶜp′, ᶜρ′, ᶜu′, ᶜv′, ᶠw′, ᶜT′)
end
