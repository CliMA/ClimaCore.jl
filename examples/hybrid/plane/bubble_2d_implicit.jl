# Dennis's notes:
# I shifted the domain from (-500, 500) to (0, 1000) for simplicity
# Using discrete hydrostatic balance to set the unperturbed initial condition greatly stabilized the inertial gravity wave test. Should we similarly modify this initial condition?

using ClimaCorePlots, Plots

# Reference paper:
# https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml,
# Section 5a

# Constant for switching between different energy variables (:ρe, :ρθ, :ρe_int)
const ᶜ𝔼_name = :ρe

# Constants required by "staggered_nonhydrostatic_model.jl"
const p_0 = FT(1.0e5)
const R_d = FT(287.058)
const κ = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.8)
const f = FT(0)
include("../staggered_nonhydrostatic_model.jl")

# Additional constants required for rising bubble initial condition
x_max = FT(1000)
z_max = FT(1000)
const x_c = FT(500)
const r_c = FT(250)
const z_c = FT(350)
const θ_b = FT(300)
const θ_c = FT(0.4)

# Additional values required for driver
upwinding_mode = :third_order
npoly = 4
horizontal_mesh = periodic_line_mesh(; x_max, x_elem = 10)
z_elem = 40
t_end = FT(500)
dt = FT(0.01)
dt_save_to_sol = FT(10)
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (;
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = ᶜ𝔼_name == :ρe ? :no_∂ᶜp∂ᶜK : :exact,
    ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact,
)
show_progress_bar = isinteractive()

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) =
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(100))
additional_tendency!(Yₜ, Y, p, t) = hyperdiffusion_tendency!(Yₜ, Y, p, t)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates

    # potential temperature perturbation
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? FT(0.5) * θ_c * (1 + cospi(r / r_c)) : FT(0)

    θ = θ_b + θ_p
    π_exn = 1 - grav * z / (cp_d * θ)
    T = π_exn * θ
    p = p_0 * π_exn^(cp_d / R_d)
    ρ = p / (R_d * T)
    uₕ_local = Geometry.UVVector(FT(0), FT(0))
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

function compute_ρe(Y)
    if ᶜ𝔼_name == :ρθ
        ᶜz = Fields.coordinate_field(Y.c).z
        ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
        ᶜT = @. Y.c.ρθ / (Y.c.ρ * (p_0 / pressure_ρθ(Y.c.ρθ))^(R_d / cp_d))
        return @. Y.c.ρ * cv_d * (ᶜT - T_tri) + ᶜK + grav * ᶜz
    elseif ᶜ𝔼_name == :ρe
        return @. Y.c.ρe
    elseif ᶜ𝔼_name == :ρe_int
        ᶜz = Fields.coordinate_field(Y.c).z
        ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
        return @. Y.c.ρe_int + ᶜK + grav * ᶜz
    end
end

function postprocessing(sol, output_dir)
    anim = Plots.@animate for u in sol.u
        Plots.plot(compute_ρe(u) ./ u.c.ρ)
    end
    Plots.mp4(anim, joinpath(output_dir, "total_energy.mp4"), fps = 20)
    
    anim = Plots.@animate for u in sol.u
        Plots.plot(Geometry.WVector.(Geometry.Covariant13Vector.(ᶜinterp.(u.f.w))))
    end
    Plots.mp4(anim, joinpath(output_dir, "vel_w.mp4"), fps = 20)
    
    anim = Plots.@animate for u in sol.u
        Plots.plot(Geometry.UVector.(Geometry.Covariant13Vector.(u.c.uₕ)))
    end
    Plots.mp4(anim, joinpath(output_dir, "vel_u.mp4"), fps = 20)
    
    ρe_sums = [sum(compute_ρe(u)) for u in sol.u]
    Plots.png(
        Plots.plot((ρe_sums .- ρe_sums[1]) ./ ρe_sums[1]),
        joinpath(output_dir, "energy_cons.png"),
    )

    ρ_sums = [sum(u.c.ρ) for u in sol.u]
    Plots.png(
        Plots.plot((ρ_sums .- ρ_sums[1]) ./ ρ_sums[1]),
        joinpath(output_dir, "mass_cons.png"),
    )
end
