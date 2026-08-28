#=
julia --threads=8 --project=.buildkite
ENV["TEST_NAME"] = "plane/inertial_gravity_wave"
include(joinpath("examples", "hybrid", "driver.jl"))
=#
using Printf
using Test
using ProgressLogging
using Plots
import ClimaComms
ClimaComms.@import_required_backends

# Reference paper: https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.2105

include("inertial_gravity_wave_utils.jl")
import .InertialGravityWaveUtils as IGWU

# Constants for switching between different experiment setups
const is_small_scale = true
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
z_max = FT(10e3)
z_stretch_scale = FT(7e3)
const x_max = is_small_scale ? FT(300e3) : FT(6000e3)
const x_mid = is_small_scale ? FT(100e3) : FT(3000e3)
const d = is_small_scale ? FT(5e3) : FT(100e3)
const u₀ = is_small_scale ? FT(20) : FT(0)
const v₀ = FT(0)
const T₀ = FT(250)
const ΔT = FT(0.01)

# Additional values required for driver
upwinding_mode = :third_order # :none to switch to centered diff

# Other convenient constants used in reference paper
const δ = grav / (R_d * T₀)        # Bretherton height parameter
const cₛ² = cp_d / cv_d * R_d * T₀ # speed of sound squared
const ρₛ = p_0 / (R_d * T₀)        # air density at surface

# TODO: Loop over all domain setups used in reference paper
const Δx = is_small_scale ? FT(1e3) : FT(20e3)
const Δz = is_small_scale ? Δx / 2 : Δx / 40
z_elem = Int(z_max / Δz) # default 20 vertical elements
npoly, x_elem = 1, Int(x_max / Δx) # max small-scale dt = 1.5
# npoly, x_elem = 4, Int(x_max / (Δx * (4 + 1))) # max small-scale dt = 0.8

# Animation-related values
animation_duration = FT(5)
fps = 2

# Set up mesh
horizontal_mesh = periodic_line_mesh(; x_max, x_elem = x_elem)

# Additional values required for driver
# dt may need tweaking for is_small_scale = false
dt = is_small_scale ? FT(1.25) : FT(20)
t_end = is_small_scale ? FT(60 * 60 * 0.5) : FT(60 * 60 * 8)
dt_save_to_sol = t_end / (animation_duration * fps)
ode_algorithm = CTS.SSP333
jacobian_flags = (;
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK,
    ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact,
)
show_progress_bar = true

function discrete_hydrostatic_balance!(ᶠΔz, ᶜΔz, grav)

    # Yₜ.f.w = 0 in implicit tendency                                        ==>
    # -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥᶜΦ) = 0                             ==>
    # ᶠgradᵥ(ᶜp) = -grav * ᶠinterp(ᶜρ)                                       ==>
    # (p(z + Δz) - p(z)) / Δz = -grav * (ρ(z + Δz) + ρ(z)) / 2               ==>
    # p(z + Δz) + grav * Δz * ρ(z + Δz) / 2 = p(z) - grav * Δz * ρ(z) / 2    ==>
    # p(z + Δz) * (1 + δ * Δz / 2) = p(z) * (1 - δ * Δz / 2)                 ==>
    # p(z + Δz) / p(z) = (1 - δ * Δz / 2) / (1 + δ * Δz / 2)                 ==>
    # p(z + Δz) = p(z) * (1 - δ * Δz / 2) / (1 + δ * Δz / 2)
    ᶜp = similar(ᶜΔz)
    ᶜp1 = Fields.level(ᶜp, 1)
    ᶜΔz1 = Fields.level(ᶜΔz, 1)
    @. ᶜp1 = p_0 * (1 - δ * ᶜΔz1 / 4) / (1 + δ * ᶜΔz1 / 4)
    @inbounds for i in 1:(Spaces.nlevels(axes(ᶜp)) - 1)
        ᶜpi = parent(Fields.level(ᶜp, i))
        ᶜpi1 = parent(Fields.level(ᶜp, i + 1))
        ᶠΔzi1 = parent(Fields.level(ᶠΔz, Spaces.PlusHalf(i)))
        @. ᶜpi1 = ᶜpi * (1 - δ * ᶠΔzi1 / 2) / (1 + δ * ᶠΔzi1 / 2)
    end
    return ᶜp
end
Tb_init(x, z, ΔT, x_mid, d, z_max) =
    ΔT * exp(-(x - x_mid)^2 / d^2) * sin(π * z / z_max)
T′_init(x, z, ΔT, x_mid, d, z_max, δ) =
    Tb_init(x, z, ΔT, x_mid, d, z_max) * exp(δ * z / 2)
# Pressure definition, when not in discrete hydrostatic balance state
p₀(z) = @. p_0 * exp(-δ * z)

function center_initial_condition(ᶜlocal_geometry)
    ᶜx = ᶜlocal_geometry.coordinates.x
    ᶜz = ᶜlocal_geometry.coordinates.z
    # Correct pressure and density if in hydrostatic balance state
    if is_discrete_hydrostatic_balance
        center_space = axes(ᶜlocal_geometry)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
        ᶠΔz = Fields.Δz_field(face_space)
        ᶜΔz = Fields.Δz_field(center_space)
        ᶜp = discrete_hydrostatic_balance!(ᶠΔz, ᶜΔz, grav)
    else
        ᶜp = @. p₀(ᶜz)
    end
    T = @. T₀ + T′_init(ᶜx, ᶜz, ΔT, x_mid, d, z_max, δ)
    ᶜρ = @. ᶜp / (R_d * T)
    ᶜuₕ_local = @. Geometry.UVVector(u₀ * one(ᶜz), v₀ * one(ᶜz))
    ᶜuₕ = @. Geometry.Covariant12Vector(ᶜuₕ_local)
    ᶜρe = @. ᶜρ * (cv_d * (T - T_tri) + norm_sqr(ᶜuₕ_local) / 2 + grav * ᶜz)
    return NamedTuple{(:ρ, :ρe, :uₕ)}.(tuple.(ᶜρ, ᶜρe, ᶜuₕ))
end

function face_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    w = @. Geometry.Covariant3Vector(zero(z))
    return NamedTuple{(:w,)}.(tuple.(w))
end

function postprocessing(sol, output_dir)
    ᶜlocal_geometry = Fields.local_geometry_field(sol.u[1].c)
    ᶠlocal_geometry = Fields.local_geometry_field(sol.u[1].f)
    lin_cache =
        linear_solution_cache(comms_ctx, ᶜlocal_geometry, ᶠlocal_geometry)
    Y_lin = similar(sol.u[1])

    ρ′ = Y -> @. Y.c.ρ - p₀(ᶜlocal_geometry.coordinates.z) / (R_d * T₀)
    T′ = Y -> begin
        ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
        ᶜΦ = Fields.coordinate_field(Y.c).z .* grav
        @. (Y.c.ρe / Y.c.ρ - ᶜK - ᶜΦ) / cv_d + T_tri - T₀
    end
    u′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:1 - u₀
    v′ = Y -> @. Geometry.UVVector(Y.c.uₕ).components.data.:2 - v₀
    w′ = Y -> @. Geometry.WVector(Y.f.w).components.data.:1

    @time "print norms" @inbounds begin
        for iframe in (1, length(sol.t))
            t = sol.t[iframe]
            Y = sol.u[iframe]
            IGWU.linear_solution!(Y_lin, lin_cache, t, FT)
            println("Error norms at time t = $t:")
            for (name, f) in
                ((:ρ′, ρ′), (:T′, T′), (:u′, u′), (:v′, v′), (:w′, w′))
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
    end

    # The wave stays in the linear regime: the initial 0.01 K perturbation may
    # disperse but must neither blow up nor vanish. A broken discrete
    # hydrostatic balance drives max|T′| to ~46 K within minutes, so the upper
    # bound has teeth.
    @testset "perturbation stays linear" begin
        Y = sol.u[end]
        max_T′ = maximum(abs, T′(Y))
        max_w′ = maximum(abs, w′(Y))
        @test 1e-4 < max_T′ < 0.02
        @test max_w′ < 0.02
    end

    anim_vars = (
        (:Tprime, T′, is_small_scale ? 0.014 : 0.014),
        (:uprime, u′, is_small_scale ? 0.042 : 0.014),
        (:wprime, w′, is_small_scale ? 0.0042 : 0.0014),
    )
    anims = [Animation() for _ in 1:(3 * length(anim_vars))]
    @info "Creating animation with $(length(sol.t)) frames."
    @inbounds begin
        @progress "Animations" for iframe in 1:length(sol.t)
            t = sol.t[iframe]
            Y = sol.u[iframe]
            IGWU.linear_solution!(Y_lin, lin_cache, t, FT)
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
            mp4(
                anims[3 * ivar - 2],
                joinpath(output_dir, "$(name)_lin.mp4");
                fps,
            )
            mp4(anims[3 * ivar - 1], joinpath(output_dir, "$name.mp4"); fps)
            mp4(
                anims[3 * ivar],
                joinpath(output_dir, "$(name)_rel_err.mp4");
                fps,
            )
        end
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

function ρfb_init_coefs_params(
    comms_ctx,
    upsampling_factor = 3,
    max_ikx = upsampling_factor * x_elem ÷ 2,
    max_ikz = upsampling_factor * z_elem,
)
    # upsampled coordinates (more upsampling gives more accurate coefficients)
    horizontal_mesh =
        periodic_line_mesh(; x_max, x_elem = upsampling_factor * x_elem)
    h_space = make_horizontal_space(horizontal_mesh, npoly, comms_ctx)
    center_space, _ = make_hybrid_spaces(
        h_space,
        z_max,
        upsampling_factor * z_elem;
        z_stretch,
    )
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶜx = ᶜlocal_geometry.coordinates.x
    ᶜz = ᶜlocal_geometry.coordinates.z

    # Bretherton transform of initial perturbation
    linearize_density_perturbation = false
    if linearize_density_perturbation
        ᶜρb_init = @. -ρₛ * Tb_init(ᶜx, ᶜz, ΔT, x_mid, d, z_max) / T₀
    else
        ᶜp₀ = @. p₀(ᶜz)
        ᶜρ₀ = @. ᶜp₀ / (R_d * T₀)
        ᶜρ′_init =
            @. ᶜp₀ / (R_d * (T₀ + T′_init(ᶜx, ᶜz, ΔT, x_mid, d, z_max, δ))) -
               ᶜρ₀
        ᶜbretherton_factor_pρ = @. exp(-δ * ᶜz / 2)
        ᶜρb_init = @. ᶜρ′_init / ᶜbretherton_factor_pρ
    end
    combine(ρ, lg) = (; ρ, x = lg.coordinates.x, z = lg.coordinates.z)
    ᶜρb_init_xz = combine.(ᶜρb_init, ᶜlocal_geometry)

    # Fourier coefficients of Bretherton transform of initial perturbation
    ρfb_init_array = Array{Complex{FT}}(undef, 2 * max_ikx + 1, 2 * max_ikz + 1)
    unit_integral = 2 * sum(one.(ᶜρb_init))
    return (;
        ρfb_init_array,
        ᶜρb_init_xz,
        max_ikz,
        max_ikx,
        x_max,
        z_max,
        unit_integral,
    )
end

function linear_solution_cache(comms_ctx, ᶜlocal_geometry, ᶠlocal_geometry)
    ᶜz = ᶜlocal_geometry.coordinates.z
    ᶠz = ᶠlocal_geometry.coordinates.z
    ρfb_init_array_params = ρfb_init_coefs_params(comms_ctx)
    @time "ρfb_init_coefs!" IGWU.ρfb_init_coefs!(FT, ρfb_init_array_params)
    (; ρfb_init_array, ᶜρb_init_xz, unit_integral) = ρfb_init_array_params
    max_ikx, max_ikz = (size(ρfb_init_array) .- 1) .÷ 2

    get_xz(lg) = (; x = lg.coordinates.x, z = lg.coordinates.z)
    ᶠxz = get_xz.(ᶠlocal_geometry)
    ᶜxz = get_xz.(ᶜlocal_geometry)

    ᶜp₀ = @. p₀(ᶜz)
    return (;
        # globals
        R_d,
        x_max,
        z_max,
        p_0,
        cp_d,
        cv_d,
        grav,
        T_tri,
        u₀,
        δ,
        cₛ²,
        f,
        ρₛ,
        ᶜinterp,
        # coordinates
        ᶜx = ᶜlocal_geometry.coordinates.x,
        ᶠx = ᶠlocal_geometry.coordinates.x,
        ᶜz,
        ᶠz,
        ᶜxz,
        ᶠxz,

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
        ρfb_init_array,
        unit_integral,
        ᶜρb_init_xz,
        max_ikx,
        max_ikz,

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
