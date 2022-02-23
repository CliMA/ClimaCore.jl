using ClimaCorePlots, Plots

const is_small_scale = false

const x_center = is_small_scale ? FT(100e3) : FT(3000e3)              # m
const d = is_small_scale ? FT(5e3) : FT(100e3)                        # m
const f = is_small_scale ? FT(0) : 2 * sin(π / 4) * 2π / FT(86164.09) # 1/s
const u₀ = is_small_scale ? FT(20) : FT(0)                            # m/s
const v₀ = FT(0)                                                      # m/s
const w₀ = FT(0)                                                      # m/s
const T₀ = FT(250)                                                    # K
const ΔT = FT(0.01)                                                   # K
const p₀_surface = FT(1e5)                                            # kg/m/s^2
const grav = FT(9.80665)                                              # m/s^2
const R_d = FT(287.05)                                                # J/kg/K
const cp_d = FT(1005.0)                                               # J/kg/K
const cv_d = cp_d - R_d                                               # J/kg/K

include("baroclinic_wave_utilities.jl")

const xmax = is_small_scale ? FT(300e3) : FT(6000e3)
const zmax = FT(10e3)
Δx = is_small_scale ? FT(250) : FT(5000)
Δz = is_small_scale ? Δx / 2 : Δx / 40
npoly = 4
helem = Int(xmax / (Δx * (npoly + 1)))
velem = Int(zmax / Δz)

tmax = is_small_scale ? FT(60 * 60 * 0.5) : FT(60 * 60 * 8)
dt = is_small_scale ? FT(5) : FT(0.1)
save_every_n_steps = 1
ode_algorithm = OrdinaryDiffEq.SSPRK33

function local_geometry_fields(FT, zmax, velem, helem, npoly)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(FT)),
        Geometry.ZPoint{FT}(zmax);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

    hdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(zero(FT)),
        Geometry.XPoint{FT}(xmax);
        periodic = true,
    )
    hmesh = Meshes.IntervalMesh(hdomain; nelems = helem)
    htopology = Topologies.IntervalTopology(hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace1D(htopology, quad)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return (
        Fields.local_geometry_field(center_space),
        Fields.local_geometry_field(face_space),
    )
end

driver_values(FT) = (;
    zmax,
    velem,
    helem,
    npoly,
    tmax,
    dt,
    ode_algorithm,
    jacobian_flags = (; ∂𝔼ₜ∂𝕄_mode = :no_∂P∂K, ∂𝕄ₜ∂ρ_mode = :exact),
    max_newton_iters = 2,
    save_every_n_steps = 1,
    additional_solver_kwargs = (;), # e.g., reltol, abstol, etc.
)

function initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates

    # Continuous hydrostatic balance
    p = p₀_surface * exp(-grav * z / (R_d * T₀))

    # Discrete hydrostatic balance
    # Δz = FT(125) # non-constant global variable...
    # value = grav * Δz / (2 * R_d * T₀)
    # p = p₀_surface * ((1 - value)/(1 + value))^(z / Δz)

    T = T₀ +
        exp(grav * z / (2 * R_d * T₀)) * ΔT * exp(-(x - x_center)^2 / d^2) *
        sin(π * z / zmax)

    ρ = p / (R_d * T)
    # ρθ = ρ * T * (p₀_surface / p)^(R_d / cp_d)
    ρe = ρ * ((T - T_tri) * cv_d + (u₀^2 + v₀^2 + w₀^2) / 2 + grav * z)
    return (; ρ, ρe)
end
initial_condition_velocity(local_geometry) =
    Geometry.Covariant12Vector(Geometry.UVVector(u₀, v₀), local_geometry)

function baroclinic_wave_cache_values(Y, dt)
    FT = eltype(Y)
    center_space = axes(Y.Yc.ρ)
    face_space = axes(Y.w)
    center_local_geometry = Fields.local_geometry_field(center_space)
    cf = map(
        lg -> Geometry.Contravariant3Vector(Geometry.WVector(f), lg),
        center_local_geometry,
    ) # modified
    cχ_energy_named_tuple =
        :ρθ in propertynames(Y.Yc) ? (; cχθ = Fields.Field(FT, center_space)) :
        (; cχe = Fields.Field(FT, center_space))
    return (;
        cχ_energy_named_tuple...,
        cχuₕ = Fields.Field(Geometry.Covariant12Vector{FT}, center_space),
        cuvw = Fields.Field(Geometry.Covariant123Vector{FT}, center_space),
        cω³ = Fields.Field(Geometry.Contravariant3Vector{FT}, center_space),
        fω¹² = Fields.Field(Geometry.Contravariant12Vector{FT}, face_space),
        fu¹² = Fields.Field(Geometry.Contravariant12Vector{FT}, face_space),
        fu³ = Fields.Field(Geometry.Contravariant3Vector{FT}, face_space),
        cf,
        cK = Fields.Field(FT, center_space),
        cp = Fields.Field(FT, center_space),
        cΦ = grav .* center_local_geometry.coordinates.z,
    )
end

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = false),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρe_remaining_tendency!(dY, Y, p, t; κ₄ = 0.0e17)
    final_adjustments!(
        dY,
        Y,
        p,
        t;
        use_flux_correction = false,
        use_rayleigh_sponge = false,
    )
    return dY
end

function postprocessing(sol, path)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"
end
