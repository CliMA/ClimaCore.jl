push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore

using OrdinaryDiffEq: ODEProblem, solve, Euler
using DiffEqCallbacks: SavingCallback, SavedValues,
    CallbackSet, FunctionCallingCallback, TerminateSteadyState

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using CLIMAParameters: AbstractEarthParameterSet, Planet, astro_unit
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using NCDatasets
using Dierckx
using Dates
using Insolation
using CUDA
using Serialization

CUDA.allowscalar(false)

include("rrtmgp_model.jl")

#=
using Profile
using ChromeProfileFormat
macro prof(ex)
    return quote
        Profile.clear()
        Profile.@profile $(esc(ex))
        ChromeProfileFormat.save_cpuprofile("profile.cpuprofile")
    end
end

=#

precomputed_grad_values(x_face, ::Val{1}) =
    (; C = 1 ./ (x_face[2:end, :] .- x_face[1:end - 1, :]))
function precomputed_grad_values(x_face, ::Val{2})
    n = size(x_face, 1)
    edges = map(((1, 1), (n, -1))) do (face_index, step)
        @views x₁, x₂, x₃ =
            x_face[face_index, :], x_face[face_index + step, :],
            x_face[face_index + 2 * step, :]
        Δx₁₂, Δx₁₃ = x₂ .- x₁, x₃ .- x₁
        Δx₂₃ = x₃ .- x₂
        C₁ = @. -1 / Δx₁₂ - 1 / Δx₁₃
        C₂ = @.  Δx₁₃ / (Δx₁₂ * Δx₂₃)
        C₃ = @. -Δx₁₂ / (Δx₁₃ * Δx₂₃)
        (; C₁, C₂, C₃)
    end
    @views x₁, x₂, x₃ =
        x_face[1:n - 2, :], x_face[2:n - 1, :], x_face[3:n, :]
    Δx₁₂, Δx₁₃ = x₂ .- x₁, x₃ .- x₁
    Δx₂₃ = x₃ .- x₂
    C₁ = @. -Δx₂₃ / (Δx₁₂ * Δx₁₃)
    C₂ = @.  1 / Δx₁₂ - 1 / Δx₂₃
    C₃ = @.  Δx₁₂ / (Δx₁₃ * Δx₂₃)
    interior = (; C₁, C₂, C₃)
    (edges[1], interior, edges[2])
end
function precomputed_grad_values(x_face, ::Val{3})
    n = size(x_face, 1)
    map(
        ((1, 1), (2:n - 2, 1:n - 3), (n - 1, n - 3))
    ) do (center_range, face_range)
        @views x₁, x₂, x₃, x₄ =
            x_face[face_range, :], x_face[face_range .+ 1, :],
            x_face[face_range .+ 2, :], x_face[face_range .+ 3, :]
        x = (x_face[center_range, :] .+ x_face[center_range .+ 1, :]) / 2
        Δx₁, Δx₂, Δx₃, Δx₄ = x .- x₁, x .- x₂, x .- x₃, x .- x₄
        Δx₁₂, Δx₁₃, Δx₁₄ = x₂ .- x₁, x₃ .- x₁, x₄ .- x₁
        Δx₂₃, Δx₂₄ = x₃ .- x₂, x₄ .- x₂
        Δx₃₄ = x₄ .- x₃
        C₁ = @. -(Δx₂ * Δx₃ + Δx₂ * Δx₄ + Δx₃ * Δx₄) / (Δx₁₂ * Δx₁₃ * Δx₁₄)
        C₂ = @.  (Δx₁ * Δx₃ + Δx₁ * Δx₄ + Δx₃ * Δx₄) / (Δx₁₂ * Δx₂₃ * Δx₂₄)
        C₃ = @. -(Δx₁ * Δx₂ + Δx₁ * Δx₄ + Δx₂ * Δx₄) / (Δx₁₃ * Δx₂₃ * Δx₃₄)
        C₄ = @.  (Δx₁ * Δx₂ + Δx₁ * Δx₃ + Δx₂ * Δx₃) / (Δx₁₄ * Δx₂₄ * Δx₃₄)
        (; C₁, C₂, C₃, C₄)
    end
end
function precomputed_grad_values(x_face, ::Val{4})
    n = size(x_face, 1)
    edges = map(((1, 1), (n, -1))) do (face_index, step)
        @views x₁, x₂, x₃, x₄, x₅ =
            x_face[face_index, :], x_face[face_index + step, :],
            x_face[face_index + 2 * step, :], x_face[face_index + 3 * step, :],
            x_face[face_index + 4 * step, :]
        Δx₁₂, Δx₁₃, Δx₁₄, Δx₁₅ = x₂ .- x₁, x₃ .- x₁, x₄ .- x₁, x₅ .- x₁
        Δx₂₃, Δx₂₄, Δx₂₅ = x₃ .- x₂, x₄ .- x₂, x₅ .- x₂
        Δx₃₄, Δx₃₅ = x₄ .- x₃, x₅ .- x₃
        Δx₄₅ = x₅ .- x₄
        C₁ = @. -1 / Δx₁₂ - 1 / Δx₁₃ - 1 / Δx₁₄ - 1 / Δx₁₅
        C₂ = @.  Δx₁₃ * Δx₁₄ * Δx₁₅ / (Δx₁₂ * Δx₂₃ * Δx₂₄ * Δx₂₅)
        C₃ = @. -Δx₁₂ * Δx₁₄ * Δx₁₅ / (Δx₁₃ * Δx₂₃ * Δx₃₄ * Δx₃₅)
        C₄ = @.  Δx₁₂ * Δx₁₃ * Δx₁₅ / (Δx₁₄ * Δx₂₄ * Δx₃₄ * Δx₄₅)
        C₅ = @. -Δx₁₂ * Δx₁₃ * Δx₁₄ / (Δx₁₅ * Δx₂₅ * Δx₃₅ * Δx₄₅)
        actual_edge = (; C₁, C₂, C₃, C₄, C₅)
        C₁ = @. -Δx₂₃ * Δx₂₄ * Δx₂₅ / (Δx₁₂ * Δx₁₃ * Δx₁₄ * Δx₁₅)
        C₂ = @.  1 / Δx₁₂ - 1 / Δx₂₃ - 1 / Δx₂₄ - 1 / Δx₂₅
        C₃ = @.  Δx₁₂ * Δx₂₄ * Δx₂₅ / (Δx₁₃ * Δx₂₃ * Δx₃₄ * Δx₃₅)
        C₄ = @. -Δx₁₂ * Δx₂₃ * Δx₂₅ / (Δx₁₄ * Δx₂₄ * Δx₃₄ * Δx₄₅)
        C₅ = @.  Δx₁₂ * Δx₂₃ * Δx₂₄ / (Δx₁₅ * Δx₂₅ * Δx₃₅ * Δx₄₅)
        one_away_from_edge = (; C₁, C₂, C₃, C₄, C₅)
        [actual_edge, one_away_from_edge]
    end
    @views x₁, x₂, x₃, x₄, x₅ =
        x_face[1:n - 4, :], x_face[2:n - 3, :], x_face[3:n - 2, :],
        x_face[4:n - 1, :], x_face[5:n, :]
    Δx₁₂, Δx₁₃, Δx₁₄, Δx₁₅ = x₂ .- x₁, x₃ .- x₁, x₄ .- x₁, x₅ .- x₁
    Δx₂₃, Δx₂₄, Δx₂₅ = x₃ .- x₂, x₄ .- x₂, x₅ .- x₂
    Δx₃₄, Δx₃₅ = x₄ .- x₃, x₅ .- x₃
    Δx₄₅ = x₅ .- x₄
    C₁ = @.  Δx₂₃ * Δx₃₄ * Δx₃₅ / (Δx₁₂ * Δx₁₃ * Δx₁₄ * Δx₁₅)
    C₂ = @. -Δx₁₃ * Δx₃₄ * Δx₃₅ / (Δx₁₂ * Δx₂₃ * Δx₂₄ * Δx₂₅)
    C₃ = @.  1 / Δx₁₃ + 1 / Δx₂₃ - 1 / Δx₃₄ - 1 / Δx₃₅
    C₄ = @.  Δx₁₃ * Δx₂₃ * Δx₃₅ / (Δx₁₄ * Δx₂₄ * Δx₃₄ * Δx₄₅)
    C₅ = @. -Δx₁₃ * Δx₂₃ * Δx₃₄ / (Δx₁₅ * Δx₂₅ * Δx₃₅ * Δx₄₅)
    interior = (; C₁, C₂, C₃, C₄, C₅)
    return (edges[1][1], edges[1][2], interior, edges[2][2], edges[2][1])
end

grad!(∂f∂x_center, f_face, precomputed_values, ::Val{1}) =
    @views ∂f∂x_center .=
        (f_face[2:end, :] .- f_face[1:end - 1, :]) .* precomputed_values.C
function grad!(∂f∂x_face, f_face, precomputed_values, ::Val{2})
    n = size(∂f∂x_face, 1)
    for (index, (face_range, step, nsteps)) in enumerate(
        ((1, 1, 0), (1:n - 2, 1, 1), (n, -1, 0))
    )
        @views f₁, f₂, f₃ =
            f_face[face_range, :], f_face[face_range .+ step, :],
            f_face[face_range .+ 2 * step, :]
        @views ∂f∂x = ∂f∂x_face[face_range .+ nsteps * step, :]
        v = precomputed_values[index]
        @. ∂f∂x = f₁ * v.C₁ + f₂ * v.C₂ + f₃ * v.C₃
    end
end
function grad!(∂f∂x_center, f_face, precomputed_values, ::Val{3})
    n = size(∂f∂x_center, 1)
    for (index, (center_range, face_range)) in enumerate(
        ((1, 1), (2:n - 1, 1:n - 2), (n, n - 2))
    )
        @views f₁, f₂, f₃, f₄ =
            f_face[face_range, :], f_face[face_range .+ 1, :],
            f_face[face_range .+ 2, :], f_face[face_range .+ 3, :]
        @views ∂f∂x = ∂f∂x_center[center_range, :]
        v = precomputed_values[index]
        @. ∂f∂x = f₁ * v.C₁ + f₂ * v.C₂ + f₃ * v.C₃ + f₄ * v.C₄
    end
end
function grad!(∂f∂x_face, f_face, precomputed_values, ::Val{4})
    n = size(∂f∂x_face, 1)
    for (index, (face_range, step, nsteps)) in enumerate(
        ((1, 1, 0), (1, 1, 1), (1:n - 4, 1, 2), (n, -1, 1), (n, -1, 0))
    )
        @views f₁, f₂, f₃, f₄, f₅ =
            f_face[face_range, :], f_face[face_range .+ step, :],
            f_face[face_range .+ 2 * step, :], f_face[face_range .+ 3 * step, :],
            f_face[face_range .+ 4 * step, :]
        @views ∂f∂x = ∂f∂x_face[face_range .+ nsteps * step, :]
        v = precomputed_values[index]
        @. ∂f∂x = f₁ * v.C₁ + f₂ * v.C₂ + f₃ * v.C₃ + f₄ * v.C₄ + f₅ * v.C₅
    end
end

# TODO: remove allocations
function smooth!(dT, p, ::Val{:nearest_neighbor})
    FT = eltype(dT)
    dT[2:end - 1, :] .=
        (dT[1:end - 2, :] .+ FT(2) .* dT[2:end - 1, :] .+ dT[3:end, :]) ./ FT(4)
end
function exp_smooth!(dT, x, b)
    FT = eltype(dT)

    coefs = similar(x)
    numerator = similar(dT)
    numerator .= FT(0)
    denominator = similar(x)
    denominator .= FT(0)

    for i in 1:size(dT, 1)
        coefs .= exp.(-(x .- x[i, :]').^2 ./ (FT(2) * b^2))
        numerator .+= dT[i, :]' .* coefs
        denominator .+= coefs
    end

    dT .= numerator ./ denominator
end
smooth!(dT, p, ::Val{:exponential_index}) =
    exp_smooth!(
        dT,
        typeof(dT)(repeat(1:size(dT, 1), 1, size(dT, 2))),
        eltype(dT)(2),
    )
smooth!(dT, p, ::Val{:exponential_pressure}) =
    exp_smooth!(
        dT,
        p.implicit_faces ?
            p.rrtmgp_model.pressure : p.rrtmgp_model.level_pressure,
        eltype(dT)(5000),
    )
smooth!(dT, p, ::Val{:exponential_elevation}) =
    exp_smooth!(dT, p.implicit_faces ? p.zc : p.z, eltype(dT)(10000))

function heat_rate_flux!(dT, F, p)
    grad!(dT, F, p.precomputed_values, p.grad_order)
    dT .*= p.heating_rate_factor

    # smooth!(dT, p, Val(:exponential_index))
end
function heat_rate_diff!(dT, T, p)
    # FT = eltype(dT)
    # z = p.implicit_faces ? p.zc : p.z
    # P = p.implicit_faces ?
    #     p.rrtmgp_model.pressure : p.rrtmgp_model.level_pressure
    
    # p.θ .= T ./ p.Π
    # p.d2θdz2[2:end - 1, :] .=
    #     (p.θ[3:end, :] .- 2 .* p.θ[2:end - 1, :] .+ p.θ[1:end - 2, :]) ./
    #     (
    #         (z[3:end, :] .- z[2:end - 1, :]) .*
    #         (z[2:end - 1, :] .- z[1:end - 2, :])
    #     )
    # p.d2θdz2[1, :] .= p.d2θdz2[2, :]
    # # p.d2θdz2[end, :] .= FT(0)
    # # p.d2θdz2[21:end, :] .= FT(0) # findall(all(p.zc .> 3000, dims = 2))[1].I[1] == 21
    # dT .+= FT(0.1) .* p.d2θdz2 .* p.Π .* P ./ P[1, :]'
end

# For instantaneous_zenith_angle, errors in predicted angle and irradiance with
# respect to input data are
#     Avg: 1.4%, 0.05%
#     Max: 17%, 0.08%

function tendency!(dT, T, p, t)
    FT = eltype(T)

    # tuples = daily_zenith_angle.( # instantaneous_zenith_angle.(
    #     p.t0 .+ Dates.Second(round(Int, t)),
    #     # p.longitude,
    #     p.rrtmgp_model.latitude,
    #     param_set,
    # ) # each tuple contains (zenith angle, azimuthal angle, earth-sun distance)
    # zenith_angles = first.(tuples)
    # earth_sun_dists = last.(tuples)

    # p.rrtmgp_model.solar_zenith_angle .= min.(zenith_angles, FT(π)/2 - eps(FT))
    # p.rrtmgp_model.weighted_irradiance .=
    #     FT(Planet.tot_solar_irrad(param_set)) .*
    #     (FT(astro_unit()) ./ earth_sun_dists).^2

    # This is necessary because T is copied internally by the ODE solver.
    # T[1, :] .= p.rrtmgp_model.surface_temperature
    # T[1, :] .= T[2, :]
    if p.implicit_faces
        p.rrtmgp_model.temperature .= T
    else
        p.rrtmgp_model.level_temperature .= T
    end

    compute_fluxes!(p.rrtmgp_model)
    # p.rrtmgp_model.flux[1, :] .+= FT(1) .* (p.rrtmgp_model.surface_temperature .- parent(T)[1, :])
    heat_rate_flux!(dT, p.rrtmgp_model.flux, p)
    heat_rate_diff!(dT, T, p)
    return dT
end

function main(param_set, implicit_faces, grad_order)
    ds_input = rrtmgp_artifact("atmos_state", "clearsky_as.nc")
    FT = Float64 # eltype(ds_input["temp_layer"])

    nlay = ds_input.dim["layer"]
    nsite = ds_input.dim["site"]
    nexpt = ds_input.dim["expt"]
    ncol = nsite * nexpt
    function get_var(string)
        var = ds_input[string]
        arr = Array(var)
        if ndims(arr) == 3
            return reshape(arr, size(arr, 1), ncol)[end:-1:1, :]
        elseif ndims(arr) == 2
            if "site" in dimnames(var) && "expt" in dimnames(var)
                return collect(reshape(arr, ncol))
            elseif "site" in dimnames(var)
                return repeat(arr; outer = (1, nexpt))[end:-1:1, :]
            else # "expt" in dimnames(var)
                return repeat(arr; inner = (1, nsite))[end:-1:1, :]
            end
        else # ndims(arr) == 1
            if "site" in dimnames(var)
                return repeat(arr; outer = nexpt)
            else # "expt" in dimnames(var)
                return repeat(arr; inner = nsite)
            end
        end
    end

    vmrs = map((
        # ("h2o", "water_vapor"),            # overwritten by vmr_h2o
        ("co2", "carbon_dioxide_GM"),
        # ("o3", "ozone"),                   # overwritten by vmr_o3
        ("n2o", "nitrous_oxide_GM"),
        ("co", "carbon_monoxide_GM"),
        ("ch4", "methane_GM"),
        ("o2", "oxygen_GM"),
        ("n2", "nitrogen_GM"),
        ("ccl4", "carbon_tetrachloride_GM"),
        ("cfc11", "cfc11_GM"),
        ("cfc12", "cfc12_GM"),
        ("cfc22", "hcfc22_GM"),
        ("hfc143a", "hfc143a_GM"),
        ("hfc125", "hfc125_GM"),
        ("hfc23", "hfc23_GM"),
        ("hfc32", "hfc32_GM"),
        ("hfc134a", "hfc134a_GM"),
        ("cf4", "cf4_GM"),
        # ("no2", nothing),                  # not available in input dataset
    )) do (lookup_gas_name, input_gas_name)
        (
            Symbol("volume_mixing_ratio_" * lookup_gas_name),
            get_var(input_gas_name)' .*
                parse(FT, ds_input[input_gas_name].attrib["units"]),
        )
    end

    volume_mixing_ratio_h2o = get_var("water_vapor")
    pressure = get_var("pres_layer")
    temperature = get_var("temp_layer")
    face_pressure = get_var("pres_level")
    mass_mixing_ratio_h2o =
        volume_mixing_ratio_h2o ./ FT(Planet.molmass_ratio(param_set))
    specific_humidity = mass_mixing_ratio_h2o ./ (1 .+ mass_mixing_ratio_h2o)
    constant = FT(Planet.R_v(param_set)) / FT(Planet.R_d(param_set)) - 1
    specific_volume_over_temperature =
        FT(Planet.R_d(param_set)) .* (1 .+ constant .* specific_humidity) ./
        pressure
    specific_volume = specific_volume_over_temperature .* temperature
    integrals = map(1:ncol) do icol
        @views spline =
            Spline1D(pressure[end:-1:1, icol], specific_volume[end:-1:1, icol])
        map(face_pressure[:, icol]) do p
            FT(integrate(spline, p, face_pressure[1, icol]))
        end
    end
    z = hcat(integrals...) ./ FT(Planet.grav(param_set))

    if implicit_faces
        p_and_t = (; pressure = pressure, temperature = temperature)
    else
        p_and_t = (;
            level_pressure = face_pressure,
            level_temperature = get_var("temp_level"),
        )
    end

    # rrtmgp_model = RRTMGPModel(
    #     param_set,
    #     z;
    #     level_computation = :average,
    #     use_ideal_coefs_for_bottom_level = false,
    #     add_isothermal_boundary_layer = false,
    #     surface_emissivity = get_var("surface_emissivity")',
    #     solar_zenith_angle = NaN,
    #     weighted_irradiance = NaN,
    #     dir_sw_surface_albedo = get_var("surface_albedo")',
    #     dif_sw_surface_albedo = get_var("surface_albedo")',
    #     p_and_t...,
    #     surface_temperature = get_var("surface_temperature"),
    #     latitude = get_var("lat"),
    #     volume_mixing_ratio_h2o = volume_mixing_ratio_h2o,
    #     volume_mixing_ratio_o3 = get_var("ozone"),
    #     vmrs...,
    #     volume_mixing_ratio_no2 = 0,
    # )

    rrtmgp_model = RRTMGPModel(
        param_set,
        z;
        optics = :gray,
        level_computation = :average,
        use_ideal_coefs_for_bottom_level = false,
        add_isothermal_boundary_layer = false,
        surface_emissivity = collect(get_var("surface_emissivity")'),
        solar_zenith_angle = NaN,
        weighted_irradiance = NaN,
        dir_sw_surface_albedo = collect(get_var("surface_albedo")'),
        dif_sw_surface_albedo = collect(get_var("surface_albedo")'),
        p_and_t...,
        surface_temperature = get_var("surface_temperature"),
        optical_thickness_parameter =
            ((300 .+ 60 .* (1 / 3 .- sin.(get_var("lat")).^2)) ./ 200).^4 .- 1,
        lapse_rate = 3.5,
    )

    T = implicit_faces ?
        rrtmgp_model.temperature : rrtmgp_model.level_temperature
    P = implicit_faces ? rrtmgp_model.pressure : rrtmgp_model.level_pressure
    F = rrtmgp_model.flux
    DA = typeof(parent(T)).name.wrapper
    z = DA(z)

    if implicit_faces
        q = DA(specific_humidity)
    else
        q = Array{FT}(undef, nlay + 1, ncol)
        q[end, :] .= 0
        for i in nlay:-1:1
            q[i, :] .= 2 .* specific_humidity[i, :] - q[i + 1, :]
        end
        q = DA(q)
    end

    specific_heat_capacity =
        (1 .- q) .* FT(Planet.cp_d(param_set)) .+
        q .* FT(Planet.cp_v(param_set))
    # heating_rate_factor =
    #     specific_volume_over_temperature ./ specific_heat_capacity
    heating_rate_factor = FT(Planet.grav(param_set)) ./ specific_heat_capacity

    gas_constant =
        (1 .- q) .* FT(Planet.R_d(param_set)) .+ q .* FT(Planet.R_v(param_set))
    Π = (P ./ FT(1e5)).^(gas_constant ./ specific_heat_capacity)
    θ = similar(T)
    d2θdz2 = similar(T)

    t0 = DA(get_var("time"))
    longitude = DA(get_var("lon"))
    zc = (z[1:end - 1, :] .+ z[2:end, :]) ./ 2

    compute_fluxes!(rrtmgp_model)
    precomputed_values = precomputed_grad_values(
        rrtmgp_model.solver.as.p_lev[1:nlay + 1, :],
        grad_order,
    )

    rrtmgp_model.solar_zenith_angle .= 0
    rrtmgp_model.weighted_irradiance .= 0
    for nday in 1:365
        tuples = daily_zenith_angle.(
            t0 .+ Dates.Second(60 * 60 * 24 * nday),
            DA(get_var("lat")), # rrtmgp_model.latitude,
            param_set,
        )
        zenith_angles = first.(tuples)
        earth_sun_dists = last.(tuples)
        rrtmgp_model.solar_zenith_angle .+= min.(zenith_angles, FT(π)/2 - eps(FT))
        rrtmgp_model.weighted_irradiance .+=
            FT(Planet.tot_solar_irrad(param_set)) .*
            (FT(astro_unit()) ./ earth_sun_dists).^2
    end
    rrtmgp_model.solar_zenith_angle ./= 365
    rrtmgp_model.weighted_irradiance ./= 365

    p = (; implicit_faces, grad_order, rrtmgp_model, heating_rate_factor, z, t0, longitude, precomputed_values, θ, d2θdz2, zc, Π)

    close(ds_input)

    # Compute the tendency at t = 0 for debugging. This is necessary because the
    # ode solver apparently calls the function callback before computing the
    # tendency at t = 0, but after computing the tendency for all other times.
    tendency!(similar(T), T, p, 0.)
    
    Δt = 12 * 60. * 60.
    N = 1
    N′ = 100 * 365 * 2
    sim(nrow) = [Array{FT}(undef, nrow, ncol) for _ in 1:N + 1]
    values =
        (sim(nlay), sim(nlay + 1), sim(nlay + 1), sim(nlay + 1), sim(nlay + 1))
    n = 0
    sol = solve(
        ODEProblem(tendency!, T, (0., N′ * Δt), p),
        Euler();
        dt = Δt,
        callback = FunctionCallingCallback(
            (u, t, integrator) -> (
                n += 1;
                n = min(n, 2);
                @inbounds begin
                    copyto!(
                        values[1][n],
                        view(rrtmgp_model.solver.as.t_lay, 1:nlay, :),
                    );
                    copyto!(values[2][n], rrtmgp_model.up_lw_flux);
                    copyto!(values[3][n], rrtmgp_model.dn_lw_flux);
                    copyto!(values[4][n], rrtmgp_model.up_sw_flux);
                    copyto!(values[5][n], rrtmgp_model.dn_sw_flux);
                end
            ),
        ),
        save_everystep = false,
        progress = true,
        progress_steps = 1,
        progress_message = (dt, u, p, t) -> "",
        # unstable_check = (dt, u, p, t) -> false,
    )

    # return p, map(v -> v[1:n], values), 0:Δt:(n - 1) * Δt, nsite, nexpt
    return p, values, [0, N′ * Δt], nsite, nexpt
end

p, values, times, nsite, nexpt = main(param_set, false, Val(2))
serialize(
    joinpath(
        "/central/scratch/dyatunin/",
        "result_gray.bin", # "result_AverageCenters_AnnualInsolation_CappedPressure_SurfaceFlux1_2hr.bin",
    ),
    (p, values, times),
)

nsite = 100
nexpt = 18
# p_ref, values_ref, times_ref = deserialize(joinpath(
#     "/central/scratch/dyatunin/",
#     "result_AverageCenters_AnnualInsolation_CappedPressure_SurfaceFlux1_2hr.bin",
# ))
comparison_name = "gray" # "BestFitCenters_AnnualInsolation_CappedPressure_SurfaceFlux1_2hr"
p, values, times = deserialize(
    joinpath("/central/scratch/dyatunin/", "result_$(comparison_name).bin")
)

# if !hasproperty(p_ref, :grad_order)
#     p_ref = (; p_ref..., implicit_faces = true, grad_order = Val(1))
# end
if !hasproperty(p, :grad_order)
    p = (; p..., implicit_faces = true, grad_order = Val(1))
end

ENV["GKSwstype"] = "nul"
using Plots
Plots.GRBackend()

function plot_values(values, n, p)
    DA = typeof(parent(p.z)).name.wrapper
    temp, flux_lw_up, flux_lw_dn, flux_sw_up, flux_sw_dn =
        map(i -> DA(values[i][n]), 1:length(values))
    flux_lw = flux_lw_up .- flux_lw_dn
    flux_sw = flux_sw_up .- flux_sw_dn
    flux = flux_lw .+ flux_sw

    hr_template = p.implicit_faces ? temp : flux
    hr_symb = p.implicit_faces ? :lay : :lev
    heat_rate_lw = similar(hr_template)
    heat_rate_sw = similar(hr_template)
    heat_rate_diff = similar(hr_template)
    heat_rate_diff .= 0
    heat_rate_flux!(heat_rate_lw, flux_lw, p)
    heat_rate_flux!(heat_rate_sw, flux_sw, p)
    # heat_rate_diff!(heat_rate_diff, temp, p)
    return (
        (temp, "Temperature", " [K]", "temp", :lay),
        (flux_lw_up, "Upward Longwave Flux", " [W/m^2]", "flux_lw_up", :lev),
        (flux_lw_dn, "Downward Longwave Flux", " [W/m^2]", "flux_lw_dn", :lev),
        (flux_sw_up, "Upward Shortwave Flux", " [W/m^2]", "flux_sw_up", :lev),
        (flux_sw_dn, "Downward Shortwave Flux", " [W/m^2]", "flux_sw_dn", :lev),
        (flux, "Net Flux", " [W/m^2]", "flux_net", :lev),
        (heat_rate_lw, "Longwave Heating Rate", " [K/s]", "heat_rate_lw", hr_symb),
        (heat_rate_sw, "Shortwave Heating Rate", " [K/s]", "heat_rate_sw", hr_symb),
        # (heat_rate_diff, "Diffusion Heating Rate", " [K/s]", "heat_rate_dif", hr_symb),
    )
end

path = joinpath(@__DIR__, "comparison3_$(comparison_name)")
mkpath(path)

N = length(times)
zkm = (; lev = Array(p.z ./ 1000), lay = Array(p.zc ./ 1000))
function to_color(isite, iexpt)
    expt_value = 0.3 + (iexpt - 1) * 0.4 / (nexpt - 1)
    return HSI((isite - 1) * 360 / nsite, expt_value, expt_value)
end
palette = reshape(
    map(tup -> to_color(tup...), Iterators.product(1:nsite, 1:nexpt)),
    nsite * nexpt,
)
for (n, filename_suffix) in ((1, "initial"), (N, "final")) # ((1, "initial"), (N ÷ 2, "middle"), (N, "final"))
    vals = plot_values(values, n, p)
    # ref_vals = plot_values(
    #     values_ref,
    #     findfirst(isequal(times[n]), times_ref),
    #     p_ref,
    # )
    tday = round.(times[n] ./ (60. * 60. * 24.), digits = 3)
    for (value, name, units, filename_prefix, coord_symb) in vals
        Plots.png(
            Plots.plot(
                Array(value),
                getproperty(zkm, coord_symb);
                title = "t = $tday days",
                xguide = "$name$units",
                yguide = "Elevation [km]",
                legend = false,
                color_palette = palette,
            ),
            joinpath(path, "$(filename_prefix)_$(filename_suffix).png"),
        )
    end
    # for (
    #     (value, name, units, filename_prefix, coord_symb),
    #     (ref_value, _, _, _)
    # ) in zip(vals, ref_vals)
    #     if !(name in ("Temperature", "Net Flux"))
    #         continue
    #     end
    #     Plots.png(
    #         Plots.plot(
    #             Array(value .- ref_value[1:size(value, 1), :]),
    #             getproperty(zkm, coord_symb);
    #             title = "t = $tday days",
    #             xguide = "Absolute Error in $name$units",
    #             yguide = "Elevation [km]",
    #             legend = false,
    #             color_palette = palette,
    #         ),
    #         joinpath(path, "abs_err_$(filename_prefix)_$(filename_suffix).png"),
    #     )
    # end
    # for (
    #     (value, name, units, filename_prefix, coord_symb),
    #     (ref_value, _, _, _)
    # ) in zip(vals, ref_vals)
    #     if name in ("Net Flux", "Longwave Heating Rate", "Shortwave Heating Rate")
    #         continue
    #     end
    #     ref_value = ref_value[1:size(value, 1), :]
    #     error = (value .- ref_value) ./ abs.(ref_value) .* 100
    #     modified_error = sign.(error) .* (log10.(abs.(error) .+ 1e-5) .+ 5)
    #     Plots.png(
    #         Plots.plot(
    #             Array(modified_error),
    #             getproperty(zkm, coord_symb);
    #             title = "t = $tday days",
    #             xguide = "Relative Error in $name [%]",
    #             yguide = "Elevation [km]",
    #             xformatter = x ->
    #                 string(
    #                     round(sign(x) * (10^(abs(x) - 5) - 1e-5); sigdigits = 2)
    #                 ),
    #             legend = false,
    #             color_palette = palette,
    #         ),
    #         joinpath(path, "rel_err_$(filename_prefix)_$(filename_suffix).png"),
    #     )
    # end
end

# anim = Plots.@animate for u in us
#     Plots.plot(u, zkm; plotargs...)
# end
# Plots.mp4(anim, joinpath(path, "radiation.mp4"), fps = 10)

# sum_dists = similar(p.rrtmgp_model.weighted_irradiance);
# sum_fluxes = similar(p.rrtmgp_model.solar_zenith_angle);
# sum_dists .= 0;
# sum_fluxes .= 0;
# for nday in 1:365
#     tuples = daily_zenith_angle.(
#         p.t0 .+ Dates.Second(60 * 60 * 24 * nday),
#         p.rrtmgp_model.latitude,
#         param_set,
#     )
#     zenith_angles = first.(tuples)
#     earth_sun_dists = last.(tuples)
#     sum_dists .+= earth_sun_dists
#     sum_fluxes .+=
#         cos.(zenith_angles) .*
#         Planet.tot_solar_irrad(param_set) .*
#         (astro_unit() ./ earth_sun_dists).^2
#     solar_zenith_angle .+= zenith_angles
#     weighted_irradiance .+=
#         Planet.tot_solar_irrad(param_set) .*
#         (astro_unit() ./ earth_sun_dists).^2
# end
# weighted_irradiance .=
#     Planet.tot_solar_irrad(param_set) .*
#     (astro_unit() ./ (sum_dists ./ 365)).^2;
# solar_zenith_angle = acos.((sum_fluxes ./ 365) ./ weighted_irradiance);


# filtering may be preventing us from reaching general conclusions about things like interpolation and boundary layers
# different ODE solvers, smaller timesteps, and diffusion do not remove the need for filtering
# "realistic" filtering (pressure- or elevation-based) is worse than index-based filtering because the top layers are too thick
# using an unfixed bottom face temperature is better with filtering (equilibrium flux profile is smoother near surface)
# it is ok if top face pressure goes bellow the minimum value due to extrapolation (even if it goes negative), as long as there is no boundary layer

# different face value interpolations/extrapolations seem to be roughly identical, except for :ideal_coefs and :best fit:
#     :ideal_coefs diverges in column 1601 after 7 days
#     :best_fit diverges at the surface if temperature is not fixed

# time varying insolation failed - seems like there is a minimum temperature in RRTMGP look up table?
# our default setup is:
#     exponential filter (index-based)
#     cell faces are interpolated between cell centers using arithmetic means
#     bottom/top cell faces are extrapolated using cell_face_1 = (3 * cell_center_1 - cell_center_2) / 2
#         this may cause the pressure to go negative/below the minimum at the top face, which is fine due to how the lookup table is being accessed
#         the temperature at the bottom face does not need to be set to the surface temperature, as long as we are using filtering
#     when the top cell center is below 50 km or so, we use the isothermal boundary layer


# TODO:
#     Explore surface fluxes with non-staggered grid
#     Experiment with extension profiles
#     Compare to the Frankenberg lab's radiation model