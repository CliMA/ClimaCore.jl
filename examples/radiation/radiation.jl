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

precomputed_grad_values(x_face, x_center, ::Val{1}) =
    @views (; C = 1 ./ (x_face[2:end, :] .- x_face[1:end - 1, :]))
function precomputed_grad_values(x_face, x_center, ::Val{3})
    n = size(x_center, 1)
    map(((1, 1), (2:n - 1, 1:n - 2), (n, n - 2))) do (center_range, face_range)
        @views x₁, x₂, x₃, x₄ =
            x_face[face_range, :], x_face[face_range .+ 1, :],
            x_face[face_range .+ 2, :], x_face[face_range .+ 3, :]
        @views x = x_center[center_range, :]
        Δx₁, Δx₂, Δx₃, Δx₄ = x .- x₁, x .- x₂, x .- x₃, x .- x₄
        Δx₁₂, Δx₁₃, Δx₁₄ = x₂ .- x₁, x₃ .- x₁, x₄ .- x₁
        Δx₂₃, Δx₂₄ = x₃ .- x₂, x₄ .- x₂
        Δx₃₄ = x₄ .- x₃
        C₁ = @. -(Δx₂ * Δx₃ + Δx₂ * Δx₄ + Δx₃ * Δx₄) / (Δx₁₂ * Δx₁₃ * Δx₁₄)
        C₂ = @. (Δx₁ * Δx₃ + Δx₁ * Δx₄ + Δx₃ * Δx₄) / (Δx₁₂ * Δx₂₃ * Δx₂₄)
        C₃ = @. -(Δx₁ * Δx₂ + Δx₁ * Δx₄ + Δx₂ * Δx₄) / (Δx₁₃ * Δx₂₃ * Δx₃₄)
        C₄ = @. (Δx₁ * Δx₂ + Δx₁ * Δx₃ + Δx₂ * Δx₃) / (Δx₁₄ * Δx₂₄ * Δx₃₄)
        (; C₁, C₂, C₃, C₄)
    end
end
grad!(∂f∂x_face, f_face, precomputed_values, ::Val{1}) =
    @views ∂f∂x_face .=
        (f_face[2:end, :] .- f_face[1:end - 1, :]) .* precomputed_values.C
function grad!(∂f∂x_face, f_face, precomputed_values, ::Val{3})
    n = size(∂f∂x_face, 1)
    for (index, (center_range, face_range)) in enumerate(
        ((1, 1), (2:n - 1, 1:n - 2), (n, n - 2))
    )
        @views f₁, f₂, f₃, f₄ =
            f_face[face_range, :], f_face[face_range .+ 1, :],
            f_face[face_range .+ 2, :], f_face[face_range .+ 3, :]
        @views ∂f∂x = ∂f∂x_face[center_range, :]
        pv = precomputed_values[index]
        ∂f∂x .= f₁ .* pv.C₁ .+ f₂ .* pv.C₂ .+ f₃ .* pv.C₃ .+ f₄ .* pv.C₄
    end
    return ∂f∂x_face
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
        xi = ndims(x) == 1 ? (CUDA.@allowscalar x[i]) : x[i, :]'
        coefs .= exp.(-(x .- xi).^2 ./ (FT(2) * b^2))
        numerator .+= dT[i, :]' .* coefs
        denominator .+= coefs
    end

    dT .= numerator ./ denominator
end
smooth!(dT, p, ::Val{:exponential_index}) =
    exp_smooth!(dT, (typeof(dT).name.wrapper)(1:size(dT, 1)), eltype(dT)(2))
smooth!(dT, p, ::Val{:exponential_pressure}) =
    exp_smooth!(dT, p.rrtmgp_model.pressure, eltype(dT)(5000))
smooth!(dT, p, ::Val{:exponential_elevation}) =
    exp_smooth!(dT, p.zc, eltype(dT)(10000))

function heat_rate_flux!(dT, F, p)
    grad!(dT, F, p.precomputed_values, Val(1))
    dT .*= p.heating_rate_factor

    smooth!(dT, p, Val(:exponential_index))
end
function heat_rate_diff!(dT, T, p)
    # FT = eltype(dT)
    
    # p.θ .= T ./ p.Π
    # p.d2θdz2[2:end - 1, :] .=
    #     (p.θ[3:end, :] .- 2 .* p.θ[2:end - 1, :] .+ p.θ[1:end - 2, :]) ./
    #     (
    #         (p.zc[3:end, :] .- p.zc[2:end - 1, :]) .*
    #         (p.zc[2:end - 1, :] .- p.zc[1:end - 2, :])
    #     )
    # p.d2θdz2[1, :] .= FT(0)
    # # p.d2θdz2[end, :] .= FT(0)
    # p.d2θdz2[21:end, :] .= FT(0) # findall(all(p.zc .> 3000, dims = 2))[1].I[1] == 21
    # dT .+= FT(0.005) .* p.d2θdz2 .* p.Π .*
    #     p.rrtmgp_model.pressure ./ p.rrtmgp_model.pressure[1, :]'
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
    p.rrtmgp_model.temperature .= T

    compute_fluxes!(p.rrtmgp_model)

    # dT .= -T .* p.heating_rate_factor .*
    #     (p.rrtmgp_model.flux[2:end, :] .- p.rrtmgp_model.flux[1:end - 1, :]) ./
    #     (p.z[2:end, :] .- p.z[1:end - 1, :])
    # dT .= p.heating_rate_factor .*
    #     (p.rrtmgp_model.flux[2:end, :] .- p.rrtmgp_model.flux[1:end - 1, :]) ./
    #     (p.rrtmgp_model.solver.as.p_lev[2:end, :] .- p.rrtmgp_model.solver.as.p_lev[1:end - 1, :])

    heat_rate_flux!(dT, p.rrtmgp_model.flux, p)
    heat_rate_diff!(dT, T, p)
    return dT
end

function main(param_set)
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

    # nlay = ds_input.dim["layer"]
    # nsite = ds_input.dim["site"]
    # nexpt = 2 # ds_input.dim["expt"]
    # ncol = nsite * nexpt
    # function get_var2(string)
    #     var = ds_input[string]
    #     arr = Array(var)
    #     if ndims(arr) == 3
    #         return reshape(arr[:, :, 1:nexpt], size(arr, 1), ncol)[end:-1:1, :]
    #     elseif ndims(arr) == 2
    #         if "site" in dimnames(var) && "expt" in dimnames(var)
    #             return collect(reshape(arr[:, 1:nexpt], ncol))
    #         elseif "site" in dimnames(var)
    #             return repeat(arr; outer = (1, nexpt))[end:-1:1, :]
    #         else # "expt" in dimnames(var)
    #             return repeat(arr[:, 1:nexpt]; inner = (1, nsite))[end:-1:1, :]
    #         end
    #     else # ndims(arr) == 1
    #         if "site" in dimnames(var)
    #             return repeat(arr; outer = nexpt)
    #         else # "expt" in dimnames(var)
    #             return repeat(arr[1:nexpt]; inner = nsite)
    #         end
    #     end
    # end

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

    rrtmgp_model = RRTMGPModel(
        param_set,
        FT,
        nlay,
        ncol;
        level_computation = :average,
        use_ideal_coefs_for_bottom_level = false,
        add_isothermal_boundary_layer = false,
        surface_emissivity = get_var("surface_emissivity")',
        solar_zenith_angle = NaN,
        weighted_irradiance = NaN,
        dir_sw_surface_albedo = get_var("surface_albedo")',
        dif_sw_surface_albedo = get_var("surface_albedo")',
        pressure = get_var("pres_layer"),
        temperature = get_var("temp_layer"),
        surface_temperature = get_var("surface_temperature"),
        latitude = get_var("lat"),
        volume_mixing_ratio_h2o = get_var("water_vapor"),
        volume_mixing_ratio_o3 = get_var("ozone"),
        vmrs...,
        volume_mixing_ratio_no2 = 0,
    )
    compute_fluxes!(rrtmgp_model) # For debugging; initializes p_lev.

    T = rrtmgp_model.temperature
    pressure = rrtmgp_model.pressure
    volume_mixing_ratio_h2o = rrtmgp_model.volume_mixing_ratio_h2o
    face_pressure = Array(rrtmgp_model.solver.as.p_lev[1:nlay + 1, :])

    DA = typeof(parent(T)).name.wrapper

    mass_mixing_ratio_h2o =
        volume_mixing_ratio_h2o ./ FT(Planet.molmass_ratio(param_set))
    specific_humidity = mass_mixing_ratio_h2o ./ (1 .+ mass_mixing_ratio_h2o)
    constant = FT(Planet.R_v(param_set)) / FT(Planet.R_d(param_set)) - 1
    specific_volume_over_temperature =
        FT(Planet.R_d(param_set)) .* (1 .+ constant .* specific_humidity) ./
        pressure
    specific_volume = specific_volume_over_temperature .* T
    integrals = map(1:ncol) do icol
        @views spline =
            Spline1D(pressure[end:-1:1, icol], specific_volume[end:-1:1, icol])
        map(face_pressure[:, icol]) do p
            FT(integrate(spline, p, face_pressure[1, icol]))
        end
    end
    z = DA(hcat(integrals...)) ./ FT(Planet.grav(param_set))

    specific_heat_capacity =
        (1 .- specific_humidity) .* FT(Planet.cp_d(param_set)) .+
        specific_humidity .* FT(Planet.cp_v(param_set))
    # heating_rate_factor =
    #     DA(specific_volume_over_temperature ./ specific_heat_capacity)
    heating_rate_factor = FT(Planet.grav(param_set)) ./ specific_heat_capacity
    t0 = DA(get_var("time"))
    longitude = DA(get_var("lon"))
    precomputed_values = precomputed_grad_values(
        rrtmgp_model.solver.as.p_lev[1:nlay + 1, :],
        rrtmgp_model.pressure,
        Val(1),
    )
    gas_constant =
        (1 .- specific_humidity) .* FT(Planet.R_d(param_set)) .+
        specific_humidity .* FT(Planet.R_v(param_set))
    Π = (rrtmgp_model.pressure ./ FT(1e5)).^(gas_constant ./ specific_heat_capacity)
    θ = similar(T)
    d2θdz2 = similar(T)
    zc = (z[1:end - 1, :] .+ z[2:end, :]) ./ 2

    rrtmgp_model.solar_zenith_angle .= 0
    rrtmgp_model.weighted_irradiance .= 0
    for nday in 1:365
        tuples = daily_zenith_angle.(
            t0 .+ Dates.Second(60 * 60 * 24 * nday),
            rrtmgp_model.latitude,
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

    p = (; rrtmgp_model, heating_rate_factor, z, t0, longitude, precomputed_values, θ, d2θdz2, zc, Π)

    close(ds_input)

    # Compute the tendency at t = 0 for debugging. This is necessary because the
    # ode solver apparently calls the function callback before computing the
    # tendency at t = 0, but after computing the tendency for all other times.
    tendency!(similar(T), T, p, 0.)
    
    Δt = 12. * 60. * 60.
    N = 2 * 365 * 2
    F = rrtmgp_model.flux
    simN(array) = [Array{eltype(array)}(undef, size(array)...) for _ in 1:N + 1]
    values = (simN(T), simN(F), simN(F), simN(F), simN(F))
    n = 0
    sol = solve(
        ODEProblem(tendency!, T, (0., N * Δt), p),
        Euler();
        dt = Δt,
        callback = FunctionCallingCallback(
            (u, t, integrator) -> (
                n += 1;
                @inbounds begin
                    copyto!(values[1][n], rrtmgp_model.temperature);
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

    return p, map(v -> v[1:n], values), 0:Δt:(n - 1) * Δt, nsite, nexpt
end

p, values, times, nsite, nexpt = main(param_set)

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
    heat_rate_lw = similar(temp)
    heat_rate_sw = similar(temp)
    heat_rate_diff = similar(temp)
    heat_rate_diff .= 0
    heat_rate_flux!(heat_rate_lw, flux_lw, p)
    heat_rate_flux!(heat_rate_sw, flux_sw, p)
    heat_rate_diff!(heat_rate_diff, temp, p)
    return (
        (temp, "Temperature [K]", "temp", :lay),
        (flux_lw_up, "Upward Longwave Flux [W/m^2]", "flux_lw_up", :lev),
        (flux_lw_dn, "Downward Longwave Flux [W/m^2]", "flux_lw_dn", :lev),
        (flux_sw_up, "Upward Shortwave Flux [W/m^2]", "flux_sw_up", :lev),
        (flux_sw_dn, "Downward Shortwave Flux [W/m^2]", "flux_sw_dn", :lev),
        (flux, "Net Flux [W/m^2]", "flux", :lev),
        (heat_rate_lw, "Longwave Heating Rate [K/s]", "heat_rate_lw", :lay),
        (heat_rate_sw, "Shortwave Heating Rate [K/s]", "heat_rate_sw", :lay),
        (heat_rate_diff, "Diffusion Heating Rate [K/s]", "heat_rate_dif", :lay),
    )
end

path = joinpath(@__DIR__, "output_expfilter2_annual_free_bottom")
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
for (n, filename_suffix) in ((1, "begin"), (N ÷ 2, "middle"), (N, "end"))
    tday = round.(times[n] ./ (60. * 60. * 24.), digits = 3)
    for (value, name, filename_prefix, coord_symb) in plot_values(values, n, p)
        Plots.png(
            Plots.plot(
                Array(value),
                getproperty(zkm, coord_symb);
                title = "t = $tday days",
                xguide = name,
                yguide = "Elevation [km]",
                legend = false,
                color_palette = palette,
            ),
            joinpath(path, "$(filename_prefix)_$(filename_suffix).png"),
        )
    end
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