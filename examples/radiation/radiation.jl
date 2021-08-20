push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33, Euler

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

include("rrtmgp_model.jl")

function do_stuff(param_set)
    ds_input = rrtmgp_artifact("atmos_state", "clearsky_as.nc")
    FT = eltype(ds_input["temp_layer"])
    nsite = size(ds_input["temp_layer"], 2)
    nexpt = size(ds_input["temp_layer"], 3)
    ncol = nsite * nexpt

    function get_var(string)
        var = ds_input[string]
        if ndims(var) == 3
            return reshape(var, size(var, 1), ncol)[end:-1:1, :]
        elseif ndims(var) == 2
            if "site" in dimnames(var) && "expt" in dimnames(var)
                return collect(reshape(var, ncol))
            elseif "site" in dimnames(var)
                return repeat(var; outer = (1, nexpt))[end:-1:1, :]
            else # "expt" in dimnames(var)
                return repeat(var; inner = (1, nsite))[end:-1:1, :]
            end
        else # ndims(var) == 1
            if "site" in dimnames(var)
                return repeat(var; outer = nexpt)
            else # "expt" in dimnames(var)
                return repeat(var; inner = nsite)
            end
        end
    end

    pressure = get_var("pres_layer")
    face_pressure = get_var("pres_level")
    temperature = get_var("temp_layer")
    volume_mixing_ratio_h2o = get_var("water_vapor")

    mass_mixing_ratio_h2o =
        volume_mixing_ratio_h2o ./ FT(Planet.molmass_ratio(param_set))
    specific_humidity = mass_mixing_ratio_h2o ./ (1 .+ mass_mixing_ratio_h2o)
    constant = FT(Planet.R_v(param_set)) / FT(Planet.R_d(param_set)) - 1
    specific_volume_over_temperature =
        FT(Planet.R_d(param_set)) .* (1 .+ constant .* specific_humidity) ./ pressure
    specific_volume = specific_volume_over_temperature .* temperature
    integrals = map(1:ncol) do icol
        spline = Spline1D(pressure[end:-1:1, icol], specific_volume[end:-1:1, icol])
        map(face_pressure[:, icol]) do p
            FT(integrate(spline, p, face_pressure[1, icol]))
        end
    end
    face_elevations =
        sum(integrals) ./ (length(integrals) * FT(Planet.grav(param_set)))

    domain = ClimaCore.Domains.IntervalDomain(
        face_elevations[1],
        face_elevations[end];
        x3boundary = (:bottom, :top),
    )
    mesh = ClimaCore.Meshes.IntervalMesh{eltype(domain)}(
        domain,
        face_elevations,
        NamedTuple{domain.x3boundary}((5, 6)),
    )
    space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(mesh)

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
        space,
        ncol;
        level_computation = :average,
        use_ideal_coefs_for_bottom_level = false,
        add_isothermal_boundary_layer = false,
        surface_emissivity = get_var("surface_emissivity")',
        solar_zenith_angle = NaN,
        top_of_atmosphere_dir_dn_sw_flux = NaN,
        dir_sw_surface_albedo = get_var("surface_albedo")',
        dif_sw_surface_albedo = get_var("surface_albedo")',
        pressure = pressure,
        temperature = temperature,
        surface_temperature = get_var("surface_temperature"),
        latitude = get_var("lat"),
        volume_mixing_ratio_h2o = volume_mixing_ratio_h2o,
        volume_mixing_ratio_o3 = get_var("ozone"),
        vmrs...,
        volume_mixing_ratio_no2 = 0,
    )

    T = ClimaCore.Fields.Field(
        ClimaCore.DataLayouts.VF{NTuple{ncol, FT}}(rrtmgp_model.temperature),
        space,
    )
    F = ClimaCore.Fields.Field(
        ClimaCore.DataLayouts.VF{NTuple{ncol, FT}}(rrtmgp_model.flux),
        ClimaCore.Spaces.FaceFiniteDifferenceSpace(space),
    )
    specific_heat_capacity = (1 .- specific_humidity) .* FT(Planet.cv_d(param_set)) .+
        specific_humidity .* FT(Planet.cv_v(param_set))
    heating_rate_factor = specific_volume_over_temperature ./ specific_heat_capacity
    hrf = ClimaCore.Fields.Field(
        ClimaCore.DataLayouts.VF{NTuple{ncol, FT}}(heating_rate_factor),
        space,
    )
    t0 = get_var("time")
    longitude = get_var("lon")
    p = (; rrtmgp_model, F, hrf, t0, longitude)

    close(ds_input)
    return T, p
end

T, p = do_stuff(param_set)

# For instantaneous_zenith_angle, errors in predicted angle and irradiance with
# respect to input data are
#     Avg: 1.4%, 0.05%
#     Max: 17%, 0.08%

function tendency!(dT, T, p, t)
    tuples = instantaneous_zenith_angle.(
        p.t0 .+ Dates.Second(round(Int, t)),
        p.longitude,
        p.rrtmgp_model.latitude,
        param_set,
    ) # each tuple contains (zenith angle, azimuthal angle, earth-sun distance)

    p.rrtmgp_model.solar_zenith_angle .= min.(first.(tuples), FT(π)/2 - eps(FT))
    p.rrtmgp_model.top_of_atmosphere_dir_dn_sw_flux .=
        FT(Planet.tot_solar_irrad(param_set)) .*
        (FT(astro_unit()) ./ last.(tuples)).^2

    compute_fluxes!(p.rrtmgp_model)
    dT .= ClimaCore.Operators.GradientF2C().(p.F) .* T .* p.hrf
    return dT
end

# Solve the ODE operator
Δt = 1.
prob = ODEProblem(tendency!, T, (0., 5.), p)
sol = solve(
    prob,
    Euler(), # SSPRK33(),
    dt = Δt,
    saveat = Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)
# ydata = parent(ClimaCore.Spaces.coordinates(axes(T)))[:, 1]; plot(parent(T), ydata; xguide = "Temperature", yguide = "Elevation", ylims = extrema(ydata), legend = false)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

path = joinpath(@__DIR__, "output")
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u)
end
Plots.mp4(anim, joinpath(path, "radiation.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol.u[1]),
    joinpath(path, "radiation_begin.png"),
)
Plots.png(
    Plots.plot(sol.u[end]),
    joinpath(path, "radiation_end.png"),
)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/radiation_end.png", "Radiation End Simulation")
