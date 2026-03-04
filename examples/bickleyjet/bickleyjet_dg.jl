using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    RecursiveApply,
    Spaces,
    Quadratures,
    Topologies,
    Remapping
import ClimaCore.Geometry: ⊗
import ClimaCore.RecursiveApply: ⊞, rdiv, rmap

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const context = ClimaComms.context()

const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)

numflux_name = get(ARGS, 1, "rusanov")
boundary_name = get(ARGS, 2, "")

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = boundary_name != "noslip",
        boundary_names = boundary_name != "noslip" ? nothing : (:south, :north),
    ),
)

n1, n2 = 32,32
Nq = 4
Nqh = 7
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

# higher-order space that can be used for over-integration
Iquad = Quadratures.GLL{Nqh}()
Ispace = Spaces.SpectralElementSpace2D(grid_topology, Iquad)

function init_state(coord, p)
    x, y = coord.x, coord.y
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(y)^(-2)

    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * x)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)


    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * y)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ * u)
end

function energy(state, p)
    ρ, ρu = state.ρ, state.ρu
    u = ρu / ρ
    return ρ * (u.u^2 + u.v^2) / 2 + p.g * ρ^2 / 2
end

entropy(state, p) = -energy(state, p)

function entropy_flux(state, p, n)
    η = entropy(state, p)
    u = state.ρu / state.ρ
    return η * (u' * n)
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end

function total_entropy(y, parameters)
    sum(state -> entropy(state, parameters), y)
end

# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5

    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)
    rmap(f -> f' * n, Favg) ⊞ Δf
end


numflux = if numflux_name == "central"
    Operators.CentralNumericalFlux(flux)
elseif numflux_name == "rusanov"
    Operators.RusanovNumericalFlux(flux, wavespeed)
elseif numflux_name == "roe"
    roeflux
elseif numflux_name == "kep"
    Operators.KineticEnergyPreservingNumericalFlux()
else
    error("Unknown numerical flux name: $numflux_name")
end

struct DGFluxConfig
    numflux
    overintegrate_volume::Bool
    overintegrate_faces::Bool
    use_split_form::Bool  # volume = FluxDifferencingVolume(entropy-conservative flux)
end

# Overintegration uses Interpolate/Restrict, which is not currently CUDA-kernel safe
# for this example (those operators carry full `space` objects). Disable it on GPU.
is_cpu_device = ClimaComms.device(context) isa ClimaComms.AbstractCPUDevice
dg_config = DGFluxConfig(numflux, is_cpu_device, false, numflux_name == "kep")

function rhs!(dydt, y, param_tuple, t)

    parameters, config = param_tuple

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order space
    #  D = derivative operator
    #  H = suffix for higher-order space operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    wdiv = Operators.WeakDivergence()
    local_geometry_field = Fields.local_geometry_field(y)

    # FluxDifferencingVolume has no GPU kernel (would pass non-bitstype to CUDA); use standard volume on GPU.
    use_split_form_volume =
        config.use_split_form &&
        (ClimaComms.device(axes(y)) isa ClimaComms.AbstractCPUDevice)

    if use_split_form_volume
        # Split-form volume: flux differencing with same two-point entropy-conservative
        # flux as used at interfaces (KEP adds dissipation only at faces). CPU only.
        flux_diff_vol = Operators.FluxDifferencingVolume(
            Operators.EntropyConservativeNumericalFlux(),
            parameters,
        )
        vol_term = if config.overintegrate_volume
            interp = Operators.Interpolate(Ispace)
            restr = Operators.Restrict(space)
            restr.(flux_diff_vol.(interp.(y)))
        else
            flux_diff_vol.(y)
        end
        dydt .= vol_term .* (.-(local_geometry_field.WJ))
    else
        vol_flux = flux.(y, Ref(parameters))
        wdiv_flux = if config.overintegrate_volume
            interp = Operators.Interpolate(Ispace)
            restr = Operators.Restrict(space)
            vol_flux_hi = interp.(vol_flux)
            wdiv.(vol_flux_hi) |> x -> restr.(x)
        else
            wdiv.(vol_flux)
        end
        dydt .= wdiv_flux .* (.-(local_geometry_field.WJ))
    end

    Operators.add_numerical_flux_internal!(config.numflux, dydt, y, parameters)
    Operators.add_numerical_flux_boundary!(
        dydt,
        y,
        parameters,
    ) do normal, (y⁻, parameters)
        y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu - dot(y⁻.ρu, normal) * normal, ρθ = y⁻.ρθ)
        config.numflux(normal, (y⁻, parameters), (y⁺, parameters))
    end

    # 6. Solve for final result
    dydt_data = Fields.field_values(dydt)
    dydt_data .=
        RecursiveApply.rdiv.(dydt_data, Spaces.local_geometry_data(space).WJ)
    M = Quadratures.cutoff_filter_matrix(
        Float64,
        Spaces.quadrature_style(space),
        3,
    )
    Operators.tensor_product!(dydt_data, M)
    return dydt
end

dydt = Fields.Field(similar(Fields.field_values(y0)), space)
rhs!(dydt, y0, (parameters, dg_config), 0.0);

# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 200.0), (parameters, dg_config))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.01,
    saveat = collect(0.0:1.0:200.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dg_$(numflux_name)"
if boundary_name != ""
    dir = "$(dir)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

# remap tracer to a uniformly spaced horizontal grid for plotting, using bilinear remapping
const Ninterp = 256
xpts =
    range(Geometry.XPoint(-2π), Geometry.XPoint(2π), length = Ninterp)
ypts =
    range(Geometry.YPoint(-2π), Geometry.YPoint(2π), length = Ninterp)

# Copy to CPU when on GPU so Plots and scalar comparisons work (handles 0-dim reduction results)
_cpu(x) = x isa Number ? x : (a = Array(x); ndims(a) == 0 ? a[] : a)

anim = Plots.@animate for u in sol.u
    # apply weighted DSS for plotting only, to recover a visually continuous field
    θ_plot = copy(u.ρθ)
    Spaces.weighted_dss!(θ_plot)
    # remap directly from GLL quadrature to uniformly spaced grid
    θ_array = Remapping.interpolate_array(
        θ_plot,
        xpts,
        ypts;
        horizontal_method = Remapping.BilinearRemapping(),
    )
    θ2 = _cpu(θ_array)
    Plots.heatmap(
        [p.x for p in xpts],
        [p.y for p in ypts],
        θ2';
        clim = (-1, 1),
        c = :RdBu,
    )
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [_cpu(total_energy(u, parameters)) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

# Entropy monitor: for split-form KEP (periodic, inviscid), total entropy is
# non-increasing (constant with entropy-conservative volume + no dissipation;
# decreases when interface dissipation is applied).
if numflux_name == "kep"
    Ss = [_cpu(total_entropy(u, parameters)) for u in sol.u]
    Plots.png(Plots.plot(Ss, title = "Total entropy"), joinpath(path, "entropy.png"))
    # Optional: assert entropy did not increase (entropy-stable scheme)
    @assert Ss[end] <= Ss[1] + 1e-12 "Entropy should not increase (got $(Ss[1]) -> $(Ss[end]))"
end

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
