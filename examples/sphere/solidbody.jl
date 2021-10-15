push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

# Advection problem on a sphere. The initial condition can be set via a command line
# argument. Possible test cases are: cosine_bell (default) and gaussian_bell

const R = 6.37122e6
const h0 = 1000.0
const r0 = R / 3
const α0 = 0.0
const u0 = 2 * pi * R / (86400 * 12)
const center = Geometry.LatLongPoint(0.0, 270.0)
const test_name = get(ARGS, 1, "cosine_bell") # default test case to run
const cosine_test_name = "cosine_bell"
const gaussian_test_name = "gaussian_bell"


FT = Float64
ne_seq = 2 .^ (2, 3, 4, 5)
err, Δh = zeros(FT, length(ne_seq)), zeros(FT, length(ne_seq))
Nq = 4

# h-refinement study
for (k, ne) in enumerate(ne_seq)
    domain = Domains.SphereDomain(R)
    mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), ne)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    coords = Fields.coordinate_field(space)

    Δh[k] = 2 * R / ne

    h_init = map(coords) do coord
        n_coord = Geometry.components(Geometry.Cartesian123Point(coord))
        n_center = Geometry.components(Geometry.Cartesian123Point(center))
        # https://en.wikipedia.org/wiki/Great-circle_distance
        rd = R * atan(norm(n_coord × n_center), dot(n_coord, n_center))

        if test_name == gaussian_test_name
            h0 * exp(-(rd / r0)^2 / 2)
        else # default test case, cosine bell
            if rd < r0
                h0 / 2 * (1 + cospi(rd / r0))
            else
                0.0
            end
        end

    end

    u = map(coords) do coord
        ϕ = coord.lat
        λ = coord.long

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end

    function rhs!(dh, h, u, t)
        div = Operators.Divergence()

        # add in pieces
        @. dh = -div(h * u)
        Spaces.weighted_dss!(dh)
    end
    rhs!(similar(h_init), h_init, u, 0.0)

    # Solve the ODE operator
    T = 86400 * 12
    dt = 30 * 60
    prob = ODEProblem(rhs!, h_init, (0.0, T), u)
    sol = solve(
        prob,
        SSPRK33(),
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    err[k] = norm(sol.u[end] .- h_init) / length(h_init)

    @info "Test case: $(test_name)"
    @info "Number of elements per cube panel: $(ne) x $(ne)"
    @info "Solution norm at t = 0: ", norm(h_init)
    @info "Solution norm at t = $(T): ", norm(sol.u[end])
    @info "L2 error at t = $(T): ", norm(sol.u[end] .- h_init)

end

conv = convergence_rate(err, Δh)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg_sphere_solidbody"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

Plots.png(Plots.plot(sol.u[end] .- h_init), joinpath(path, "error.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/error.png", "Absolute error in height")

#----------------lat, long plots
Plots.png(Plots.plot(coords.lat), joinpath(path, "latitude.png"))
linkfig("output/$(dirname)/latitude.png", "latitude")
Plots.png(Plots.plot(coords.long), joinpath(path, "longitude.png"))
linkfig("output/$(dirname)/longitude.png", "longitude")
