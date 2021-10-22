push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
import ClimaCore.Domains.Geometry: Cartesian2DPoint
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include("../implicit_solver_utils.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("right_hand_sides.jl")

# set up function space
function hvspace_2D(xmin, xmax, zmin, zmax, velem, helem, npoly)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zmin),
        Geometry.ZPoint{FT}(zmax);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xmin)..Geometry.XPoint{FT}(xmax),
        Geometry.YPoint{FT}(-0)..Geometry.YPoint{FT}(0),
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, helem, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return hv_center_space, hv_face_space
end

gravitational_potential(z) = grav * z

function init_inertial_gravity_wave_ρθ(x, z, A)
    p_0 = MSLP
    g = grav
    cp_d = R_d * γ / (γ - 1)
    x_c = 0.
    θ_0 = 300.
    Δθ = 0.01
    H = 10000.
    NBr = 0.01
    S = NBr * NBr / g

    p_ref = p_0 * (1 - g / (cp_d * θ_0 * S) * (1 - exp(-S * z)))^(cp_d / R_d)
    θ = θ_0 * exp(z * S) + Δθ * sin(pi * z / H) / (1 + ((x - x_c) / A)^2)
    ρ = p_ref / ((p_ref / p_0)^(R_d / cp_d) * R_d * θ)
    ρθ = ρ * θ

    return (ρ = ρ, ρθ = ρθ, ρuₕ = Geometry.Cartesian1Vector(0.))
end
function init_inertial_gravity_wave_ρe_tot(x, z, A)
    ρ, ρθ, ρuₕ = init_inertial_gravity_wave_ρθ(x, z, A)
    ρe_tot = (P_ρθ_factor * ρθ^γ) / (γ - 1) + ρ * gravitational_potential(z)
    return (ρ = ρ, ρe_tot = ρe_tot, ρuₕ = ρuₕ)
end

using OrdinaryDiffEq
function inertial_gravity_wave_prob(;
    𝔼_var,
    𝕄_var,
    helem,
    velem,
    npoly,
    is_large_domain,
    ode_algorithm,
    is_imex,
    tspan,
)
    xmax = is_large_domain ? 1500000. : 150000.
    zmax = 10000.
    A = is_large_domain ? 100000. : 5000.

    hv_center_space, hv_face_space =
        hvspace_2D(-xmax, xmax, 0., zmax, velem, helem, npoly)
    coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)
    if 𝔼_var == :ρθ
        Yc = map(c -> init_inertial_gravity_wave_ρθ(c.x, c.z, A), coords)
    elseif 𝔼_var == :ρe_tot
        Yc = map(c -> init_inertial_gravity_wave_ρe_tot(c.x, c.z, A), coords)
    else
        throw(ArgumentError("Invalid 𝔼_var $𝔼_var"))
    end
    𝕄 = map(c -> Geometry.Cartesian3Vector(0.), face_coords)

    uₕ = map(c -> Geometry.Cartesian1Vector(0.), coords)
    uₕ_f = map(c -> Geometry.Cartesian1Vector(0.), face_coords)
    P = map(c -> 0., coords)
    Φ = map(c -> gravitational_potential(c.z), coords)
    ∇ᵥf_Φ = Operators.GradientC2F(
        bottom = Operators.SetValue(gravitational_potential(0.)),
        top = Operators.SetValue(gravitational_potential(zmax)),
    )
    ∇Φ = @. Geometry.transform(ẑ(), ∇ᵥf_Φ(Φ))

    if 𝕄_var == :ρw
        Y = Fields.FieldVector(; Yc, ρw = 𝕄)
        p = (; coords, face_coords, uₕ, uₕ_f, P, Φ, ∇Φ)
    elseif 𝕄_var == :w
        Y = Fields.FieldVector(; Yc, w = 𝕄)
        ρw = similar(𝕄)
        p = (; coords, face_coords, ρw, uₕ, uₕ_f, P, Φ, ∇Φ)
    else
        throw(ArgumentError("Invalid 𝕄_var $𝕄_var"))
    end
    
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        coords,
        face_coords,
        use_transform,
        true,
        true,
    )
    w_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)
    if is_imex
        prob = SplitODEProblem(
            ODEFunction(
                rhs_vertical!;
                w_kwarg...,
                jac_prototype = jac_prototype,
                tgrad = (dT, Y, p, t) -> fill!(dT, 0),
            ),
            rhs_horizontal!,
            Y,
            tspan,
            p,
        )
    else
        prob = ODEProblem(
            ODEFunction(
                rhs!;
                w_kwarg...,
                jac_prototype = jac_prototype,
                tgrad = (dT, Y, p, t) -> fill!(dT, 0),
            ),
            Y,
            tspan,
            p,
        )
    end

    return prob
end

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

function get_ρθ(Y, p)
    if :ρθ in propertynames(Y.Yc)
        return Y.Yc.ρθ
    elseif :ρe_tot in propertynames(Y.Yc)
        @unpack P, Φ = p
        if :ρw in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot - Y.Yc.ρ * Φ -
                norm_sqr(Y.Yc.ρuₕ, Ic(Y.ρw)) / (2. * Y.Yc.ρ)
            )
        elseif :w in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot -
                Y.Yc.ρ * (Φ + norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ, Ic(Y.w)) / 2.)
            )
        end
        return @. (P / P_ρθ_factor)^(1. / γ)
    end
end

function inertial_gravity_wave_plots(sol, name)
    dirname = "inertial_gravity_wave"
    path = joinpath(@__DIR__, "output", dirname)
    mkpath(path)

    p = sol.prob.p
    coords = p.coords
    θ_ref = 300. .* exp.(coords.z .* (0.01 * 0.01 / grav))
    anim = Plots.@animate for Y in sol.u
        Plots.plot(get_ρθ(Y, p) ./ Y.Yc.ρ .- θ_ref, clim = (-0.002, 0.012))
    end
    Plots.mp4(anim, joinpath(path, "Δθ_$name.mp4"), fps = 20)
    anim = Plots.@animate for Y in sol.u
        Plots.plot(
            P_ρθ_factor .* (get_ρθ(Y, p).^γ .- (Y.Yc.ρ .* θ_ref).^γ),
            clim = (0., 3.),
        )
    end
    Plots.mp4(anim, joinpath(path, "Δp_$name.mp4"), fps = 20)
end

# reltol = 1e-2
# abstol = 1e-8
# use generated functions to build timestepper from Butcher Tableau
#     handle 0, 1, -1, and 1//N nicely
#     automatically unroll stage looops and eliminate tableau dereferences


# 1. Redo plots (bar graphs with total f+J evals per solver type)
# 2. Change tolerances on KenCarp4
#    - Plot of adaptive timestep over time
# 3. Redo current examples with large rising bubble
#    - Increase vertical resolution until explicit solver hits tme limit
#    - Show imex solver exceeds limit
# 4. Implent arbitrary Butcher tableau explicit and Rosenbrock implicit solvers
# 5. Make new rhs with ρe_tot instead of ρθ

# Explicit: 
# | Algorithm type | Δt | wall time | final error | num f calls |

# Implicit: 
# | Algorithm type | Δt | wall time | final error | num f calls | num J calls | num linsolves |

# Imex: 
# | Algorithm type | Δt | wall time | final error | num f_ex calls | num f_im calls | num J calls | num linsolves |