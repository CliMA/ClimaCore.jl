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
    
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zmin),
        Geometry.ZPoint{FT}(zmax);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

    hdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xmin),
        Geometry.XPoint{FT}(xmax);
        periodic = true,
    )
    hmesh = Meshes.IntervalMesh(hdomain; nelems = helem)
    htopology = Topologies.IntervalTopology(hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace1D(htopology, quad)

    space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(space)
    return space, face_space
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

    return (ρ = ρ, ρθ = ρθ, ρuₕ = Geometry.UVector(0.))
end
function ρθ_to_ρe_tot(Yc, Φ)
    ρe_tot = (P_ρθ_factor * Yc.ρθ^γ) / P_ρe_factor + Yc.ρ * Φ
    return (ρ = Yc.ρ, ρe_tot = ρe_tot, ρuₕ = Yc.ρuₕ)
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
    J_𝕄ρ_overwrite = :none,
)
    xmax = is_large_domain ? 1500000. : 150000.
    zmax = 10000.
    A = is_large_domain ? 100000. : 5000.

    space, face_space = hvspace_2D(-xmax, xmax, 0., zmax, velem, helem, npoly)
    coords = Fields.coordinate_field(space)
    face_coords = Fields.coordinate_field(face_space)

    uₕ = map(c -> Geometry.UVector(0.), coords)
    uₕ_f = map(c -> Geometry.UVector(0.), face_coords)
    P = map(c -> 0., coords)
    Φ = map(c -> gravitational_potential(c.z), coords)
    ∇ᵥf_Φ = Operators.GradientC2F(
        bottom = Operators.SetValue(gravitational_potential(0.)),
        top = Operators.SetValue(gravitational_potential(zmax)),
    )
    ∇Φ = @. Geometry.transform(ŵ(), ∇ᵥf_Φ(Φ))
    p = (; coords, face_coords, uₕ, uₕ_f, P, Φ, ∇Φ)

    Yc = map(c -> init_inertial_gravity_wave_ρθ(c.x, c.z, A), coords)
    if 𝔼_var == :ρe_tot
        Yc = ρθ_to_ρe_tot.(Yc, Φ)
    elseif 𝔼_var != :ρθ
        throw(ArgumentError("Invalid 𝔼_var $𝔼_var (must be :ρθ or :ρe_tot)"))
    end
    𝕄 = map(c -> Geometry.WVector(0.), face_coords)

    if !(J_𝕄ρ_overwrite in (:none, :grav, :pres))
        throw(ArgumentError(string(
            "Invalid J_𝕄ρ_overwrite $J_𝕄ρ_overwrite (must be :none, :grav, ",
            "or :pres)",
        )))
    end

    if 𝕄_var == :ρw
        Y = Fields.FieldVector(; Yc, ρw = 𝕄)
        if J_𝕄ρ_overwrite == :grav && 𝔼_var == :ρθ
            throw(ArgumentError(
                "J_𝕄ρ_overwrite must be :none if 𝔼_var is :ρθ and 𝕄_var is :ρw"
            ))
        end
        if J_𝕄ρ_overwrite == :pres
            throw(ArgumentError(
                "J_𝕄ρ_overwrite can't be :pres if 𝕄_var is :ρw"
            ))
        end
    elseif 𝕄_var == :w
        p = (; ρw = similar(𝕄), p...)
        Y = Fields.FieldVector(; Yc, w = 𝕄)
        if J_𝕄ρ_overwrite == :pres && 𝔼_var == :ρθ
            throw(ArgumentError(
                "J_𝕄ρ_overwrite can't be :pres if 𝔼_var is :ρθ and 𝕄_var is :w"
            ))
        end
    else
        throw(ArgumentError("Invalid 𝕄_var $𝕄_var (must be :ρw or :w)"))
    end
    
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        coords,
        face_coords,
        use_transform,
        J_𝕄ρ_overwrite,
    )
    w_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)
    if is_imex
        prob = SplitODEProblem(
            ODEFunction(
                rhs_implicit!;
                w_kwarg...,
                jac_prototype = jac_prototype,
                tgrad = (dT, Y, p, t) -> fill!(dT, 0),
            ),
            rhs_remainder!,
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

# algorithm isa OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm -> use linsolve!

# profiling code; use @prof "<name>" <code> to generate "name.cpuprofile" file
# using Profile
# using ChromeProfileFormat
# Profile.init(n = 10^7, delay = 0.001)
# macro prof(s::String, ex)
#     return quote
#         Profile.clear()
#         Profile.@profile $(esc(ex))
#         ChromeProfileFormat.save_cpuprofile(
#             string($s, ".cpuprofile");
#             from_c = true,
#         )
#     end
# end

# temporary FieldVector broadcast and fill patches that speeds up solves by 2-3x
import Base: copyto!, fill!
using Base.Broadcast: Broadcasted, broadcasted, BroadcastStyle
transform_broadcasted(bc::Broadcasted{Fields.FieldVectorStyle}, symb, axes) =
    Broadcasted(bc.f, map(arg -> transform_broadcasted(arg, symb, axes), bc.args), axes)
transform_broadcasted(fv::Fields.FieldVector, symb, axes) =
    parent(getproperty(fv, symb))
transform_broadcasted(x, symb, axes) = x
@inline function Base.copyto!(
    dest::Fields.FieldVector,
    bc::Broadcasted{Fields.FieldVectorStyle},
)
    for symb in propertynames(dest)
        p = parent(getproperty(dest, symb))
        copyto!(p, transform_broadcasted(bc, symb, axes(p)))
    end
    return dest
end
function Base.fill!(a::Fields.FieldVector, x)
    for symb in propertynames(a)
        fill!(parent(getproperty(a, symb)), x)
    end
    return a
end