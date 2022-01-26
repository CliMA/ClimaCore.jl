using OrdinaryDiffEq:
    OrdinaryDiffEq,
    ODEProblem,
    ODEFunction,
    SplitODEProblem,
    solve,
    SSPRK33,
    Rosenbrock23,
    Rosenbrock32,
    ImplicitEuler,
    NLNewton,
    KenCarp4
import ClimaCore.Utilities: half

include("held_suarez_rho_theta_utils.jl")

# Mesh setup
zmax = 30.0e3
z_D = 25.0e3 # sponge layer
helem = 4
velem = 10
npoly = 4

# set up 3D spherical domain and coords
hv_center_space, hv_face_space = sphere_3D(R, (0, zmax), helem, velem, npoly)
c_coords = Fields.coordinate_field(hv_center_space)
f_coords = Fields.coordinate_field(hv_face_space)
local_geometries = Fields.local_geometry_field(hv_center_space)

# Coriolis
const f = @. Geometry.Contravariant3Vector(
    Geometry.WVector(2 * Ω * sind(c_coords.lat)),
)

# set up initial condition
Yc = map(coord -> initial_condition(coord.lat, coord.long, coord.z), c_coords)
uₕ = map(
    local_geometry -> initial_condition_velocity(local_geometry),
    local_geometries,
)
w = map(_ -> Geometry.Covariant3Vector(0.0), f_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

# initialize tendency
dYdt = similar(Y)

Test_Type = "Implicit"    # "Explicit" # "Semi-Explicit"  "Implicit-Explicit"

# setup p
P = map(c -> 0.0, c_coords.z)
Φ = @. gravitational_potential(c_coords.z)
∇Φ = vgradc2f.(Φ)
p = (; P, Φ, ∇Φ, c_coords, zmax, z_D)

if Test_Type == "Explicit"
    T = 3600
    dt = 5
    prob = ODEProblem(rhs!, Y, (0.0, T), p)
    # solve ode
    sol = solve(
        prob,
        SSPRK33();
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
elseif Test_Type == "Semi-Explicit"
    T = 3600
    dt = 5

    prob = SplitODEProblem(rhs_implicit!, rhs_remainder!, Y, (0.0, T), p)

    # solve ode
    sol = solve(
        prob,
        SSPRK33();
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )

elseif Test_Type == "Implicit"
    nday = 10 #300 10 days for ci but need a atable run for longer, e.g., 1200 days
    T = 86400.0 * nday
    dt = 400
    p = (; p..., dt = dt)
    # p = (; p..., Ycopy = copy(Y)) # for debug purposes only

    ode_algorithm = Rosenbrock23
    J_𝕄ρ_overwrite = :grav
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    # TODO
    𝕄 = map(c -> Geometry.WVector(0.0), f_coords)
    p = (; ρw = similar(𝕄), p...)

    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        hv_center_space.center_local_geometry,
        hv_face_space.face_local_geometry,
        use_transform,
        J_𝕄ρ_overwrite;
        Y,
    )

    w_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)

    prob = ODEProblem(
        ODEFunction(
            rhs!;
            w_kwarg...,
            jac_prototype = jac_prototype,
            tgrad = (dT, Y, p, t) -> fill!(dT, 0),
        ),
        Y,
        (0, T),
        p,
    )

    sol = solve(
        prob,
        dt = dt,
        # TODO Newton
        ode_algorithm(linsolve = linsolve!),
        reltol = 1e-1,
        abstol = 1e-6,
        # TODO Linear
        # ode_algorithm(linsolve = linsolve!);
        #
        saveat = 3600 * 24 * 5,
        adaptive = false,
        progress = true,
        progress_steps = 1,
        progress_message = (dt, u, p, t) -> t,
    )
elseif Test_Type == "Implicit-Explicit"
    T = 3600
    dt = 300

    ode_algorithm = ImplicitEuler
    J_𝕄ρ_overwrite = :grav
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    # TODO
    𝕄 = map(c -> Geometry.WVector(0.0), f_coords)
    p = (; ρw = similar(𝕄), p...)

    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        hv_center_space.center_local_geometry,
        hv_face_space.face_local_geometry,
        use_transform,
        J_𝕄ρ_overwrite;
        Y,
    )

    w_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)

    prob = SplitODEProblem(
        ODEFunction(
            rhs_implicit!;
            w_kwarg...,
            jac_prototype = jac_prototype,
            tgrad = (dT, Y, p, t) -> fill!(dT, 0),
        ),
        rhs_remainder!,
        Y,
        (0, T),
        p,
    )

    sol = solve(
        prob,
        dt = dt,
        # TODO Newton
        # ode_algorithm(linsolve = linsolve!, nlsolve = NLNewton(; max_iter = 10)),
        # reltol = 1e-1,
        # abstol = 1e-6,
        # TODO Linear
        ode_algorithm(linsolve = linsolve!);
        #
        saveat = dt,
        adaptive = false,
        progress = true,
        progress_steps = 1,
        progress_message = (dt, u, p, t) -> t,
    )

else
    error("Test Type: ", Test_Type, " is not recognized.")
end

using ClimaCorePlots, Plots
ENV["GKSwstype"] = "nul"

anim = Plots.@animate for sol1 in sol.u
    uₕ = sol1.uₕ
    uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), uₕ)
    w_phy = Geometry.transform.(Ref(Geometry.WAxis()), sol1.w)
    u = uₕ_phy.components.data.:1
    v = uₕ_phy.components.data.:2
    Plots.plot(v, level = 10, clim = (-6, 6))
end

dirname = "held_suarez_rho_theta_implicit"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)
Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)
