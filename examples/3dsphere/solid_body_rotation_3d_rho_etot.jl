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

include("solid_body_rotation_3d_rho_etot_utils.jl")

# Mesh setup
zmax = 30.0e3
helem = 4
velem = 15
npoly = 4


# set up 3D spherical domain and coords
hv_center_space, hv_face_space = sphere_3D(R, (0, 30.0e3), helem, velem, npoly)
c_coords = Fields.coordinate_field(hv_center_space)
f_coords = Fields.coordinate_field(hv_face_space)

# Coriolis
const f = @. Geometry.Contravariant3Vector(
    Geometry.WVector(2 * Ω * sind(c_coords.lat)),
)

# set up initial condition
Yc = map(coord -> init_sbr_thermo(coord.z), c_coords)
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), c_coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), f_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

# initialize tendency
dYdt = similar(Y)


Test_Type =  "Implicit-Explicit" #"Seim-Explicit"  #"Implicit-Explicit"    # "Explicit" # "Seim-Explicit"  "Implicit-Explicit"

# setup p
P = map(c -> 0., c_coords.z)
Φ = @. gravitational_potential(c_coords.z)
∇Φ = vgradc2f.(Φ)
p = (;P, Φ, ∇Φ)

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
elseif Test_Type == "Seim-Explicit"
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
elseif Test_Type == "Implicit-Explicit"
    T = 86400
    dt = 100

    ode_algorithm =  ImplicitEuler
    J_𝕄ρ_overwrite = :none
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    # TODO
    𝕄 = map(c -> Geometry.WVector(0.), f_coords)
    p = (; ρw = similar(𝕄), p...)

    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        hv_center_space.center_local_geometry, 
        hv_face_space.face_local_geometry,
        use_transform,
        J_𝕄ρ_overwrite,
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
uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), sol.u[end].uₕ)
w_phy = Geometry.transform.(Ref(Geometry.WAxis()), sol.u[end].w)

@test maximum(abs.(uₕ_phy.components.data.:1)) ≤ 1e-11
@test maximum(abs.(uₕ_phy.components.data.:2)) ≤ 1e-11

@info "maximum vertical velocity is ", maximum(abs.(w_phy.components.data.:1))

@test maximum(abs.(w_phy.components.data.:1)) ≤ 1.0

@test norm(sol.u[end].Yc.ρ) ≈ norm(sol.u[1].Yc.ρ) rtol = 1e-2
@test norm(sol.u[end].Yc.ρe_tot) ≈ norm(sol.u[1].Yc.ρe_tot) rtol = 1e-2
