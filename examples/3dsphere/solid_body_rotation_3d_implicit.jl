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
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

global_logger(TerminalLogger())

const R = 6.4e6 # radius
const Ω = 7.2921e-5 # Earth rotation (radians / sec)
const z_top = 3.0e4 # height position of the model top
const grav = 9.8 # gravitational constant
const p_0 = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)
const cp_d = R_d * γ / (γ - 1)
const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height

# set up function space
function sphere_3D(
    R = 6.4e6,
    zlim = (0, 30.0e3),
    helem = 4,
    zelem = 15,
    npoly = 5,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# geopotential
Φ(z) = grav * z
Π(ρθ) = cp_d * (R_d * ρθ / p_0)^(R_d / cv_d)
pressure(ρθ) = (ρθ*R_d/p_0)^γ * p_0

# initial conditions for density and ρθ
function init_sbr_thermo(z)

    p = p_0 * exp(-z / H)

    ρ = p / (R_d * T_0)

    θ = T_0*(p_0/p)^(R_d/cp_d)

    return (ρ = ρ, ρθ = ρ*θ)
end

function rhs!(dY, Y, _, t)
    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.Gradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.Curl() # Operator.WeakCurl()

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    cp = @. pressure(cρθ)

    @. duₕ -= hgrad(cp) / cρ


    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    # TODO which one
    # @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= R_d/cv_d * Ic2f(Π(cρθ)) * vgradc2f(cρθ) / Ic2f(cρ)


    cE = @. (norm(cuvw)^2) / 2 + Φ(c_coords.z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) ρθ

    @. dρθ -= hdiv(cuvw * cρθ)
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY

end




function rhs_explicit!(dY, Y, _, t)
    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.Gradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.Curl() # Operator.WeakCurl()

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # TODO implicit
    # dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    cp = @. pressure(cρθ)

    @. duₕ -= hgrad(cp) / cρ


    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    # TODO implicit
    #@. dw -= vgradc2f(cp) / Ic2f(cρ)
    #@. dw -= R_d/cv_d * Ic2f(Π(cρθ)) * vgradc2f(cρθ) / Ic2f(cρ)


    cE = @. (norm(cuvw)^2) / 2 + Φ(c_coords.z)
    @. duₕ -= hgrad(cE)

    # TODO implicit
    #@. dw -= vgradc2f(cE)
    cK = @. (norm(cuvw)^2) / 2
    @. dw -= vgradc2f(cK)
    


    # 3) ρθ
    @. dρθ -= hdiv(cuvw * cρθ)
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))

    # TODO  implicit
    # @. dρθ -= vdivf2c(fw * Ic2f(cρθ))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end



function rhs_implicit!(dY, Y, _, t)
    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation

    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # TODO implicit
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    cp = @. pressure(cρθ)

    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )

    # @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= R_d/cv_d * Ic2f(Π(cρθ)) * vgradc2f(cρθ) / Ic2f(cρ)
    @. dw -= vgradc2f(Φ(c_coords.z))

    # 3) ρθ
    # TODO implicit
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))

    return dY
end



















# Mesh setup
zmax = 30.0e3
helem = 4
velem = 15
npoly = 5

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
# set up rhs
rhs!(dYdt, Y, nothing, 0.0)

# run!
using OrdinaryDiffEq
# Solve the ODE



Test_Type = "Explicit"  # "Explicit" # "Seim-Explicit"  "Implicit-Explicit"

if Test_Type == "Explicit"
    T = 3600
    dt = 5
    prob = ODEProblem(rhs!, Y, (0.0, T))
    # solve ode
    sol = solve(
        prob,
        SSPRK33(),
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
elseif Test_Type == "Seim-Explicit"
    prob = SplitODEProblem(rhs_implicit!, rhs_explicit!, Y, (0.0, T))
    T = 3600
    dt = 5
    # solve ode
    sol = solve(
        prob,
        SSPRK33(),
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
elseif Test_Type == "Implicit-Explicit"
    T = 3600
    dt = 5

    ode_algorithm = 
    J_𝕄ρ_overwrite = 

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

    sol = solve(
    prob,
    ode_algorithm(linsolve = linsolve!, nlsolve = NLNewton(; max_iter = 10));
    dt = 25.,
    reltol = 1e-1,
    abstol = 1e-6,
    adaptive = false,
    saveat = 25.,
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
@test norm(sol.u[end].Yc.ρθ) ≈ norm(sol.u[1].Yc.ρθ) rtol = 1e-2
