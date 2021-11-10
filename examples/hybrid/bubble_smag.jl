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
global_logger(TerminalLogger())

# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 5,
    velem = 20,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# set up rhs!
hv_center_space, hv_face_space = hvspace_2D((-500, 500), (0, 1000))
#hv_center_space, hv_face_space = hvspace_2D((-500,500),(0,30000), 5, 30)

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

function pressure(ρθ)
    if ρθ >= 0
        return MSLP * (R_d * ρθ / MSLP)^γ
    else
        return NaN
    end
end

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_dry_rising_bubble_2d(x, z)
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    ρθ = ρ * θ # potential temperature density

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.UVector(0.0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space);
face_coords = Fields.coordinate_field(hv_face_space);

Yc = map(coords) do coord
    bubble = init_dry_rising_bubble_2d(coord.x, coord.z)
    bubble
end;

ρw = map(face_coords) do coord
    Geometry.WVector(0.0)
end;

Y = Fields.FieldVector(Yc = Yc, ρw = ρw);

function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hwdiv = Operators.WeakDivergence()
    hwgrad = Operators.WeakGradient()

    # vertical FD operators with BC's
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(0.0)),
        top = Operators.SetDivergence(Geometry.WVector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0),
        ),
        top = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0),
        ),
    )
    If_bc = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.UVector(0.0)),
        top = Operators.SetValue(Geometry.UVector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂o = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    ∂_w= Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.UVector(0.0) ⊗ Geometry.WVector(0.0)),
            top = Operators.SetValue(Geometry.UVector(0.0) ⊗ Geometry.WVector(0.0)),
    )
    ∂c2f = Operators.DivergenceC2F(
            bottom = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
            top = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
    )
    #################################
    ∂f = Operators.GradientC2F(
                               bottom = Operators.SetValue(Geometry.WVector(0.0)),
                               top = Operators.SetValue(Geometry.WVector(0.0))
                 )
    ∂hf = Operators.GradientC2F(
                               bottom = Operators.SetValue(Geometry.UVector(0.0)),
                               top = Operators.SetValue(Geometry.UVector(0.0))
                 )
    #################################
    ∂c = Operators.GradientF2C(
                               bottom = Operators.SetValue(Geometry.WVector(0.0)),
                               top = Operators.SetValue(Geometry.WVector(0.0))
                 )
    ∂hc = Operators.GradientF2C(
                               bottom = Operators.SetValue(Geometry.UVector(0.0)),
                               top = Operators.SetValue(Geometry.UVector(0.0))
                 )
    #################################
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    # 1) compute hyperviscosity coefficients
    @. dYc.ρ = hdiv(hgrad(Yc.ρ))
    @. dYc.ρθ = hdiv(hgrad(Yc.ρθ))
    @. dYc.ρuₕ = hdiv(hgrad(Yc.ρuₕ))
    @. dρw = hdiv(hgrad(ρw))
    Spaces.weighted_dss!(dYc)

    κ₄ = 0.0
    @. dYc.ρ = -κ₄ * hdiv(hgrad(dYc.ρ))
    @. dYc.ρθ = -κ₄ * hdiv(hgrad(dYc.ρθ))
    @. dYc.ρuₕ = -κ₄ * hdiv(hgrad(dYc.ρuₕ))
    @. dρw = -κ₄ * hdiv(hgrad(dρw))

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    wc = @. Ic(ρw) / Yc.ρ
    p = @. pressure(Yc.ρθ)

    # density
    @. dYc.ρ += -∂o(ρw) #+ fcc(w, Yc.ρ)
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ += -(∂o(ρw * If(Yc.ρθ / Yc.ρ))) #+ fcc(w, Yc.ρθ)
    @. dYc.ρθ -= hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.UAxis(), Geometry.UAxis()),
            @SMatrix [1.0]
        ),
    ) # ∂u∂x
    Iv = Ref(
        Geometry.Axis2Tensor(
            (Geometry.WAxis(), Geometry.WAxis()),
            @SMatrix [1.0]
        ),
    ) # ∂w∂z 
    Ihv = Ref(
        Geometry.Axis2Tensor(
            (Geometry.UAxis(), Geometry.WAxis()),
            @SMatrix [1.0]
        ),
    ) # ∂u∂z, ∂w∂x 

    @. dYc.ρuₕ += -uvdivf2c(ρw ⊗ If_bc(uₕ)) #+ fcc(w, Yc.ρuₕ)
    @. dYc.ρuₕ -= hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)

    # vertical momentum
    @. dρw +=
        B(
            Geometry.transform(
                Geometry.WAxis(),
                -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z)),
            ) - vvdivc2f(Ic(ρw ⊗ w)),
        ) #+ fcf(wc, ρw)
    uₕf = @. If_bc(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    @. dρw -= hdiv(uₕf ⊗ ρw) # Ca1Ca3 Axis2Tensor

    # Preallocate derivative variables
    Yfρ = @. If(Yc.ρ)           # Density scalars at cell faces (interpolated)
    u = @. Yc.ρuₕ / Yc.ρ      # Horizontal Velocity at Centers # UVector
    w = @. ρw / Yfρ             # Vertical Velocity at Faces @faces # WVector

    # Horizontal Gradients
    # X == cartesian1Axis
    ∂u∂x_c = @. hgrad(u)          # @centers
    ∂u∂x_f = @. hgrad(uₕf)        # @faces

    ∂w∂x_c = @. hgrad(wc)         # @centers  # where w isa WVector
    ∂w∂x_f = @. hgrad(w)  # @faces # where w==WVector

    # Vertical Gradients
    ∂u∂z_c = @. ∂hc(uₕf)        # specify b.c. on ∂hc, ∂c @faces
    ∂u∂z_f = @. ∂hf(u)          # specify b.c. on ∂hf, ∂c @faces

    ∂w∂z_c = @. ∂c(w)           # @centers
    ∂w∂z_f = @. ∂f(wc)          # @centers

    # Stress tensor components  2AxisTensors
    ρτxx = @. Yc.ρ * ∂u∂x_c       # Stress tensor diagonal components: available on faces

    # Paired off-diagonal terms (dudz + dwdx)/2  @faces
    ρτxz_13_1 = @. 1/2 * Yfρ * ∂u∂z_f 
    ρτxz_13_2 = @. 1/2 * Yfρ * ∂w∂x_f

    # Paired off-diagonal terms (dudz + dwdx)/2  @centers
    ρτxz_31_1 = @. 1/2 * Yc.ρ * ∂u∂z_c
    ρτxz_31_2 = @. 1/2 * Yc.ρ * ∂w∂x_c

    ρτzz = @. Yc.ρ * ∂w∂z_c
    
    θ = @. Yc.ρθ / Yc.ρ
    ∂θ∂x = @. hgrad(θ)

    S11 = getproperty(getproperty(getproperty(ρτxx, 2), 1), 1)
    S13 = getproperty(getproperty(getproperty(ρτxz_31_1, 2), 1), 1)
    S31 = getproperty(getproperty(getproperty(ρτxz_31_2, 2), 1), 1)
    S33 = getproperty(getproperty(getproperty(ρτzz, 2), 1), 1)
    ΣS = @. 2.0 * (S11^2 + S13^2  + S31^2 + S33^2)
    nS = @. sqrt(ΣS)
    Cₛ = 0.18

    # g^{ij} S_{jk} g^{kl} S_{li} 

    # 3) Diffusion
    κh = @. Cₛ * nS * 100.0^2 # m^2/s # Needs dynamic viscosity coefficient values
    κv = @. Cₛ * nS * 100.0^2 # m^2/s # Needs dynamic viscosity coefficient values
    
    # Smagorinsky Contributions to Horizontal Momentum
    ## Horizontal divergence of horizontal gradient of horizontal velocity @centers
    @. dYc.ρuₕ += hdiv(κh * ρτxx * Ih) 
    ## Vertical divergence of vertical gradient of horizontal velocity @centers
    @. dYc.ρuₕ += uvdivf2c(If(κh) * ρτxz_13_1)
    ## Vertical divergence of horizontal gradient of vertical velocity (Cart3) @centers
    @. dYc.ρuₕ += Geometry.transform(Geometry.UAxis(), ∂_w(If(κh) * ρτxz_13_2))

    # Smagorinsky Contributions to Vertical Momentum
    ## Vertical divergence of vertical gradient of vertical velocity @faces
    @. dρw += vvdivc2f(κv * ρτzz)
    ## Horizontal divergence of horizontal gradient of vertical velocity @faces
    @. dρw += hdiv(1.0 * ρτxz_13_2)
    ## Horizontal divergence of vertical gradient of horizontal velocity @faces
    #@. dρw += Geometry.transform(Geometry.WAxis(), hdiv(1.0 * ρτxz_13_1))
    #@. dρw += Geometry.transform(Geometry.WAxis(), hdiv(1.0 * ρτxz_13_1))
    #checkvar = @. Geometry.transform(Geometry.WAxis(), norm(hdiv(1.0 * ρτxz_13_1)) * Geometry.WVector(1.0))
    #@show summary(checkvar)

    ### Potential temperature
    ## Horizontal div of horizontal grad of potential temperature
    @. dYc.ρθ += hdiv(κh * (Yc.ρ * ∂θ∂x))
    ## Vertical div of vertial grad of potential temperature
    @. dYc.ρθ += ∂o(If(κv) * (Yfρ * ∂f(θ)))
    
    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)

    return dY
end

dYdt = similar(Y);
rhs!(dYdt, Y, nothing, 0.0);


# run!
using OrdinaryDiffEq
Δt = 0.05
prob = ODEProblem(rhs!, Y, (0.0, 700.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

@info("Simulation Ended")

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "bubble"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# post-processing
import Plots
anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρθ ./ u.Yc.ρ, clim = (300.0, 300.5))
end
Plots.mp4(anim, joinpath(path, "bubble.mp4"), fps = 20)
