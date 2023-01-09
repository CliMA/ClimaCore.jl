using CUDA
using ClimaComms
using DocStringExtensions

import ClimaCore:
    Device,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

"""
    PhysicalParameters{FT}

Physical parameters needed for the simulation.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PhysicalParameters{FT} # rename to PhysicalParameters
    "Radius of earth"
    R::FT = FT(6.37122e6)
    "Rotation rate of earth"
    Ω::FT = FT(7.292e-5)
    "Gravitational constant"
    g::FT = FT(9.80616)
    "Hyperdiffusion coefficient"
    D₄::FT = FT(1.0e16)
end
#This example solves the shallow-water equations on a cubed-sphere manifold.
#This file contains five test cases:
abstract type AbstractTest end
"""
    SteadyStateTest{FT, P} <: AbstractTest

The first one, called "steady_state", reproduces Test Case 2 in Williamson et al,
"A standard test set for numerical approximations to the shallow water
equations in spherical geometry", 1992. This test case gives the steady-state
solution to the non-linear shallow water equations. It consists of solid
body rotation or zonal flow with the corresponding geostrophic height field.
This can be run with an angle α that represents the angle between the north
pole and the center of the top cube panel.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SteadyStateTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 2 * pi * params.R / (12 * 86400)
    "peak of analytic height field"
    h0::FT = 2.94e4 / params.g
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
SteadyStateTest(α::FT) where {FT} =
    SteadyStateTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    SteadyStateCompactTest{FT, P} <: AbstractTest

A second one, called "steady_state_compact", reproduces Test Case 3 in the same
reference paper. This test case gives the steady-state solution to the
non-linear shallow water equations with nonlinear zonal geostrophic flow
with compact support.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SteadyStateCompactTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 2 * pi * params.R / (12 * 86400)
    "peak of analytic height field"
    h0::FT = 2.94e4 / params.g
    "latitude lower bound for coordinate transformation parameter"
    ϕᵦ::FT = -30.0
    "latitude upper bound for coordinate transformation parameter"
    ϕₑ::FT = 90.0
    "velocity perturbation parameter"
    xₑ::FT = 0.3
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
SteadyStateCompactTest(α::FT) where {FT} =
    SteadyStateCompactTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    MountainTest{FT, P} <: AbstractTest

A third one, called "mountain", reproduces Test Case 5 in the same
reference paper. It represents a zonal flow over an isolated mountain,
where the governing equations describe a global steady-state nonlinear
zonal geostrophic flow, with a corresponding geostrophic height field over
a non-uniform reference surface h_s.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MountainTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 20.0
    "peak of analytic height field"
    h0::FT = 5960
    "radius of conical mountain"
    a::FT = 20.0
    "center of mountain long coord, shifted by 180 compared to the paper, 
    because our λ ∈ [-180, 180] (in the paper it was 270, with λ ∈ [0, 360])"
    λc::FT = 90.0
    "latitude coordinate for center of mountain"
    ϕc::FT = 30.0
    "mountain peak height"
    h_s0::FT = 2e3
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
MountainTest(α::FT) where {FT} =
    MountainTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    RossbyHaurwitzTest{FT, P} <: AbstractTest

A fourth one, called "rossby_haurwitz", reproduces Test Case 6 in the same
reference paper. It represents the solution of the nonlinear barotropic
vorticity equation on the sphere

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct RossbyHaurwitzTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "velocity amplitude parameter"
    a::FT = 4.0
    "peak of analytic height field"
    h0::FT = 8.0e3
    "vorticity amplitude parameter (1/sec)"
    ω::FT = 7.848e-6
    "vorticity amplitude parameter (1/sec)"
    K::FT = 7.848e-6
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
RossbyHaurwitzTest(α::FT) where {FT} =
    RossbyHaurwitzTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    BarotropicInstabilityTest{FT, P} <: AbstractTest

A fifth one, called "barotropic_instability", reproduces the test case in
Galewsky et al, "An initial-value problem for testing numerical models of
the global shallow-water equations", 2004 (also in Sec. 7.6 of Ullirch et al,
"High-order ﬁnite-volume methods for the shallow-water equations on
the sphere", 2010). This test case consists of a zonal jet with compact
support at a latitude of 45°. A small height disturbance is then added,
which causes the jet to become unstable and collapse into a highly vortical
structure.

https://doi.org/10.3402/tellusa.v56i5.14436
https://doi.org/10.1016/j.jcp.2010.04.044

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct BarotropicInstabilityTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "maximum zonal velocity"
    u_max::FT = 80.0
    "mountain shape parameters"
    αₚ::FT = 19.09859
    "mountain shape parameters"
    βₚ::FT = 3.81971
    "peak of balanced height field from Tempest 
    https://github.com/paullric/tempestmodel/blob/master/test/shallowwater_sphere/BarotropicInstabilityTest.cpp#L86"
    h0::FT = 10158.18617
    "local perturbation peak height"
    h_hat::FT = 120.0
    "southern jet boundary"
    ϕ₀::FT = 25.71428
    "northern jet boundary"
    ϕ₁::FT = 64.28571
    "height perturbation peak location"
    ϕ₂::FT = 45.0
    "zonal velocity decay parameter"
    eₙ::FT = exp(-4.0 / (deg2rad(ϕ₁) - deg2rad(ϕ₀))^2)
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
BarotropicInstabilityTest(α::FT) where {FT} =
    BarotropicInstabilityTest{FT, PhysicalParameters{FT}}(; α = α)

function shallow_water_driver_cuda(ARGS, ::Type{FT}) where {FT}
    device = Device.device()
    context = ClimaComms.SingletonCommsContext(device)
    println("running serial simulation on $device device")
    # Test case specifications
    test_name = get(ARGS, 1, "steady_state") # default test case to run
    test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
    α = parse(FT, replace(test_angle_name, "alpha" => ""))

    println("Test name: $test_name, α = $(α)⁰")
    # Test-specific physical parameters
    test =
        test_name == "steady_state_compact" ? SteadyStateCompactTest(α) :
        (
            test_name == "mountain" ? MountainTest(α) :
            (
                test_name == "rossby_haurwitz" ? RossbyHaurwitzTest(α) :
                (
                    test_name == "barotropic_instability" ?
                    BarotropicInstabilityTest(α) : SteadyStateTest(α)
                )
            )
        )
    # Set up discretization
    ne = 9 # the rossby_haurwitz test case's initial state has a singularity at the pole. We avoid it by using odd number of elements
    Nq = 4

    domain = Domains.SphereDomain(test.params.R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Spaces.Quadratures.GLL{Nq}()
    grid_topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    return nothing
end

shallow_water_driver_cuda(ARGS, Float64)
