# Differentiability

ClimaCore fields and operators are generic in their element type, and forward-
mode automatic differentiation with dual numbers works through them. This page
states what that covers, how the library itself uses it, and what is
planned.

## Forward mode today

A `Field` may hold `ForwardDiff.Dual` values, and broadcast expressions,
reductions, finite-difference stencils, and spectral-element operators
propagate them. A model tendency evaluated on a dual-valued state returns the
tendency and its directional derivative in one pass. The example below
differentiates the squared norm of a diffusive tendency with respect to the
diffusivity and checks it against the analytic answer, which is exact because
the tendency is linear in `κ`.

```@example forwarddiff
import ForwardDiff
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Meshes, Spaces, Fields, Geometry, Operators

FT = Float64
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(1);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain; nelems = 32)
space = Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), mesh)
z = Fields.coordinate_field(space).z

# ∑ [∂z(κ ∂z θ)]² for θ = sin(2πz) with zero-flux boundaries, as a function
# of the diffusivity κ. Adding zero(κ) promotes θ to the dual element type.
function dissipation(κ)
    θ = @. sin(2π * z) + zero(κ)
    zero_flux = Operators.SetGradient(Geometry.WVector(FT(0)))
    grad = Operators.GradientC2F(bottom = zero_flux, top = zero_flux)
    div = Operators.DivergenceF2C()
    tendency = @. div(κ * grad(θ))
    return sum(tendency .^ 2)
end

κ = 0.1
derivative = ForwardDiff.derivative(dissipation, κ)
(derivative, 2 * dissipation(κ) / κ)
```

Boundary-condition values are stored with the space's float type, so they are
given as `FT` here while the dual number enters through the field values.

The library uses forward mode in two places of its own. The metric terms of
the cubed sphere are the Jacobian of the coordinate map, computed by
`ForwardDiff.jacobian` at grid construction (`autodiff_metric = true`, the
default of `SpectralElementGrid2D`) rather than by differentiating the nodal
coordinates spectrally, so they are exact to round-off. And the atmosphere's
implicit vertical solver assembles its Jacobian by evaluating the implicit
tendency on a dual-valued state with a seed matrix, exactly or in sparse
column-colored form, and storing the result in `MatrixFields` banded matrices;
ClimaAtmos's [implicit solver page](https://clima.github.io/ClimaAtmos.jl/stable/implicit_solver/)
describes the algorithms.

## Why it matters

The CliMA models are calibrated against observations and high-resolution
simulations. The methods in production use are gradient-free ensemble Kalman
schemes [Schneider17c, Lopez-Gomez22a](@cite), which need only forward model
evaluations and cope with the noisy statistics of turbulent flows.
Gradient-based calibration, sensitivity analysis, and the training of
embedded machine-learned closures need derivatives of the model with respect
to its parameters; a dynamical core through which dual numbers propagate
supplies those in forward mode, at a cost per parameter direction comparable
to one model evaluation, which is the regime of tens of parameters.

## Planned

  - **Reverse-mode differentiation** (adjoints, as Enzyme.jl or Zygote.jl
    provide) is future work. Kernels mutate their outputs, the GPU extension
    uses low-level indexing, and DSS is a scatter-add; each of these needs
    adjoint rules or Enzyme support before a gradient with respect to many
    parameters, where forward mode becomes too expensive, is available.
  - **Test coverage of dual-valued fields on GPUs.** The atmosphere's Jacobian
    assembly exercises this path; ClimaCore's own test suite covers it on CPUs.
  - **Dual-valued grids.** Coordinates and metric terms are computed once with
    the space's float type; derivatives with respect to a grid parameter (a
    mountain height, a stretching parameter) will need dual numbers to flow
    through grid construction.

The shared developer guide
[ad\_compatibility.md](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/performance/ad_compatibility.md)
lists the coding patterns that keep new code differentiable.
