# Examples

## 1D Column examples

## 2D Cartesian examples

## 3D Cartesian examples

### Flux Limiters advection

The 3D Cartesian advection/transport example in [`examples/hybrid/box/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples/hybrid/box/limiters_advection.jl) demonstrates the application of flux limiters in the horizontal direction, namely [`QuasiMonotoneLimiter`](https://clima.github.io/ClimaCore.jl/previews/PR729/api/#ClimaCore.Limiters.QuasiMonotoneLimiter), in a hybrid Cartesian domain. It also demonstrates the usage of the high-order upwinding scheme in the vertical direction, called [`Upwind3rdOrderBiasedProductC2F`](@ref).

#### Equations and discretizations

#### Mass

Follows the continuity equation
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
\label{eq:continuity}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho \approx - D_h[ \rho (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] - D^c_v[I^f(\rho \boldsymbol{u}_h)) + I^f(\rho) \boldsymbol{u}_v)] .
\label{eq:discrete-continuity}
\end{equation}
```

#### Tracers

For the tracer concentration per unit mass ``q``, the tracer density (scalar) ``\rho q`` follows the advection/transport equation

```math
\begin{equation}
  \frac{\partial}{\partial t} \rho q = - \nabla \cdot(\rho q \boldsymbol{u})  + g(\rho, q).
\label{eq:tracers}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
\frac{\partial}{\partial t} \rho q \approx
- D_h[ \rho q (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))]
- D^c_v\left[I^f(\rho) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho q}{\rho} \right) \right] + g(\rho, q),
\label{eq:discrete-tracers}
\end{equation}
```
where ``g(\rho, q) = - \nu_4 [\nabla^4_h (\rho q)]`` represents the horizontal hyperdiffusion operator, with ``\nu_4`` (measured in m^4/s) the hyperviscosity constant coefficient.

Currently tracers are only treated explicitly in the time discretization.


#### Prognostic variables

* ``\rho``: _density_ measured in kg/m³. This is discretized at cell centers.
* ``\boldsymbol{u}`` _velocity_, a vector measured in m/s. This is discretized via ``\boldsymbol{u} = \boldsymbol{u}_h + \boldsymbol{u}_v`` where
  - ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
  - ``\boldsymbol{u}_v = u_3 \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
* ``\rho q``: the tracer density scalar, where ``q`` is the tracer concentration per unit mass, is stored at cell centers.

#### Operators

We make use of the following operators

#### Reconstructions

* ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C), called `first_order_If2c` in the example code.
* ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F), called `first_order_Ic2f` in the example code.
  - Currently this is just the arithmetic mean, but we will need to use a weighted version with stretched vertical grids.
* ``U^f`` is the [center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F), called `third_order_upwind_c2f` in the example code
  - This operator is of third-order of accuracy (when used with a constant vertical velocity and some reduced, but still high-order for non constant vertical velocity).


#### Differentiation operators

- ``D_h`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `hwdiv` in the example code.
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `vdivf2c` in the example code.
  - This example uses advective fluxes equal to zero at the top and bottom boundaries.
- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `hgrad` in the example code.

To discretize the hyperdiffusion operator, ``g(\rho, q) = - \nu_4 [\nabla^4 (\rho q)]``, in the horizontal direction, we compose the horizontal weak divergence, ``D_h``, and the horizontal gradient operator, ``G_h``, twice, with an intermediate call to [`weighted_dss!`](@ref) between the two compositions, as in ``[g_2(\rho, g) \circ DSS(\rho, q) \circ g_1(\rho, q)]``, with:
- ``g_1(\rho, q) = D_h(G_h(q))``
- ``DSS(\rho, q) = DSS(g_1(\rho q))``
- ``g_2(\rho, q) = -\nu_4 D_h(\rho G_h(\rho q))``
    - with ``\nu_4`` the hyperviscosity coefficient.

#### Problem flow and set-up

This test case is set up in a Cartesian (box) domain `` [-2 \pi, 2 \pi]^2 \times [0, 4 \pi] ~\textrm{m}^3``, doubly periodic in the horizontal direction, but not in the vertical direction.

The flow was chosen to be a spiral, i.e., so to have a horizontal uniform rotation, and a vertical velocity ``\boldsymbol{u}_v \equiv w = 0`` at the top and bottom boundaries, and ``\boldsymbol{u}_v \equiv w = 1`` in the center of the domain. Moreover, the flow is reversed in all directions halfway through the period so that the tracer blobs go back to its initial configuration (using the same speed scaling constant which was derived to account for the distance travelled in all directions in a half period).

```math
\begin{align}
    u &= -u_0 (y - c_y) \cos(\pi t / T_f) \nonumber \\
    v &= u_0 (x - c_x) \cos(\pi t / T_f) \nonumber \\
    w &= u_0 \sin(\pi z / z_m) \cos(\pi t / T_f) \nonumber
\label{eq:flow}
\end{align}
```
where ``u_0 = \pi / 2`` is the speed scaling factor to have the flow reversed halfway through the period, ``\boldsymbol{c} = (c_x, c_y)`` is the center of the rotational flow, which coincides with the center of the domain,  ``z_m = 4 \pi`` is the maximum height of the domain, and `` T_f = 2 \pi`` is the final simulation time, which coincides with the temporal period to have a full rotation in the horizontal direction.

This example is set up to run with three possible initial conditions:
- `cosine_bells`
- `gaussian_bells`
- `slotted_spheres`: a slight modification of the 2D slotted cylinder test case available in the literature (cfr: [GubaOpt2014](@cite)).

## 2D Sphere examples

## 3D Sphere examples
