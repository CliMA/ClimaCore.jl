# Examples

## 1D Column examples

### Heat equation

The 1D Column heat example in [`examples/column/heat.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/column/heat.jl).

#### Equations and discretizations

Follows the heat equation

```math
\begin{equation}
  \frac{\partial T}{\partial t} = \alpha \cdot \nabla^2 T.
\label{eq:1d-column-heat-continuity}
\end{equation}
```

This is discretized using the following

```math
\begin{equation}
  \frac{\partial T}{\partial t} \approx \alpha \cdot D(G(T)).
\label{eq:1d-column-heat-discrete}
\end{equation}
```

#### Prognostic Variables

* ``\alpha``: thermal diffusivity measured in  $\frac{m^2}{s}$
* ``T``: temperature

#### Differentiation Operators

 * ``D`` is the [face-to-center divergence](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.DivergenceF2C) operator, called `divf2c` in the example code
 * ``G`` is the [center-to-face gradient](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.GradientC2F) operator, called `gradc2f` in the example code

#### Set Up

This test case is set up in a 1D column domain ``z \in [0, 1]`` and discretized into a mesh of 10 elements. A homogeneous Dirichlet boundary condition is set at the bottom boundary, `bcs_bottom`, setting the temperature to 0. A Neumann boundary condition is applied to the top boundary, `bcs_top`, setting the temperature gradient to 1. 

### Advection equation

The 1D Column advection example in [`examples/column/advect.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/column/advect.jl).

#### Equations and Discretizations

Follows the advection equation

```math
\begin{equation}
  \frac{\partial \theta}{\partial t} = -\frac{\partial (v \theta)}{\partial z} .
\label{eq:1d-column-advection-continuity}
\end{equation}
```
This is discretized using the following

```math
\begin{equation}
  \frac{\partial \theta}{\partial t} \approx - D(v, \theta)  .
\label{eq:1d-column-advection-discrete}
\end{equation}
```

#### Prognostic Variables

* ``\theta``: the scalar field
* ``v``: the velocity field

#### Tendencies 

The example code solves the equation for 4 different tendencies with the following discretizations:

- Tendency 1:

  $$D = \partial(UB),$$  

  where ``\partial`` is the [`face-to-center divergence`](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.DivergenceF2C) and $UB$ is the [`center-to-face upwind biased product`](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.UpwindBiasedProductC2F) operator.
- Tendency 2:

  $$D = \partial(UB) + \textrm{fcc}(v, \theta),$$  
  
  where $\textrm{fcc}(v, \theta)$ is the [`center-to-center flux correction`](https://github.com/CliMA/ClimaCore.jl/blob/main/src/Operators/finitedifference.jl#L2617) operator.
- Tendency 3:

  $$D = A,$$  
   
  where $A$ is the [`center-to-center vertical advection`](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.AdvectionC2C) operator.
- Tendency 4:

  $$D = A + \textrm{fcc}(v, \theta),$$  
  
  where $\textrm{fcc}(v, \theta)$ is the [`center-to-center flux correction`](https://github.com/CliMA/ClimaCore.jl/blob/main/src/Operators/finitedifference.jl#L2617) operator.

#### Set Up

This test case is set up in a 1D column domain ``z \in [0, 4\pi]``, discretized into a mesh of 128 elements. The velocity field is defined as a sinusoidal wave. The boundary conditions are operator dependent, so they depend on the tendency. 
* For tendencies 1 and 2 where the upwind biased operator ``UB`` is used, the left boundary is defined as ``sin(a - t)``. The right boundary is ``sin(b - t)``. Here ``a`` and ``b`` are the left and right bounds of the domain. 
* For tendencies 3 and 4, where the advection operator ``A`` is used, the left boundary is defined as ``sin(-t)``.  The right boundary is extrapolated, meaning its value is set to the closest interior point.

### Ekman Layer

The 1D vertical Ekman layer simulation in [`examples/column/ekman.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/column/ekman.jl) demonstrates the simulation of atmospheric boundary layer dynamics, specifically the Ekman spiral phenomenon resulting from the balance between Coriolis force and vertical diffusion.

#### Equations and discretizations

#### Momentum

Follows the momentum equations with vertical diffusion and Coriolis force:
```math
\begin{align}
\frac{\partial u}{\partial t} &= \frac{\partial}{\partial z}\left(\nu \frac{\partial u}{\partial z}\right) + f(v - v_g) - w\frac{\partial u}{\partial z} \\
\frac{\partial v}{\partial t} &= \frac{\partial}{\partial z}\left(\nu \frac{\partial v}{\partial z}\right) - f(u - u_g) - w\frac{\partial v}{\partial z}
\end{align}
```

These are discretized using the following:
```math
\begin{align}
\frac{\partial u}{\partial t} &\approx D \left(\nu G(u)\right) + f(v - v_g) - A(w, u) \\
\frac{\partial v}{\partial t} &\approx D \left(\nu G(v)\right) - f(u - u_g) - A(w, v)
\end{align}
```

#### Prognostic variables

* `u`: _horizontal velocity (zonal or east-west component)_ measured in m/s.
* `v`: _horizontal velocity (meridional or north-south component)_ measured in m/s.
* `w`: _vertical velocity_ measured in m/s (manually set to zero in this example to disable vertical advection effects).

#### Differentiation operators

Because this is a 1D vertical problem, we utilize the staggered vertical grid with:

- ``D`` is the [face-to-center divergence](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.DivergenceF2C) operator, called `divf2c` in the example code
- ``G`` is the [center-to-face gradient](https://clima.github.io/ClimaCore.jl/dev/operators/#ClimaCore.Operators.GradientC2F) operator, called `gradc2f` in the example code
- ``A`` is the [center-to-center vertical advection](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.AdvectionC2C) operator, called `A` in the example code.

#### Problem flow and set-up

This test case is set up in a vertical column domain $[0, L]$ with height $L = 200\,\mathrm{m}$. The Coriolis parameter is set to $f = 5 \times 10^{-5}\,\mathrm{s}^{-1}$, the viscosity coefficient to $\nu = 0.01\,\mathrm{m}^2/\mathrm{s}$, and the geostrophic wind to $(u_g, v_g) = (1.0, 0.0)\,\mathrm{m/s}$.

The Ekman depth parameter is calculated as  
$d = \sqrt{\dfrac{2\nu}{f}}$,  
which determines the characteristic depth of the boundary layer.

The initial condition is set to a uniform wind profile equal to the geostrophic wind throughout the vertical column. The simulation is run for 50 hours to allow the boundary layer to develop fully.

Boundary conditions are applied as follows:  
- At the top boundary: wind velocity equals the geostrophic wind $(u_g, v_g)$  
- At the bottom boundary: drag condition proportional to the wind speed, where the surface stress is  
  $C_d \, |\mathbf{U}| \, u$ and $C_d \, |\mathbf{U}| \, v$

#### Application of boundary conditions

The application of boundary conditions is implemented through the operators:

1. For the u-momentum equation:
   - Top boundary: `Operators.SetValue(FT(ug))` sets $u$ to the geostrophic value
   - Bottom boundary: `Operators.SetValue(Geometry.WVector(Cd * u_wind * u_1))` applies the drag condition

2. For the v-momentum equation:
   - Top boundary: `Operators.SetValue(FT(vg))` sets $v$ to the geostrophic value
   - Bottom boundary: `Operators.SetValue(Geometry.WVector(Cd * u_wind * v_1))` applies the drag condition

The example verifies the numerical solution by comparing it to the analytical solution:
```math
\begin{align}
u(z) &= u_g - e^{-z/d}(u_g\cos(z/d) + v_g\sin(z/d)) \\
v(z) &= v_g + e^{-z/d}(u_g\sin(z/d) - v_g\cos(z/d))
\end{align}
```

The results demonstrate accurate capture of the characteristic Ekman spiral, where wind speed increases with height and wind direction rotates with increasing height until it aligns with the geostrophic wind.


### Hydrostatic Balance

The 1D Column hydrostatic balance example in [`examples/column/hydrostatic.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/column/hydrostatic.jl) demonstrates the setup and maintenance of hydrostatic balance in a single vertical column using a finite difference discretization.

#### Equations and discretizations

##### Mass and Potential Temperature

The system maintains hydrostatic balance while solving the continuity equations for density and potential temperature density:

```math
\begin{equation}
  \frac{\partial \rho}{\partial t} = - \nabla \cdot(\rho \boldsymbol{w})
\label{eq:1d-column-continuity}
\end{equation}
```

```math
\begin{equation}
  \frac{\partial \rho\theta}{\partial t} = - \nabla \cdot(\rho\theta \boldsymbol{w})
\label{eq:1d-column-theta}
\end{equation}
```

These are discretized using the following:

```math
\begin{equation}
  \frac{\partial \rho}{\partial t} \approx - D^c_v[I^f(\rho) \boldsymbol{w}]
\label{eq:1d-column-discrete-continuity}
\end{equation}
```

```math
\begin{equation}
  \frac{\partial \rho\theta}{\partial t} \approx - D^c_v[I^f(\rho\theta) \boldsymbol{w}]
\label{eq:1d-column-discrete-theta}
\end{equation}
```

#### Vertical Momentum

The vertical momentum follows:

```math
\begin{equation}
  \frac{\partial \boldsymbol{w}}{\partial t} = -I^f\left(\frac{\rho\theta}{\rho}\right) \nabla^f_v \Pi(\rho\theta) - \nabla^f_v \Phi(z)
\label{eq:1d-column-momentum}
\end{equation}
```

Where:
- $\Pi(\rho\theta) = C_p \left(\frac{R_d \rho\theta}{p_0}\right)^{\frac{R_m}{C_v}}$ is the Exner function
- $\Phi(z) = gz$ is the geopotential

This is discretized using the following:

```math
\begin{equation}
  \frac{\partial \boldsymbol{w}}{\partial t} \approx B\left(-I^f\left(\frac{\rho\theta}{\rho}\right) \nabla^f_v \Pi(\rho\theta) - \nabla^f_v \Phi(z)\right)
\label{eq:1d-column-discrete-momentum}
\end{equation}
```

Where $B$ applies boundary conditions to enforce $\boldsymbol{w} = 0$ at the domain boundaries.

#### Prognostic variables

**$\rho$**: *Density*, measured in kg/m³, discretized at cell centers.

**$\rho\theta$**: *Potential temperature density*, measured in K·kg/m³, discretized at cell centers.

**$\boldsymbol{w}$**: *Vertical velocity*, measured in m/s, discretized at cell faces.

#### Operators

##### Reconstructions

**$I^f$** is the [center-to-face reconstruction](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F) operator, called `If` in the example code.  
Currently this is implemented as the arithmetic mean.

##### Differentiation operators

**$D^c_v$** is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `∂` in the example code.  
This example uses zero vertical velocity at the top and bottom boundaries.

**$\nabla^f_v$** is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F), called `∂f` in the example code.

**$B$** is the [boundary operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.SetBoundaryOperator), called `B` in the example code.  
This enforces zero vertical velocity at domain boundaries.


#### Problem flow and set-up

This test case is set up in a vertical column domain from $z=0$ m to $z=30$ km with 30 vertical elements. The column is initialized with a decaying temperature profile, where:

- The virtual temperature starts at $T_{\textrm{virt}\_\textrm{surf}} = 280$ K at the surface
- It asymptotically approaches $T_{\textrm{min}\_\textrm{ref}} = 230$ K with height
- The profile follows a hyperbolic tangent function with height
- The pressure follows a hydrostatic balance equation
- Density is calculated from the equation of state using virtual temperature and pressure

The initial vertical velocity is set to zero everywhere. To maintain hydrostatic balance, the discrete form computes iteratively the values of density that ensure the vertical pressure gradient balances gravity.

The simulation is run for 10 days to verify that the hydrostatic balance is maintained over time. Results are plotted showing density ($\rho$), vertical velocity ($\boldsymbol{w}$), and potential temperature density ($\rho\theta$) profiles.


## 2D Cartesian examples

### Flux Limiters advection

The 2D Cartesian advection/transport example in [`examples/plane/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples/plane/limiters_advection.jl) demonstrates the application of flux limiters in the horizontal direction, namely [`QuasiMonotoneLimiter`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Limiters.QuasiMonotoneLimiter), in a 2D Cartesian domain.

#### Equations and discretizations

#### Mass

Follows the continuity equation
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
\label{eq:2d-plane-advection-lim-continuity}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho \approx - wD[ \rho \boldsymbol{u}] .
\label{eq:2d-plane-advection-lim-discrete-continuity}
\end{equation}
```

#### Tracers

For the tracer concentration per unit mass ``q``, the tracer density (scalar) ``\rho q`` follows the advection/transport equation

```math
\begin{equation}
  \frac{\partial}{\partial t} \rho q = - \nabla \cdot(\rho q \boldsymbol{u})  + g(\rho, q).
\label{eq:2d-plane-advection-lim-tracers}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
\frac{\partial}{\partial t} \rho q \approx - wD[ \rho q \boldsymbol{u}] + g(\rho, q),
\label{eq:2d-plane-advection-lim-discrete-tracers}
\end{equation}
```
where ``g(\rho, q) = - \nu_4 [\nabla^4_h (\rho q)]`` represents the horizontal hyperdiffusion operator, with ``\nu_4`` (measured in m^4/s) the hyperviscosity constant coefficient (set equal to zero by default in the example).

Currently tracers are only treated explicitly in the time discretization.


#### Prognostic variables

* ``\rho``: _density_ measured in kg/m³.
* ``\boldsymbol{u}`` _velocity_, a vector measured in m/s. Since this is a 2D problem, ``\boldsymbol{u} \equiv \boldsymbol{u}_h``.
* ``\rho q``: the tracer density scalar, where ``q`` is the tracer concentration per unit mass.

#### Differentiation operators

Because this is a purely 2D problem, there is no staggered vertical discretization, hence, there is no need of specifying variables at cell centers, faces or to reconstruct from faces to centers and vice versa.

- ``wD`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `wdiv` in the example code.
- ``G`` is the [discrete horizontal strong spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `grad` in the example code.

To discretize the hyperdiffusion operator, ``g(\rho, q) = - \nu_4 [\nabla^4 (\rho q)]``, in the horizontal direction, we compose the horizontal weak divergence, ``wD``, and the horizontal gradient operator, ``G_h``, twice, with an intermediate call to [`weighted_dss!`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Spaces.weighted_dss!) between the two compositions, as in ``[g_2(\rho, g) \circ DSS(\rho, q) \circ g_1(\rho, q)]``, with:
- ``g_1(\rho, q) = wD(G_h(q))``
- ``DSS(\rho, q) = DSS(g_1(\rho q))``
- ``g_2(\rho, q) = -\nu_4 wD(\rho G_h(\rho q))``
    - with ``\nu_4`` the hyperviscosity coefficient.

#### Problem flow and set-up

This test case is set up in a Cartesian planar domain `` [-2 \pi, 2 \pi]^2 ``, doubly periodic.

The flow was chosen to be a horizontal uniform rotation. Moreover, the flow is reversed halfway through the time period so that the tracer blobs go back to its initial configuration (using the same speed scaling constant which was derived to account for the distance travelled in all directions in a half period).

```math
\begin{align}
    u &= -u_0 (y - c_y) \cos(\pi t / T_f) \nonumber \\
    v &= u_0 (x - c_x) \cos(\pi t / T_f)
\label{eq:2d-plane-advection-lim-flow}
\end{align}
```
where ``u_0 = \pi / 2`` is the speed scaling factor to have the flow reversed halfway through the time period, ``\boldsymbol{c} = (c_x, c_y)`` is the center of the rotational flow, which coincides with the center of the domain, and `` T_f = 2 \pi`` is the final simulation time, which coincides with the temporal period to have a full rotation.

This example is set up to run with three possible initial conditions:
- `cosine_bells`
- `gaussian_bells`
- `cylinders`: two 2D slotted cylinders (test case available in the literature, cfr: [GubaOpt2014](@cite)).

#### Application of Flux Limiters

Because this is a fully 2D problem, the application of limiters does not affect the order of operations, which is implemented as follows:

1. Horizontal trasport with hyperdiffusion (with weak divergence ``wD``)
2. Horizontal flux limiters
3. DSS
## 3D Cartesian examples

### Flux Limiters advection

The 3D Cartesian advection/transport example in [`examples/hybrid/box/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples/hybrid/box/limiters_advection.jl) demonstrates the application of flux limiters in the horizontal direction, namely [`QuasiMonotoneLimiter`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Limiters.QuasiMonotoneLimiter), in a hybrid Cartesian domain. It also demonstrates the usage of the high-order upwinding scheme in the vertical direction, called [`Upwind3rdOrderBiasedProductC2F`](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F).

#### Equations and discretizations

#### Mass

Follows the continuity equation
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
\label{eq:3d-box-advection-lim-continuity}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho \approx - D_h[ \rho (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] - D^c_v[I^f(\rho \boldsymbol{u}_h)) + I^f(\rho) \boldsymbol{u}_v)] .
\label{eq:3d-box-advection-lim-discrete-continuity}
\end{equation}
```

#### Tracers

For the tracer concentration per unit mass ``q``, the tracer density (scalar) ``\rho q`` follows the advection/transport equation

```math
\begin{equation}
  \frac{\partial}{\partial t} \rho q = - \nabla \cdot(\rho q \boldsymbol{u})  + g(\rho, q).
\label{eq:3d-box-advection-lim-tracers}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
\frac{\partial}{\partial t} \rho q \approx
- D_h[ \rho q (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))]
- D^c_v\left[I^f(\rho q) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho q}{\rho} \right) \right] + g(\rho, q),
\label{eq:3d-box-advection-lim-discrete-tracers}
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

- ``D_h`` is the [discrete horizontal strong spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence), called `hdiv` in the example code.
- ``wD_h`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `hwdiv` in the example code.
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `vdivf2c` in the example code.
  - This example uses advective fluxes equal to zero at the top and bottom boundaries.
- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `hgrad` in the example code.

To discretize the hyperdiffusion operator, ``g(\rho, q) = - \nu_4 [\nabla^4 (\rho q)]``, in the horizontal direction, we compose the horizontal weak divergence, ``wD_h``, and the horizontal gradient operator, ``G_h``, twice, with an intermediate call to [`weighted_dss!`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Spaces.weighted_dss!) between the two compositions, as in ``[g_2(\rho, g) \circ DSS(\rho, q) \circ g_1(\rho, q)]``, with:
- ``g_1(\rho, q) = wD_h(G_h(q))``
- ``DSS(\rho, q) = DSS(g_1(\rho q))``
- ``g_2(\rho, q) = -\nu_4 wD_h(\rho G_h(\rho q))``
    - with ``\nu_4`` the hyperviscosity coefficient.

#### Application of Flux Limiters

!!! note

    Since we use flux limiters that limit only operators defined in the spectral space (i.e., they are applied level-wise in the horizontal direction), the application of limiters has to follow a precise order in the sequence of operations that specifies the total tendency.

The order of operations should be the following:

1. Horizontal transport (with strong divergence ``D_h``)
2. Horizontal Flux Limiters
3. Horizontal hyperdiffusion (with weak divergence ``wD_h``)
4. Vertical transport
5. DSS
#### Problem flow and set-up

This test case is set up in a Cartesian (box) domain `` [-2 \pi, 2 \pi]^2 \times [0, 4 \pi] ~\textrm{m}^3``, doubly periodic in the horizontal direction, but not in the vertical direction.

The flow was chosen to be a spiral, i.e., so to have a horizontal uniform rotation, and a vertical velocity ``\boldsymbol{u}_v \equiv w = 0`` at the top and bottom boundaries, and ``\boldsymbol{u}_v \equiv w = 1`` in the center of the domain. Moreover, the flow is reversed in all directions halfway through the time period so that the tracer blobs go back to its initial configuration (using the same speed scaling constant which was derived to account for the distance travelled in all directions in a half period).

```math
\begin{align}
    u &= -u_0 (y - c_y) \cos(\pi t / T_f) \nonumber \\
    v &= u_0 (x - c_x) \cos(\pi t / T_f) \nonumber \\
    w &= u_0 \sin(\pi z / z_m) \cos(\pi t / T_f) \nonumber
\label{eq:3d-box-advection-lim-flow}
\end{align}
```
where ``u_0 = \pi / 2`` is the speed scaling factor to have the flow reversed halfway through the time period, ``\boldsymbol{c} = (c_x, c_y)`` is the center of the rotational flow, which coincides with the center of the domain,  ``z_m = 4 \pi`` is the maximum height of the domain, and `` T_f = 2 \pi`` is the final simulation time, which coincides with the temporal period to have a full rotation in the horizontal direction.

This example is set up to run with three possible initial conditions:
- `cosine_bells`
- `gaussian_bells`
- `slotted_spheres`: a slight modification of the 2D slotted cylinder test case available in the literature (cfr: [GubaOpt2014](@cite)).

#### Application of Flux Limiters

Because this is a Cartesian 3D problem, the application of limiters does not affect the order of operations, which is implemented as follows:

1. Horizontal transport + hyperdiffusion (with weak divergence ``wD_h``)
2. Horizontal flux limiters
3. Vertical transport
4. DSS

## 2D Sphere examples

### Flux Limiters advection

The 2D sphere advection/transport example in [`examples/sphere/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples/sphere/limiters_advection.jl) demonstrates the application of flux limiters in the horizontal direction, namely [`QuasiMonotoneLimiter`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Limiters.QuasiMonotoneLimiter), in a 2D spherical domain.

#### Equations and discretizations

#### Mass

Follows the continuity equation
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
\label{eq:2d-sphere-advection-lim-continuity}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho \approx - wD[ \rho \boldsymbol{u}] .
\label{eq:2d-sphere-advection-lim-discrete-continuity}
\end{equation}
```

#### Tracers

For the tracer concentration per unit mass ``q``, the tracer density (scalar) ``\rho q`` follows the advection/transport equation

```math
\begin{equation}
  \frac{\partial}{\partial t} \rho q = - \nabla \cdot(\rho q \boldsymbol{u})  + g(\rho, q).
\label{eq:2d-sphere-advection-lim-tracers}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
\frac{\partial}{\partial t} \rho q \approx - wD[ \rho q \boldsymbol{u}] + g(\rho, q),
\label{eq:2d-sphere-advection-lim-discrete-tracers}
\end{equation}
```
where ``g(\rho, q) = - \nu_4 [\nabla^4_h (\rho q)]`` represents the horizontal hyperdiffusion operator, with ``\nu_4`` (measured in m^4/s) the hyperviscosity constant coefficient.

Currently tracers are only treated explicitly in the time discretization.


#### Prognostic variables

* ``\rho``: _density_ measured in kg/m³.
* ``\boldsymbol{u}`` _velocity_, a vector measured in m/s. Since this is a 2D problem, ``\boldsymbol{u} \equiv \boldsymbol{u}_h``.
* ``\rho q``: the tracer density scalar, where ``q`` is the tracer concentration per unit mass.

#### Differentiation operators

Because this is a purely 2D problem, there is no staggered vertical discretization, hence, there is no need of specifying variables at cell centers, faces or to reconstruct from faces to centers and vice versa.

- ``wD`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `wdiv` in the example code.
- ``G`` is the [discrete horizontal strong spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `grad` in the example code.

To discretize the hyperdiffusion operator, ``g(\rho, q) = - \nu_4 [\nabla^4 (\rho q)]``, in the horizontal direction, we compose the horizontal weak divergence, ``wD``, and the horizontal gradient operator, ``G_h``, twice, with an intermediate call to [`weighted_dss!`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Spaces.weighted_dss!) between the two compositions, as in ``[g_2(\rho, g) \circ DSS(\rho, q) \circ g_1(\rho, q)]``, with:
- ``g_1(\rho, q) = wD(G_h(q))``
- ``DSS(\rho, q) = DSS(g_1(\rho q))``
- ``g_2(\rho, q) = -\nu_4 wD(\rho G_h(\rho q))``
    - with ``\nu_4`` the hyperviscosity coefficient.

#### Problem flow and set-up

This test case is set up in a Cartesian planar domain `` [-2 \pi, 2 \pi]^2 ``, doubly periodic.

The flow was chosen to be a horizontal uniform rotation. Moreover, the flow is reversed halfway through the time period so that the tracer blobs go back to its initial configuration (using the same speed scaling constant which was derived to account for the distance travelled in all directions in a half period).

```math
\begin{align}
    u &= k sin (\lambda)^2  sin (2  \phi)  \cos(\pi  t / T_f) +
       \frac{2 \pi}{T_f} \cos (\phi) \nonumber \\
    v &= k \sin (2  \lambda) \cos (\phi) \cos(\pi t / T_f)
\label{eq:2d-sphere-lim-flow}
\end{align}
```
where ``u_0 = 2 \pi R / T_f`` is the speed scaling factor to have the flow reversed halfway through the time period, `` T_f = 86400 * 12`` (i.e., ``12`` days in seconds) is the final simulation time, which coincides with the temporal period to have a full rotation around the sphere of radius ``R``.

This example is set up to run with three possible initial conditions:
- `cosine_bells`
- `gaussian_bells`
- `cylinders`: two 2D slotted cylinders (test case available in the literature, cfr: [GubaOpt2014](@cite)).

#### Application of Flux Limiters

Because this is a fully 2D problem, the application of limiters does not affect the order of operations, which is implemented as follows:

1. Horizontal trasport with hyperdiffusion (with weak divergence ``wD``)
2. Horizontal flux limiters
3. DSS

### Shallow-water equations

The shallow water equations in the so-called *vector invariant form* from [Bao2014](@cite) are:


```math
\begin{align}
  \frac{\partial h}{\partial t} + \nabla \cdot (h u) &= 0\\
  \frac{\partial u_i}{\partial t} + \nabla (\Phi + \tfrac{1}{2}\|u\|^2)_i  &= (\boldsymbol{u} \times (f + \nabla \times \boldsymbol{u}))_i
\label{eq:shallow-water}
\end{align}
```
where ``f`` is the Coriolis term and ``\Phi = g(h+h_s)``, with ``g`` the gravitational accelration constant, ``h`` the (free) height of the fluid and  ``h_s`` a non-uniform reference surface.

To the above set of equations, we allow the uset to add a hyperdiffusion operator, ``g(h, \boldsymbol{u}) = - \nu_4 [\nabla^4 (h, \boldsymbol{u})]``, with ``\nu_4`` (measured in m^4/s) the hyperviscosity constant coefficient. In the hyperdiffusion expression, ``\nabla^4`` represents a biharmonic operator, and it assumes a different formulation on curvilinear reference systems, depending on it being applied to a scalar field, such as ``h``, or a vector field, such as ``\boldsymbol{u}``.


The governing equations then become:

```math
\begin{align}
  \frac{\partial h}{\partial t} + \nabla \cdot (h u) &= g(h, \boldsymbol{u})\\
  \frac{\partial u_i}{\partial t} + \nabla (\Phi + \tfrac{1}{2}\|u\|^2)_i  &= (\boldsymbol{u} \times (f + \nabla \times \boldsymbol{u}))_i + g(h, \boldsymbol{u})
\label{eq:shallow-water-with-hyperdiff}
\end{align}
```

Since this is a 2D problem (with related 2D vector field), the curl is defined to be

```math
\begin{align}
 \omega^i = (\nabla \times u)^i &=
    \begin{cases}
        0 &\text{ if $i =1,2$},\\
        \frac{1}{J} \left[ \frac{\partial u_2}{\partial \xi^1} - \frac{\partial u_1}{\partial \xi^2} \right] &\text{ if $i=3$},
    \end{cases}
\label{eq:2Dvorticity}
\end{align}
```
where we have used the coordinate system in each 2D reference element, i.e., ``(\xi^1, \xi^2) \in [-1,1]\times[-1,1]``. Similarly, if additionally ``v^1 = v^2 = 0``, then

```math
\begin{align}
   (\boldsymbol{u} \times \boldsymbol{v})_i =
    \begin{cases}
          J u^2 v^3 &\text{ if $i=1$},\\
        - J u^1 v^3 &\text{ if $i=2$},\\
        0 &\text{ if $i=3$}.
    \end{cases}
\end{align}
```

Hence, we can rewrite equations \eqref{eq:shallow-water} using the velocity representation in covariant coordinates, in this case ``u = u_1 \boldsymbol{b}^1 + u_2 \boldsymbol{b}^2 + 0\boldsymbol{b}^3``, and ``g(h, \boldsymbol{u}) = 0 `` for simplicity, as:

```math
\begin{align}
    \frac{\partial h}{\partial t} + \frac{1}{J}\frac{\partial}{\partial \xi^j}\Big(h J u^j\Big) &= 0\\
    \frac{\partial u_i}{\partial t} + \frac{\partial}{\partial \xi^i} (\Phi + \tfrac{1}{2}\|u\|^2)  &= E_{ijk}u^j (f^k + \omega^k) .
\label{eq:covariant-shallow-water}
\end{align}
```

#### Prognostic variables

* ``h``: scalar _height_ field of the fluid, measured in m.
* ``\boldsymbol{u}`` _velocity_, a 2D vector measured in m/s.
#### Differentiation operators

Because this is a purely 2D problem, there is no staggered vertical discretization, hence, there is no need of specifying variables at cell centers, faces or to reconstruct from faces to centers and vice versa.

- ``D`` is the [discrete horizontal strong spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence), called `div` in the example code.
- ``wD`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `wdiv` in the example code.
- ``G`` is the [discrete horizontal strong spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `grad` in the example code.
- ``wG`` is the [discrete horizontal weak spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakGradient), called `wgrad` in the example code.
- ``Curl`` is the [discrete curl](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl), called `curl` in the example code.
- ``wCurl`` is the [discrete weak curl](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakCurl), called `wcurl` in the example code.


To discretize the hyperdiffusion operator, ``g(h, \boldsymbol{u}) = - \nu_4 [\nabla^4 (h, \boldsymbol{u})]``, in the horizontal direction, we compose the weak divergence, ``wD``, and the gradient operator, ``G``, twice, with an intermediate call to [`weighted_dss!`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Spaces.weighted_dss!) between the two compositions, as in ``[g_2(h, \boldsymbol{u}) \circ DSS(h, \boldsymbol{u}) \circ g_1(h, \boldsymbol{u})]``. Moreover, when ``g(h, \boldsymbol{u}) = - \nu_4 [\nabla^4 (h)]``, i.e., the operator is applied to a scalar field only, it is discretized composing the following operations:
- ``g_1(h) = wD(G(h))``
- ``DSS(g_1(h))``
- ``g_2(h) = -\nu_4 wD(G(h))``
whereas, when the operator is applied to a vector field, i.e., ``g(h, \boldsymbol{u}) = - \nu_4 [\nabla^4 (\boldsymbol{u})]``, it is discretized as:
- ``g_1(h, \boldsymbol{u}) = wG(D(\boldsymbol{u})) - wCurl(Curl(\boldsymbol{u}))``
- ``DSS(h, \boldsymbol{u}) = DSS(g_1(h, \boldsymbol{u}))``
- ``g_2(h, \boldsymbol{u}) = -\nu_4 \left[ wG(D(\boldsymbol{u})) - wCurl(Curl(\boldsymbol{u})) \right]``

In both cases, ``\nu_4`` is the hyperviscosity coefficient.

#### Problem flow and set-up

This test case is set up on a 2D (surface) spherical domain represented by a cubed-sphere manifold.

This suite of examples contains five different test cases:
- One, invoked via the command-line argument `steady_state`, which reproduces Test Case 2 in [Williamson1992](@cite). This test case gives the steady-state solution to the non-linear shallow water equations. It consists of a solid body rotation or zonal flow with the corresponding geostrophic height field. The Coriolis parameter is a function of latitude and longitude so the flow can be specified with the spherical coordinate poles not necessarily coincident with Earth's rotation axis. Hence, this test case can be run with a specified command-line argument for the angle ``\alpha`` that represents the angle between the north pole and the center of the top cube panel of the cubed-sphere geometry.
- A second one, invoked via the command-line argument `steady_state_compact`, reproduces Test Case 3 in [Williamson1992](@cite). This test case gives the steady-state solution to the non-linear shallow water equations with nonlinear zonal geostrophic flow with compact support.
- A third one, invoked via the command-line argument `mountain`, reproduces Test Case 5 in [Williamson1992](@cite). It represents a zonal flow over an isolated mountain, where the governing equations describe a global steady-state nonlinear zonal geostrophic flow, with a corresponding geostrophic height field over a non-uniform reference surface `h_s`.
- A fourth one, invoked via the command-line argument `rossby_haurwitz`, reproduces Test Case 6 in [Williamson1992](@cite). It represents the solution of the nonlinear barotropic vorticity equation on the sphere.
- A fifth one, invoked via the command-line argument `barotropic_instability`, reproduces the test case in [Galewsky2004](@cite) (also in Sec. 7.6 in [Ullrich2010](@cite)). This test case consists of a zonal jet with compact support at a latitude of ``45°``. A small height disturbance is then added, which causes the jet to become unstable and collapse into a highly vortical structure.


## 3D Sphere examples

### Deformation Flow with Flux Limiters

The 3D sphere advection/transport example in [`examples/hybrid/sphere/deformation_flow.jl`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples/hybrid/sphere/deformation_flow.jl) demonstrates the application of flux limiters in the horziontal direction, namely [`QuasiMonotoneLimiter`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Limiters.QuasiMonotoneLimiter), in a hybrid 3D spherical domain. It also demonstrates the usage of the flux-corrected transport in the vertical direction; by default, it uses `FCTZalesak`.

#### Equations and discretizations

The original test case (without limiters or flux-corrected transport) is specified in Section 1.1 of [Ullrich2012DynamicalCM](@cite).

#### Mass

Follows the continuity equation
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
\label{eq:3d-sphere-lim-continuity}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
  \frac{\partial}{\partial t} \rho \approx - D_h[\rho \boldsymbol{u}^c] - D^c_v[I^f(\rho) \boldsymbol{u}^f].
\label{eq:3d-sphere-lim-discrete-continuity}
\end{equation}
```

#### Tracers

This test case has five different tracer concentrations per unit mass ``q_i``, hence five different tracer densities (scalar) ``\rho q_i``. They all follow the same advection/transport equation

```math
\begin{equation}
  \frac{\partial}{\partial t} \rho q = - \nabla \cdot(\rho q \boldsymbol{u})  + g(\rho, q).
\label{eq:3d-sphere-lim-tracers}
\end{equation}
```

This is discretized using the following
```math
\begin{equation}
\frac{\partial}{\partial t} \rho q \approx
- D_h[ \rho q \boldsymbol{u}^c]
- D^c_v\left[I^f(\rho q) * \boldsymbol{u}^f_h + FCT^f\left( \boldsymbol{u}^f_v, \frac{\rho q}{\rho} \right) \right] + g(\rho, q),
\label{eq:3d-sphere-lim-discrete-tracers}
\end{equation}
```
where ``g(\rho, q) = - \nu_4 [\nabla^4_h (\rho q)]`` represents the horizontal hyperdiffusion operator, with ``\nu_4`` (measured in m^4/s) the hyperviscosity constant coefficient.

Currently tracers are only treated explicitly in the time discretization.


#### Prognostic variables

* ``\rho``: _density_ measured in kg/m³. This is discretized at cell centers.
* ``\boldsymbol{u}`` _velocity_, a vector measured in m/s. This is discretized via ``\boldsymbol{u} = \boldsymbol{u}_h + \boldsymbol{u}_v`` where
  - ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
  - ``\boldsymbol{u}_v = u_3 \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
* ``\rho q_i``: tracer density scalars, where ``q_i`` is a tracer concentration per unit mass, are stored at cell centers.

#### Operators

We make use of the following operators

#### Reconstructions

* ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C), called `If2c` in the example code.
* ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F), called `Ic2f` in the example code.
  - Currently this is just the arithmetic mean, but we will need to use a weighted version with stretched vertical grids.
* ``FCT^f`` denotes either the [center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F) (which represents no flux-corrected transport), the center-to-face Boris & Book FCT operator, or the center-to-face Zalesak FCT operator.


#### Differentiation operators

- ``D_h`` is the [discrete horizontal strong spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence), called `hdiv` in the example code.
- ``wD_h`` is the [discrete horizontal weak spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence), called `hwdiv` in the example code.
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `vdivf2c` in the example code.
  - This example uses advective fluxes equal to zero at the top and bottom boundaries.
- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient), called `hgrad` in the example code.

To discretize the hyperdiffusion operator for each tracer concentration, ``g(\rho, q_i) = - \nu_4 [\nabla^4 (\rho q_i)]``, in the horizontal direction, we compose the horizontal weak divergence, ``wD_h``, and the horizontal gradient operator, ``G_h``, twice, with an intermediate call to [`weighted_dss!`](https://clima.github.io/ClimaCore.jl/stable/api/#ClimaCore.Spaces.weighted_dss!) between the two compositions, as in ``[g_2(\rho, g) \circ DSS(\rho, q) \circ g_1(\rho, q_i)]``, with:
- ``g_1(\rho, q_i) = wD_h(G_h(q_i))``
- ``DSS(\rho, q_i) = DSS(g_1(\rho q_i))``
- ``g_2(\rho, q_i) = -\nu_4 wD_h(\rho G_h(\rho q_i))``
    - with ``\nu_4`` the hyperviscosity coefficient.

#### Application of Flux Limiters

!!! note

    Since we use flux limiters that limit only operators defined in the spectral space (i.e., they are applied level-wise in the horizontal direction), the application of limiters has to follow a precise order in the sequence of operations that specifies the total tendency.

The order of operations should be the following:

1. Horizontal transport (with strong divergence ``D_h``)
2. Horizontal flux limiters
3. Horizontal hyperdiffusion (with weak divergence ``wD_h``)
4. Vertical transport
5. DSS

#### Problem flow and set-up

This test case is set up in a 3D (shell) spherical domain where the elevation goes from ``z=0~\textrm{m}`` (i.e., from the radius of the sphere ``R = 6.37122 10^6~\textrm{m}``) to ``z_{\textrm{top}} = 12000~\textrm{m}``.

The flow (reversed halfway through the time period) is specified as ``\boldsymbol{u} = \boldsymbol{u}_a + \boldsymbol{u}_d``, where the components are defined as follows:

```math
\begin{align}
  u_a &= k \sin (\lambda')^2  \sin (2  \phi)  \cos(\pi  t / \tau) +
      \frac{2 \pi R}{\tau} \cos (\phi) \nonumber \\
  v_a &= k \sin (2  \lambda') \cos (\phi) \cos(\pi t / \tau) \nonumber \\
  u_d &= \frac{\omega_0  R}{ b / p_{\textrm{top}}} \cos (\lambda') \cos(\phi)^2 \cos(2 \pi t / \tau) \left[-exp \left( \frac{(p - p_0)}{ b p_{\textrm{top}}} \right) + exp \left( \frac{(p_{\textrm{top}} - p(zc))}{b p_{\textrm{top}}} \right) \right] \nonumber
\label{eq:3d-sphere-lim-flow}
\end{align}
```

where all values of the parameters can be found in Table 1.1 in the reference [Ullrich2012DynamicalCM](@cite).
