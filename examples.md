# Examples

## 1D Column examples

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
\frac{\partial u}{\partial t} &\approx D_{f2c}\left(\nu G_{c2f}(u)\right) + f(v - v_g) - A(w, u) \\
\frac{\partial v}{\partial t} &\approx D_{f2c}\left(\nu G_{c2f}(v)\right) - f(u - u_g) - A(w, v)
\end{align}
```

#### Prognostic variables

* `u`: _horizontal velocity (east-west component)_ measured in m/s.
* `v`: _horizontal velocity (north-south component)_ measured in m/s.
* `w`: _vertical velocity_ measured in m/s (set to zero in this example).

#### Differentiation operators

Because this is a 1D vertical problem, we utilize the staggered vertical grid with:

- `G_{c2f}` is the [gradient operator from cell centers to faces](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F), called `gradc2f` in the example code.
- `D_{f2c}` is the [divergence operator from faces to centers](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `divf2c` in the example code.
- `A` is the [advection operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.AdvectionC2C), called `A` in the example code (though vertical advection is set to zero in this example).

#### Problem flow and set-up

This test case is set up in a vertical column domain `[0, L]` with height `L = 200m`. The Coriolis parameter is set to `f = 5e-5 s⁻¹`, the viscosity coefficient to `ν = 0.01 m²/s`, and the geostrophic wind to `(u_g, v_g) = (1.0, 0.0) m/s`.

The Ekman depth parameter is calculated as `d = sqrt(2 * ν / f)`, which determines the characteristic depth of the boundary layer.

The initial condition is set to a uniform wind profile equal to the geostrophic wind throughout the vertical column. The simulation is run for 50 hours to allow the boundary layer to develop fully.

Boundary conditions are applied as follows:
- At the top boundary: wind velocity equals the geostrophic wind (`u_g`, `v_g`)
- At the bottom boundary: drag condition proportional to the wind speed, where the surface stress is `Cd * |U| * u` and `Cd * |U| * v`

#### Application of boundary conditions

The application of boundary conditions is implemented through the operators:

1. For the u-momentum equation:
   - Top boundary: `Operators.SetValue(FT(ug))` sets u to the geostrophic value
   - Bottom boundary: `Operators.SetValue(Geometry.WVector(Cd * u_wind * u_1))` applies the drag condition

2. For the v-momentum equation:
   - Top boundary: `Operators.SetValue(FT(vg))` sets v to the geostrophic value
   - Bottom boundary: `Operators.SetValue(Geometry.WVector(Cd * u_wind * v_1))` applies the drag condition

3. DSS (Direct Stiffness Summation) is applied implicitly through the operators to ensure numerical stability at element boundaries

The example verifies the numerical solution by comparing it to the analytical solution:
```math
\begin{align}
u(z) &= u_g - e^{-z/d}(u_g\cos(z/d) + v_g\sin(z/d)) \\
v(z) &= v_g + e^{-z/d}(u_g\sin(z/d) - v_g\cos(z/d))
\end{align}
```

The results demonstrate accurate capture of the characteristic Ekman spiral, where wind speed increases with height and wind direction rotates with increasing height until it aligns with the geostrophic wind.




### Hydrostatic Balance

The 1D Column hydrostatic balance example in ['examples/column/hydrostatic.jl'](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/column/hydrostatic.jl) demonstrates the setup and maintenance of hydrostatic balance in a single vertical column using a finite difference discretization.

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

### Prognostic variables

* $\rho$: _density_ measured in kg/m³, discretized at cell centers.
* $\rho\theta$: _potential temperature density_ measured in K·kg/m³, discretized at cell centers.
* $\boldsymbol{w}$: _vertical velocity_ measured in m/s, discretized at cell faces.

### Operators

#### Reconstructions

* $I^f$ is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F), called `If` in the example code.
  - Currently this is implemented as the arithmetic mean.

#### Differentiation operators

* $D^c_v$ is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C), called `∂` in the example code.
  - This example uses zero vertical velocity at the top and bottom boundaries.
* $\nabla^f_v$ is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F), called `∂f` in the example code.
* $B$ is the [boundary operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.SetBoundaryOperator), called `B` in the example code.
  - This enforces zero vertical velocity at domain boundaries.

### Problem flow and set-up

This test case is set up in a vertical column domain from $z=0$ m to $z=30$ km with 30 vertical elements. The column is initialized with a decaying temperature profile, where:

- The virtual temperature starts at $T_{virt\_surf} = 280$ K at the surface
- It asymptotically approaches $T_{min\_ref} = 230$ K with height
- The profile follows a hyperbolic tangent function with height
- The pressure follows a hydrostatic balance equation
- Density is calculated from the equation of state using virtual temperature and pressure

The initial vertical velocity is set to zero everywhere. To maintain hydrostatic balance, the discrete form computes iteratively the values of density that ensure the vertical pressure gradient balances gravity.

The simulation is run for 10 days to verify that the hydrostatic balance is maintained over time. Results are plotted showing density ($\rho$), vertical velocity ($\boldsymbol{w}$), and potential temperature density ($\rho\theta$) profiles.


