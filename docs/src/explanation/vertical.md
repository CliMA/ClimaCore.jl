# Staggered vertical discretization

The vertical direction is discretized by finite differences on a staggered
grid, not by spectral elements. This page states the grid, the operators that
move between its two sets of points, and the discrete identities that make
vertical transport conservative.

## Why a staggered finite-difference grid

An atmosphere model resolves the vertical far more finely than the
horizontal: tens of meters near the surface against tens of kilometers
horizontally. Two consequences shape the vertical discretization.

First, the fast vertical waves that the fine spacing admits would severely
limit the time step if treated explicitly. A finite-difference stencil couples only
neighboring levels, so the vertical terms can be solved implicitly column by
column with a banded matrix, with all communication local to the column.
Spectral elements in the vertical would couple all levels of an element and
lose that structure.

Second, every unstaggered vertical grid carries computational modes:
grid-scale oscillations that are invisible to the discrete operators and
therefore undamped by the equations. Thuburn and Woollings
[Thuburn05n](@cite) analyzed which vertical arrangements of the prognostic
variables give the compressible Euler equations a discrete normal-mode
spectrum free of such modes, and staggering the vertical velocity against the
thermodynamic variables is what achieves it. The harm the modes do grows with
the aspect ratio of the grid cells, so suppressing them matters most when the
horizontal spacing exceeds the vertical spacing by orders of magnitude, as it
does here.

ClimaCore uses the Lorenz staggering: the covariant vertical velocity
component `u₃` lives on cell **faces**, and every other prognostic variable,
including the horizontal velocity components, lives on cell **centers**. The
Charney–Phillips arrangement, which also places the thermodynamic variable on
faces, has marginally better normal-mode properties for some waves; Lorenz is
used because it is the arrangement under which the exact conservation
properties of the atmosphere's energy formulation hold most directly
[Yatunin2026](@cite).

## Centers, faces, and spaces

A column with `Nv` cells has `Nv` centers and `Nv + 1` faces, the first face
at the surface. In ClimaCore, these are two spaces over one grid:
`Spaces.CenterFiniteDifferenceSpace` and `Spaces.FaceFiniteDifferenceSpace`
for a single column, and `CenterExtrudedFiniteDifferenceSpace` /
`FaceExtrudedFiniteDifferenceSpace` when the column is extruded from a
horizontal spectral-element grid. Centers are indexed by integers `1, …, Nv`.
Faces sit between them and are addressed with the `Utilities.PlusHalf` type:
`PlusHalf(i)` is an integer `i` tagged as the face above center `i`, written
`i + half` in code with `half = PlusHalf(0)`, so the faces run from `half`
(the surface) to `Nv + half` (the top). Docstrings write these positions as
`i + ½` for readability; the stored index stays an integer. The
levels are uniform in the reference coordinate `ξ³` and stretched, and
terrain-following, in physical space ([Hybrid grids and generalized
coordinates](geometry.md)).

## Operators

Every vertical operator maps between the two staggerings, and its name says
which way: `C2F` from centers to faces, `F2C` from faces to centers. Each
takes boundary conditions by keyword for the `bottom` and `top` faces.

**Interpolation.** `InterpolateC2F` and `InterpolateF2C` are the arithmetic
mean of the two neighbors,

```math
I^f(x)[i + \tfrac12] = \tfrac12 (x[i] + x[i + 1]), \qquad
I^c(x)[i] = \tfrac12 (x[i - \tfrac12] + x[i + \tfrac12]).
```

At a boundary face `InterpolateC2F` needs a rule: `SetValue(x₀)` prescribes
the face value, `Extrapolate()` copies the nearest center. The mass-weighted
average `WeightedInterpolateC2F`, `WI^f(w, x) = I^f(w x) / I^f(w)`, is used
for quantities that multiply a mass flux, so that the face value of `ρ ψ` is
consistent with the cell masses on either side.

**Derivatives.** `GradientC2F` and `GradientF2C` are differences between
adjacent points, returning the covariant component along `ξ³`,

```math
G^f(x)[i + \tfrac12]_3 = x[i + 1] - x[i], \qquad
G^c(x)[i]_3 = x[i + \tfrac12] - x[i - \tfrac12];
```

the reference cell has unit length in `ξ³`, and the metric terms convert to
a physical derivative. `DivergenceF2C` and `DivergenceC2F` difference the
Jacobian-weighted contravariant component and divide by the Jacobian at the
result,

```math
D^c(u)[i] = \frac{(J u^3)[i + \tfrac12] - (J u^3)[i - \tfrac12]}{J[i]},
```

and `CurlC2F` differences the horizontal covariant components to give the
horizontal contravariant components of the curl. Boundary conditions
prescribe the operator's output at the boundary face (`SetGradient`,
`SetDivergence`, `SetCurl`) or, through a `SetValue` on the argument, replace
the boundary stencil by one built from a prescribed argument value
(`DirichletOperator`, e.g. `gradient_c2f_dirichlet`).

**Advective reconstructions.** A centered average of an advected scalar
produces dispersive oscillations and negative concentrations. The
`C2F` advection operators reconstruct the face value from the upwind side:
`UpwindBiasedProductC2F` (first order), `Upwind3rdOrderBiasedProductC2F`,
the van Leer limiter `LinVanLeerC2F` with the local-extrema constraint of
[Lin1994](@cite), the flux-corrected transport operators `FCTBorisBook` and
`FCTZalesak` [BorisBook1973, zalesak1979fully](@cite), and the
total-variation-diminishing family `TVDLimitedFluxC2F` with several slope
limiters. Each returns the face flux `w ψ_face`, given the face velocity and
the center field.

## Discrete identities

The two differences are adjoint to each other under the discrete volumes:
for a center field `φ` and a face flux `ψ = J u³`, summing the divergence
against `φ` over the centers of a column telescopes,

```math
\sum_{i} J[i]\, \phi[i]\, D^c(u)[i]
  + \sum_{i} \psi[i + \tfrac12]\, G^f(\phi)[i + \tfrac12]_3
  = \bigl[\phi\, \psi\bigr]_{\text{bottom}}^{\text{top}},
```

with the boundary term evaluated from the boundary-face flux and the
adjacent center value. Setting `φ = 1` shows that a vertical flux divergence
integrated over a column reduces to the fluxes through its two ends, and
pairing a gradient with the divergence of a mass flux gives the discrete
kinetic-energy budget its telescoping form, the same structure as the
strong/weak pairing in the horizontal. Setting the transported scalar to one
in a flux reconstruction recovers the mass flux divergence, so tracer
transport is consistent with mass transport and a uniform tracer stays
uniform. These identities, together with the horizontal ones, are what the
atmosphere's conservation proofs rest on [Yatunin2026](@cite); the
energy-conserving vertical differencing follows the tradition of
[Simmons81](@cite).

## Where the vertical meets the horizontal

On an extruded space, a horizontal operator acts level by level and a vertical
operator column by column, and both read the same three-dimensional local
geometry. A horizontal derivative of a face field is a face field; a vertical
derivative of a center field is a face field. Most tendencies therefore
interleave the two: a flux is built on faces from center quantities by a
`C2F` reconstruction, its vertical divergence lands on centers, and its
horizontal divergence is taken on the same level. ClimaAtmos's
[discretization page](https://clima.github.io/ClimaAtmos.jl/stable/discretization/)
shows the resulting semi-discrete equations term by term.
