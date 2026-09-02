# Use terrain-following coordinates

An extruded space follows terrain when its grid carries a
`Hypsography.HypsographyAdaption`: the vertical levels are remapped between a
surface-elevation field and the flat model top. Operators need no change; the
metric terms of the warped grid carry the terrain.

## Prerequisites

A horizontal space on which the surface elevation is a field of `ZPoint`s, or
an analytic surface.

## Steps

 1. Define the surface elevation on the horizontal grid. Inside a
    `CommonSpaces` constructor, this is the `hypsography_fun(h_grid, z_grid)`
    keyword, called with the horizontal and vertical grids:

    ```@example topography
    import ClimaComms
    ClimaComms.@import_required_backends
    using ClimaCore.CommonSpaces
    import ClimaCore
    import ClimaCore: Spaces, Fields, Geometry, Hypsography
    function mountain(h_grid, z_grid)
        h_space = Spaces.SpectralElementSpace2D(h_grid)
        x = Fields.coordinate_field(h_space).x
        z_surface = @. Geometry.ZPoint(500 * exp(-(x - 50e3)^2 / (2 * (10e3)^2)))
        return Hypsography.LinearAdaption(z_surface)
    end
    space = Box3DSpace(;
        x_elem = 10, y_elem = 2, x_min = 0.0, x_max = 100e3, y_min = 0.0, y_max = 10e3,
        periodic_x = true, periodic_y = true, n_quad_points = 4,
        z_elem = 20, z_min = 0.0, z_max = 10e3,
        staggering = CellFace(), hypsography_fun = mountain,
    )
    z = Fields.coordinate_field(space).z
    extrema(Fields.level(z, ClimaCore.Utilities.half))   # the surface follows the mountain
    ```

 2. Choose the adaption, which maps the reference height `z_ref ∈ [0, z_top]`
    of each level to its physical height above terrain of elevation `h(x, y)`.

    `Hypsography.LinearAdaption(z_surface)` is the Gal-Chen transformation
    [GalChen1975](@cite):

    ```math
    z = z_\mathrm{ref} + \left(1 - \frac{z_\mathrm{ref}}{z_\mathrm{top}}\right) h .
    ```

    Every level is displaced by a fraction of the terrain height that falls
    linearly from 1 at the surface to 0 at the top, so the terrain signature
    reaches all the way up and the levels tilt over a mountain even near the
    model top.

    `Hypsography.SLEVEAdaption(z_surface, ηₕ, s)` is the smooth-level
    vertical coordinate of [Schar2002](@cite) in the single-scale form the
    package implements. With `η = z_ref / z_top`,

    ```math
    z = \eta\, z_\mathrm{top} + h\, \frac{\sinh\bigl((\eta_h - \eta)/(s\,\eta_h)\bigr)}{\sinh(1/s)}
    \quad (\eta \le \eta_h), \qquad z = \eta\, z_\mathrm{top} \quad (\eta > \eta_h).
    ```

    The displacement equals `h` at the surface, decays like a hyperbolic sine
    with height, and is zero at and above the reference height `ηₕ z_top`, so
    the upper levels are flat. The two parameters are dimensionless fractions
    of the domain height:

      + `ηₕ ∈ [0, 1]`: the level above which no warping is applied (`0.7` puts
        flat levels in the top 30% of the domain);
      + `s > 0`: the decay scale of the terrain signature, as a fraction of
        `z_top`; smaller `s` flattens the levels faster. The constructor requires
        `s z_top` to exceed the maximum surface elevation and raises an error
        otherwise, since the map would fold over.

    ```@example topography
    sleve(h_grid, z_grid) =
        Hypsography.SLEVEAdaption(mountain(h_grid, z_grid).surface, 0.7, 0.3)
    nothing # hide
    ```

    With `LinearAdaption`, the metric terms `g³¹, g³²` over a slope persist to
    the top; with `SLEVEAdaption`, they vanish above `ηₕ z_top`, which removes
    the terrain-induced error in the horizontal pressure gradient from the upper
    levels.

 3. Smooth real terrain before use. Elevation data at the grid scale produces
    large metric terms and a spurious pressure-gradient error;
    `Hypsography.diffuse_surface_elevation!(z_surface; κ, maxiter, dt)` applies
    a spectral second-order diffusion for `maxiter` forward-Euler steps of
    size `dt` and mutates the field in place. The defaults (`κ = 1e8`,
    `maxiter = 100`, `dt = 0.1`) are sized for a global grid in meters.

 4. Stretch the levels so that the thin cells sit near the surface. Stretching
    acts on the reference coordinate before the terrain-following map: the
    vertical mesh places its `Nv + 1` faces at reference heights that are
    uniform in a stretched coordinate `ζ`, and the adaption then warps them.
    A `Meshes.StretchingRule` is passed as the `stretch` keyword of the
    `CommonSpaces` constructors or as the second argument of
    `Meshes.IntervalMesh`. The rules, with `η = (z − z₀)/(z₁ − z₀)`:

    | Rule                                                         | Parameters                                          | Spacing                                                                                                                                            |
    |:------------------------------------------------------------ |:--------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `Meshes.Uniform()`                                           | none                                                | Equal cells.                                                                                                                                       |
    | `Meshes.ExponentialStretching(H)`                            | `H`: scale height [m]                               | Uniform in `ζ = (1 − e^(−η/h)) / (1 − e^(−1/h))` with `h = H / (z₁ − z₀)`; cells grow exponentially with height (`H ≈ 7.5 km` for the atmosphere). |
    | `Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)` | target cell heights at the bottom and top [m]       | Exponential growth whose rate is solved from the two target spacings.                                                                              |
    | `Meshes.HyperbolicTangentStretching(dz_surface)`             | `dz_surface`: target cell height at the surface [m] | Uniform in `ζ` with `η = 1 − tanh[γ (1 − ζ)] / tanh γ`; `γ` is solved so the first cell has height `dz_surface`.                                   |

    All three stretched rules take `reverse_mode = true` to put the thin cells
    at the top instead, the arrangement of a land model. The CliMA atmosphere
    uses hyperbolic-tangent stretching with a 30 m surface cell in a 30 km
    domain of 43 levels [Yatunin2026](@cite):

    ```@example topography
    import ClimaCore: Meshes, Domains
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0), Geometry.ZPoint(30e3); boundary_names = (:bottom, :top),
    )
    z_mesh =
        Meshes.IntervalMesh(z_domain, Meshes.HyperbolicTangentStretching(30.0); nelems = 43)
    dz = diff([f.z for f in z_mesh.faces])
    (bottom_cell = dz[1], top_cell = dz[end], nlevels = length(dz))
    ```

## What changes on a warped grid

Terrain is where the two component sets of a vector differ. The figure on the
[Mathematical framework](../explanation/math_framework.md) page shows the
bases on a grid like this one: over a slope, the covariant `ê₃` follows the
tilted column while the contravariant `ê³` stays normal to the level, so the
covariant component `u₃` (flow along the column) and the contravariant `u³`
(flow through the level) differ, and the metric terms `g³¹, g³²` that relate
them are nonzero. Two consequences for a model:

  - The surface is the coordinate surface `ξ³ = 0`, so "no flow through the
    surface" is the condition `u³ = 0` on the contravariant component. With the
    horizontal velocity given, the covariant `u₃` at the surface follows from
    `u³ = g³¹ u₁ + g³² u₂ + g³³ u₃ = 0`. Gradients return covariant components
    and divergences consume contravariant ones, so a vertical `SetGradient`
    boundary condition prescribes `∂/∂ξ³` along the tilted column, and a zero
    there is a zero normal derivative only where the surface is flat
    ([Apply boundary conditions](boundary_conditions.md)).
  - `Fields.coordinate_field(space).z` varies horizontally on every level, and
    `Fields.local_geometry_field(space).J` carries the cell volumes, so
    `sum(field)` remains the physical integral.

[Hybrid grids and generalized coordinates](../explanation/geometry.md) defines
the bases, the metric tensor, and the operator forms.

## Where it is used

[`examples/hybrid/plane/schar_mountain.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/plane/schar_mountain.jl) and `agnesi_mountain.jl` run the
linear mountain-wave cases on an x–z slice; ClimaAtmos's
[topography page](https://clima.github.io/ClimaAtmos.jl/stable/topography/)
describes how it regrids and smooths the ETOPO elevation data for the sphere.
