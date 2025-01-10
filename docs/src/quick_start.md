# Quick start

This code:

```@example quickstart
using ClimaCore: Fields, Operators, Geometry
using ClimaCore.Utilities: PlusHalf
using ClimaCore.CommonSpaces
FT = Float64;
cspace = ExtrudedCubedSphereSpace(FT;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 100,
    h_elem = 30,
    n_quad_points = 4,
    staggering = CellCenter()
)
gradc2f = Operators.GradientC2F(
    bottom = Operators.SetValue(sin(0.0)),
    top = Operators.SetGradient(
        Geometry.WVector(cos(10.0)),
    ),
);
lat = Fields.coordinate_field(cspace).lat;
long = Fields.coordinate_field(cspace).long;
z = Fields.coordinate_field(cspace).z;
sinz = sin.(lat .* long .* z);
∇sinz = gradc2f.(sinz);

import ClimaCoreMakie
import CairoMakie # needed for backend
ClimaCoreMakie.fieldheatmap(Fields.level(sinz, 1))
ClimaCoreMakie.fieldheatmap(Fields.level(∇sinz, PlusHalf(2)))
```

Creates an cubed sphere space, and a field that lives on that space. Then we
compute the gradient of a `sin` function based on the z-coordinates using the
`GradientC2F` [functor]
(https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects) in
ClimaCore `Operators` module.

## Visualization

Let's visualize this field, using the [`ClimaCoreMakie`](@ref) library package.

```@example quickstart
```

## What's next?


