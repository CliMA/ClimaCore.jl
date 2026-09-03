# Build a space with CommonSpaces

`ClimaCore.CommonSpaces` builds the standard configurations in one call. Each
constructor takes the float type as an optional first argument
(default `Float64`), the geometry as keywords, and, for spaces with a vertical
direction, a `staggering` of `CellCenter()` or `CellFace()`.

## Prerequisites

`using ClimaCore.CommonSpaces` exports the constructors. `ClimaComms.device()`
and `ClimaComms.context()` are the defaults for the `device` and `context`
keywords, so the space is built on whatever `CLIMACOMMS_DEVICE` selects.

## The constructors

| Constructor                | Space                                               | Geometry keywords                                                                   |
|:-------------------------- |:--------------------------------------------------- |:----------------------------------------------------------------------------------- |
| `ColumnSpace`              | Single column, finite difference                    | `z_elem`, `z_min`, `z_max`, `stretch`                                               |
| `SliceXZSpace`             | x–z slice: 1D spectral elements × finite difference | `x_elem`, `x_min`, `x_max`, `periodic_x`, `n_quad_points`, plus the column keywords |
| `RectangleXYSpace`         | 2D plane, spectral elements                         | `x_elem`, `y_elem`, `x_min`, …, `periodic_x`, `periodic_y`, `n_quad_points`         |
| `Box3DSpace`               | 3D box: 2D spectral elements × finite difference    | The plane keywords plus the column keywords                                         |
| `CubedSphereSpace`         | 2D cubed sphere, spectral elements                  | `radius`, `h_elem`, `n_quad_points`                                                 |
| `ExtrudedCubedSphereSpace` | 3D shell: cubed sphere × finite difference          | The sphere keywords plus the column keywords                                        |
| `MultiColumnSpace`         | `N` independent columns at given `LatLongPoint`s    | `points`, `radius`, plus the column keywords                                        |

Every extruded constructor accepts `hypsography_fun`, a function of the
horizontal and vertical grids returning a `Hypsography` adaption
([Use terrain-following coordinates](topography.md)), and `stretch`, a
`Meshes.StretchingRule` for the vertical levels. The horizontal
spectral-element constructors accept `discretization = Grids.DG()` for a
discontinuous space ([Choose CG or DG](choose_cg_dg.md)), `enable_bubble` for
the element-area correction on the sphere, and `enable_mask` for horizontal
masks ([Mask horizontal points](masks.md)).

## Steps

 1. Build the space with the geometry of your case. A 3D box with 3 × 4
    elements of cubic polynomials and 10 vertical cells, on cell centers:

    ```@example common_spaces
    import ClimaComms
    ClimaComms.@import_required_backends
    using ClimaCore.CommonSpaces
    import ClimaCore: Spaces, Fields, Grids
    space = Box3DSpace(;
        x_elem = 3, y_elem = 4, x_min = 0.0, x_max = 1.0, y_min = 0.0, y_max = 1.0,
        periodic_x = true, periodic_y = true, n_quad_points = 4,
        z_elem = 10, z_min = 0.0, z_max = 1.0,
        staggering = CellCenter(),
    )
    ```

 2. Derive the other staggering from it rather than constructing it again; the
    two share one grid.

    ```@example common_spaces
    face_space = Spaces.face_space(space)
    Spaces.grid(face_space) === Spaces.grid(space)
    ```

 3. Read back what was built. The horizontal space, its discretization, and
    the coordinates are available from the space.

    ```@example common_spaces
    (
        horizontal = typeof(Spaces.horizontal_space(space)).name.name,
        discretization = Spaces.discretization(space),
        nlevels = Spaces.nlevels(space),
        z_range = extrema(Fields.coordinate_field(space).z),
    )
    ```

 4. For the sphere, pass the radius and the number of elements per panel edge.
    With `h_elem = 30` and `n_quad_points = 4`, the node spacing is about
    103 km; `h_elem = 120` gives about 26 km.

    ```@example common_spaces
    sphere = ExtrudedCubedSphereSpace(;
        radius = 6.371e6, h_elem = 6, n_quad_points = 4,
        z_elem = 10, z_min = 0.0, z_max = 30e3,
        staggering = CellCenter(),
    )
    Spaces.node_horizontal_length_scale(Spaces.horizontal_space(sphere))
    ```

## The four point-like spaces

Four names look alike and mean different things:

  - `Spaces.PointSpace` is a single point, the horizontal space of one column.
  - `Spaces.MultiPointSpace` is `N` disconnected horizontal points.
  - `MultiColumnSpace` (`Spaces.MultiColumnFiniteDifferenceSpace`) is `N`
    independent columns over those points; it supports vertical operators and
    `Fields.bycolumn`, but no horizontal operators or DSS.
  - `ColumnSpace` (`Spaces.FiniteDifferenceSpace`) is one column.

## What the constructors do

Each space constructor calls the matching `CommonGrids` constructor and wraps
the grid: `Spaces.ExtrudedFiniteDifferenceSpace(grid, staggering)` for the
extruded cases, `Spaces.SpectralElementSpace2D(grid)` for the plane and
sphere. Building the grid yourself, with a custom mesh or topology, and
wrapping it the same way is the escape hatch when a keyword is missing.
