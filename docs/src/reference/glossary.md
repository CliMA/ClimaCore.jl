# Glossary

Terms as ClimaCore uses them. For the atmosphere model's state-vector
notation (`Y`, `Yₜ`, `p`, `ᶜ`, `ᶠ`) see ClimaAtmos's
[notation](https://clima.github.io/ClimaAtmos.jl/stable/notation/) and
[glossary](https://clima.github.io/ClimaAtmos.jl/stable/glossary/) pages; for
ecosystem-wide software terms see [`docs/dev-guides/code-quality/glossary.md`](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/code-quality/glossary.md).

**CG (continuous Galerkin).** The spectral-element discretization in which
node values on shared element boundaries are single-valued and element-local
weak-form operators are completed by DSS. Selected with `Grids.CG()`.

**Completion.** The step that turns an element-local weak-form tendency into
the tendency of the whole discretization: DSS on a CG space, interface
numerical fluxes on a DG space. `Operators.tendency_completion` builds the
completion object for a tendency field; `Operators.complete_tendency!` applies it.

**Contravariant components.** Components of a vector in the basis `eⁱ = ∇ξⁱ`
normal to the coordinate surfaces; the form a divergence needs. Types
`Contravariant12Vector`, `Contravariant3Vector`, `Contravariant123Vector`.

**Covariant components.** Components in the basis `e_i = ∂x/∂ξⁱ` tangent to
the coordinate lines; the form a gradient returns and the atmosphere's
prognostic velocity uses. Types `Covariant12Vector`, `Covariant3Vector`,
`Covariant123Vector`.

**DG (discontinuous Galerkin).** The spectral-element discretization in which
each element holds its own copy of boundary-node values and elements are
coupled by numerical fluxes across their faces. Selected with `Grids.DG()`.

**Domain.** The continuous region a grid covers: an interval, a rectangle, or
the sphere, with named boundaries.

**DSS (direct stiffness summation).** Replacing every copy of an
element-boundary node value by the volume-weighted average of its copies in
the adjacent elements, which makes a field continuous. `Spaces.weighted_dss!`.

**Element.** One quadrilateral (horizontal) or interval (vertical) cell of a
mesh; the unit on which spectral-element operators act.

**Extruded space.** The product of a horizontal spectral-element grid and a
vertical finite-difference grid; the space of a three-dimensional model.
`Spaces.ExtrudedFiniteDifferenceSpace`.

**Face, center.** The two staggerings of the vertical grid: faces are the
interfaces between cells, centers their midpoints. A column with `Nv` cells
has `Nv` centers and `Nv + 1` faces. `Grids.CellFace()`, `Grids.CellCenter()`.

**Field.** Values of one type at every node of a space. `Fields.Field`.

**FieldVector.** A named collection of fields, possibly on different spaces,
that acts as one vector for a time stepper. `Fields.FieldVector`.

**GLL nodes.** Gauss–Lobatto–Legendre quadrature points, the nodes of a
spectral element; they include the element endpoints, so neighboring elements
share nodes. `Quadratures.GLL{Nq}`.

**Grid.** Topology, quadrature, and the local geometry at every node; for
spectral-element grids also the discretization. `Grids.SpectralElementGrid2D`
and relatives.

**Halo (ghost) exchange.** The MPI communication of element-boundary data
between neighboring processes, inside DSS or before DG face fluxes.

**Hypsography.** The terrain-following transformation of the vertical
coordinate: `Hypsography.LinearAdaption`, `Hypsography.SLEVEAdaption`.

**Local geometry.** The coordinates, Jacobian, and metric tensors stored at
each node. `Fields.local_geometry_field(space)`.

**Mesh.** The partition of a domain into elements. `Meshes.IntervalMesh`,
`Meshes.RectilinearMesh`, `Meshes.EquiangularCubedSphere`.

**Node.** A quadrature point of an element; the location of one degree of
freedom.

**Numerical flux.** On a DG grid, the single-valued flux through a face
computed from the states on its two sides, `numflux(normal, argvals⁻, argvals⁺)`;
`Operators.RusanovNumericalFlux`, `Operators.CentralNumericalFlux`.

**Operator.** An object that computes a spatial derivative or interpolation
inside a broadcast expression. Spectral-element operators are element-local;
finite-difference operators act along a column.

**Space.** A grid plus a staggering; what a field lives on.

**Staggering.** See face, center.

**Strong and weak form.** The two forms of a spectral-element derivative: the
strong form differentiates the interpolant; the weak form is its negative
adjoint under the quadrature inner product. `Divergence()` versus
`Divergence{WeakForm}()`.

**Topology.** The connectivity of a mesh's elements, and their distribution
over MPI processes. `Topologies.Topology2D`.

**UVW components.** Components in the local orthonormal frame: zonal,
meridional, radial on the sphere; `x`, `y`, `z` in a box. `UVector`,
`UVVector`, `WVector`, `UVWVector`.

**WJ.** The product of the quadrature weights and the Jacobian at a node: its
discrete volume, the weight of every integral and of DSS.
