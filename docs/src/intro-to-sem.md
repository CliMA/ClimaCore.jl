#### Introduction to the Finite/Spectral Element Method

In finite element formulations, the weak form of a Partial Differential Equation
(PDE)---which involves integrating all terms in the PDE over the domain---is
evaluated on a subdomain ``\Omega_e`` (element) and the local results are composed
into a larger system of equations that models the entire problem on the global domain ``\Omega``.

A spectral element space is a function space in which each function is approximated
with a finite-dimensional polynomial interpolation in each element. Hence, we use
polynomials as basis functions to approximate a given function (e.g., solution state).
There are different ways of defininig basis functions: _nodal_ basis functions
and _modal_ basis functions. We use _nodal_ basis functions (e.g. by using
Lagrange interpolation), which are defined via the values of the polynomials
at particular nodal points in each element (termed Finite Element *nodes*).
Even though the basis functions can interpolate globally, it’s better to limit
each function to interpolate locally within each element, so to avoid a dense
matrix system of equations when adding up the element contributions on the
global domain ``\Omega``.

The Finite Element nodes can be chosen to coincide with those of a particular
*quadrature rule*, (this is referred to as using _collocated_ nodes) which
allows us to integrate functions over the domain.

Let us give a concrete example of strong and weak form of a PDE.
A Poisson's problem (in strong form) is given by

```math
   \nabla \cdot \nabla u = f, \textrm{ for  } \mathbf{x} \in \Omega .
```

To obtain the weak form, let us multiply all terms by a test function ``v``
and integrate by parts (i.e., apply the divergence theorem in multiple dimensions):

```math
   \int_\Omega \nabla v \cdot \nabla u \, dV - \int_{\partial \Omega} v \nabla u \cdot \hat{\mathbf n}\, dS = \int_\Omega  v f \, dV .
```

Often, we choose to represent a field (say, the velocity field) such
that ``\nabla u \cdot \hat{\mathbf n} = 0``, so that we're only left with the volumetric parts of the equation above.

The only supported choice for now in ClimaCore.jl is a `Gauss-Legendre-Lobatto`
rule and nodes.
