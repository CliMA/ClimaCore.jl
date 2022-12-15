# # Introduction to ClimaCore.jl
#
# ### What is ClimaCore?
#
# A suite of tools for constructing spatial discretizations
#
# - primarily aimed at climate and weather models
# - initial aim:
#   - spectral element discretization in the horizontal
#   - staggered finite difference in the vertical
# - currently under development

using ClimaComms,
    ClimaCore,
    ClimaCorePlots,
    LinearAlgebra,
    IntervalSets,
    UnPack,
    Plots,
    OrdinaryDiffEq
#----------------------------------------------------------------------------

# ## 1. Constructing a discretization

# ### 1.1 Domains
#
# A _domain_ is a region of space (think of a mathematical domain).

column_domain = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint(0.0) .. ClimaCore.Geometry.ZPoint(10.0),
    boundary_tags = (:bottom, :top),
)
#----------------------------------------------------------------------------

rectangle_domain = ClimaCore.Domains.RectangleDomain(
    ClimaCore.Geometry.XPoint(-2π) .. ClimaCore.Geometry.XPoint(2π),
    ClimaCore.Geometry.YPoint(-2π) .. ClimaCore.Geometry.YPoint(2π),
    x1periodic = true,
    x2periodic = true,
)
#----------------------------------------------------------------------------

# ### 1.2 Meshes
#
# A _mesh_ is a division of a domain into elements

column_mesh = ClimaCore.Meshes.IntervalMesh(column_domain, nelems = 32)
#----------------------------------------------------------------------------

rectangle_mesh = ClimaCore.Meshes.RectilinearMesh(rectangle_domain, 16, 16)
#----------------------------------------------------------------------------

# ### 1.3 Topologies
#
# A _topology_ determines the ordering and connections between elements of a mesh
# - At the moment, this is only required for 2D meshes

rectangle_topology = ClimaCore.Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(),
    rectangle_mesh,
)
#----------------------------------------------------------------------------

# ### 1.4 Spaces
#
# A _space_ represents a discretized function space over some domain. Currently two discretizations are supported.
#

# #### 1.4.1 Staggered finite difference discretization
#
# This discretizes an interval domain by approximating the function by a value at either the center of each element (`CenterFiniteDifferenceSpace`), or the faces between elements (`FaceFiniteDifferenceSpace`).
#
# You can construct either the center or face space from the mesh, then construct the opposite space from the original one (this is to avoid allocating additional memory).

column_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(column_mesh)
## construct the face space from the center one
column_face_space =
    ClimaCore.Spaces.FaceFiniteDifferenceSpace(column_center_space)
#----------------------------------------------------------------------------

# #### 1.4.2 Spectral element discretization
#
# A spectral element space approximates the function with polynomials in each element. The polynomials are represented using a *nodal discretization*, which stores the values of the polynomials at particular points in each element (termed *nodes* or *degrees of freedom*).
#
# These nodes are chosen by a particular *quadrature rule*, which allows us to integrate functions over the domain. The only supported choice for now is a Gauss-Legendre-Lobatto rule.

## Gauss-Legendre-Lobatto quadrature with 4 nodes in each direction, so 16 in each element
quad = ClimaCore.Spaces.Quadratures.GLL{4}()
rectangle_space =
    ClimaCore.Spaces.SpectralElementSpace2D(rectangle_topology, quad)
#----------------------------------------------------------------------------

# ### 1.5 Fields
#
# Finally, we can construct a *field*: a function in a space. A field is simply a space and the values at each node in the space.
#
# The easiest field to construct is the _coordinate field_

coord = ClimaCore.Fields.coordinate_field(rectangle_space)
#----------------------------------------------------------------------------

# This is a *struct-value field*: it contains coordinates in a struct at each point. We can extract just the `x` coordinate, to get a *scalar field*:

x = coord.x
#----------------------------------------------------------------------------

# Although you can't index directly into a field, it can be used in some other ways similar to a Julia `Array`. For example, broadcasting can be used to define new fields in terms of other ones:

sinx = sin.(x)
#----------------------------------------------------------------------------

# Fields can be easily vizualized with Plots.jl:

import Plots
Plots.plot(sinx)
#----------------------------------------------------------------------------

# If you're using the terminal, `UnicodePlots` is also supported.

# This works similarly for finite difference discretizations

column_center_coords = ClimaCore.Fields.coordinate_field(column_center_space)
column_face_coords = ClimaCore.Fields.coordinate_field(column_face_space)
#----------------------------------------------------------------------------

plot(sin.(column_center_coords.z), ylim = (0.0, 10.0))
plot!(cos.(column_face_coords.z), ylim = (0.0, 10.0))
#----------------------------------------------------------------------------

# Reduction operations are defined anologously:
#
# - `sum` will give the integral of the function
# ```math
# \int_D f(x) dx
# ```
# - `norm` will give the L² function norm
# ```math
# \sqrt{\int_D |f(x)|^2 dx}
# ```
#

sum(sinx) ## integral
#----------------------------------------------------------------------------

norm(sinx) ## L² norm
#----------------------------------------------------------------------------

# ### 1.6 Vectors and vector fields
#
# A *vector field* is a field with vector-valued quantity, i.e. at every point in space, you have a vector.
#
# However one of the key requirements of ClimaCore is to support vectors specified in curvilinear or non-Cartesian coordinates. We will discuss this in a bit further, but for now, you can define a 2-dimensional vector field using `Geometry.UVVector`:

v = ClimaCore.Geometry.UVVector.(coord.y, .-coord.x)
#----------------------------------------------------------------------------

# ## 2. Operators
#
# _Operators_ can compute spatial derivative operations.
#
#  - for performance reasons, we need to be able to "fuse" multiple operators and function applications
#  - Julia provides a tool for this: **broadcasting**, with a very flexible API
#
# Can think of operators are "pseudo-functions": can't be called directly, but act similar to functions in the context of broadcasting.

# ### 2.1 Spectral element operators
#
# The `Gradient` operator takes the gradient of a scalar field, and returns a vector field.

grad = ClimaCore.Operators.Gradient()
∇sinx = grad.(sinx)
#----------------------------------------------------------------------------

plot(∇sinx.components.data.:1, clim = (-1, 1))
#----------------------------------------------------------------------------

# This returns the gradient in [_covariant_](https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors) coordinates
# ```math
# (\nabla f)_i = \frac{\partial f}{\partial \xi^i}
# ```
# where $(\xi^1,\xi^2)$ are the coordinates in the *reference element*: a square $[-1,1]^2$.
#
# This can be converted to a local orthogonal basis by multiplying by the partial derivative matrix
# ```math
# \frac{\partial \xi}{\partial x}
# ```
# This can be done calling `ClimaCore.Geometry.LocalVector:

∇sinx_cart = ClimaCore.Geometry.LocalVector.(∇sinx)
#----------------------------------------------------------------------------

plot(∇sinx_cart.components.data.:1, clim = (-1, 1))
#----------------------------------------------------------------------------

plot(∇sinx_cart.components.data.:2, clim = (-1, 1))
#----------------------------------------------------------------------------

∇sinx_ref = ClimaCore.Geometry.UVVector.(cos.(x), 0.0)
norm(∇sinx_cart .- ∇sinx_ref)
#----------------------------------------------------------------------------

# Similarly, the `Divergence` operator takes the divergence of vector field, and returns a scalar field.
#
# If we take the divergence of a gradient, we can get a Laplacian:

div = ClimaCore.Operators.Divergence()
∇²sinx = div.(grad.(sinx))
plot(∇²sinx)
#----------------------------------------------------------------------------

# *Note*: In curvilinear coordinates, the divergence is defined in terms of the _contravariant_ components $u^i$:
# ```math
# \nabla \cdot u = \frac{1}{J} \sum_i \frac{\partial}{\partial \xi^i} (J u^i)
# ```
# The `Divergence` operator handles this conversion internally.

# #### 2.1.1 Direct stiffness summation
#
# Spectral element operators only operate _within_ a single element, and so the result may be discontinuous. To address this, the usual fix is _direct stiffness summation_ (DSS), which averages the values at the element boundaries.
#
# This corresponds to the $L^2$ projection onto the subset of continuous functions in our function space.

∇²sinx_dss = ClimaCore.Spaces.weighted_dss!(copy(∇²sinx))
plot(∇²sinx_dss)
#----------------------------------------------------------------------------

plot(∇²sinx_dss .- ∇²sinx)
#----------------------------------------------------------------------------

# ### 2.2 Finite difference operators
#
# Finite difference operators are similar with some subtle differences:
# - they can change staggering (center to face, or vice versa)
# - they can span multiple elements
#   - no DSS is required
#   - boundary handling may be required
#
# We use the following convention:
#  - centers are indexed by integers `1, 2, ..., n`
#  - faces are indexed by half integers `½, 1+½, ..., n+½`

# **Face to center gradient**
#
# An finite-difference operator defines a _stencil_. For example, the gradient operator
#
# ```math
# \nabla\theta[i] = \frac{\theta [i+\tfrac{1}{2}] - \theta[i-\tfrac{1}{2}]}{\Delta z}
# ```
# (actually, a little more complicated as it gives a vector in a covariant basis)
#
#
# ```
#         ...
#       /
# θ[2+½]
#       \
#         ∇θ[2]
#       /
# θ[1+½]
#       \
#         ∇θ[1]
#       /
# θ[½]
# ```
#
# Every center value is well-defined, so boundary handling is optional.
#

cosz = cos.(column_face_coords.z)
gradf2c = ClimaCore.Operators.GradientF2C()
∇cosz = gradf2c.(cosz)
#----------------------------------------------------------------------------

plot(map(x -> x.w, ClimaCore.Geometry.WVector.(∇cosz)), ylim = (0, 10))
#----------------------------------------------------------------------------

# **Center to face gradient**
#
# Uses the same stencil, but doesn't work directly:

sinz = sin.(column_center_coords.z)
gradc2f = ClimaCore.Operators.GradientC2F()
## ∇sinz = gradc2f.(sinz) ## this would throw an error
#----------------------------------------------------------------------------

# This throws an error because face values at the boundary are _not_ well-defined:
#
# ```
# ...
#       \
#         ∇θ[2+½]
#       /
# θ[2]
#       \
#         ∇θ[1+½]
#       /
# θ[1]
#       \
#         ????
# ```
#
# To handle boundaries we need to *modify the stencil*. Two options:
# - provide the _value_ $\theta^*$ of $\theta$ at the boundary:
# ```math
# \nabla\theta[\tfrac{1}{2}] = \frac{\theta[1] - \theta^*}{\Delta z /2}
# ```
#
# - provide the *gradient* $\nabla\theta^*$ of $\theta$ at the boundary:
# ```math
# \nabla\theta[\tfrac{1}{2}] = \nabla\theta^*
# ```
#
# These modified stencils are provided as keyword arguments to the operator (based on the boundary label names):

sinz = sin.(column_center_coords.z)
gradc2f = ClimaCore.Operators.GradientC2F(
    bottom = ClimaCore.Operators.SetValue(sin(0.0)),
    top = ClimaCore.Operators.SetGradient(
        ClimaCore.Geometry.WVector(cos(10.0)),
    ),
)
∇sinz = gradc2f.(sinz)
#----------------------------------------------------------------------------

plot(map(x -> x.w, ClimaCore.Geometry.WVector.(∇sinz)), ylim = (0, 10))
#----------------------------------------------------------------------------

# As before, multiple operators (or functions) can be fused together with broadcasting.
#
# One extra advantage of this is that boundaries of the inner operators only need to be specified if they would affect the final result.
#
# Consider the center-to-center Laplacian:
#
# ```
# ...
#       \       /
#         ∇θ[2+½]
#       /       \
# θ[2]            ∇⋅∇θ[2]
#       \       /
#         ∇θ[1+½]
#       /       \
# θ[1]            ∇⋅∇θ[1]
#               /
#          ∇θ*
# ```

sinz = sin.(column_center_coords.z)
## we don't need to specify boundaries, as the stencil won't reach that far
gradc2f = ClimaCore.Operators.GradientC2F()
divf2c = ClimaCore.Operators.DivergenceF2C(
    bottom = ClimaCore.Operators.SetValue(ClimaCore.Geometry.WVector(cos(0.0))),
    top = ClimaCore.Operators.SetValue(ClimaCore.Geometry.WVector(cos(10.0))),
)
∇∇sinz = divf2c.(gradc2f.(sinz))
#----------------------------------------------------------------------------

plot(∇∇sinz, ylim = (0, 10))
#----------------------------------------------------------------------------

# # 3. Solving PDEs
#
# ClimaCore can be used for spatial discretizations of PDEs. For temporal discretization, we can use the OrdinaryDiffEq package, which we aim to be compatibile with.

using OrdinaryDiffEq
#----------------------------------------------------------------------------

# ### 3.1 Heat equation using finite differences
#
# We will use a cell-center discretization of the heat equation:
# ```math
# \frac{\partial y}{\partial t} = \alpha \nabla \cdot \nabla y
# ```
#
# At the bottom we will use a Dirichlet condition ``y(0) = 1``` at the bottom: since we don't actually have a value located at the bottom, we will use a `SetValue` boundary modifier on the inner gradient.
#
# At the top we will use a Neumann condition ``\frac{\partial y}{\partial z}(10) = 0``. We can do this two equivalent ways:
#  - a `SetGradient` on the gradient operator
#  - a `SetValue` on the divergence operator
#
# either will work.

y0 = zeros(column_center_space)

## define the tendency function
function heat_fd_tendency!(dydt, y, α, t)
    gradc2f = ClimaCore.Operators.GradientC2F(
        bottom = ClimaCore.Operators.SetValue(1.0),
        top = ClimaCore.Operators.SetGradient(ClimaCore.Geometry.WVector(0.0)),
    )
    divf2c = ClimaCore.Operators.DivergenceF2C()
    ## the @. macro "dots" the whole expression
    ## i.e.  dydt .= α .* divf2c.(gradc2f.(y))
    @. dydt = α * divf2c(gradc2f(y))
end

heat_fd_prob = ODEProblem(heat_fd_tendency!, y0, (0.0, 5.0), 0.1)
heat_fd_sol = solve(heat_fd_prob, SSPRK33(), dt = 0.1, saveat = 0.25)
#----------------------------------------------------------------------------

anim = Plots.@animate for u in heat_fd_sol.u
    plot(u, xlim = (0, 1), ylim = (0, 10))
end
mp4(anim)
#----------------------------------------------------------------------------

# ### 3.2 Heat equation using continuous Galerkin (CG) spectral element
#
#

function heat_cg_tendency!(dydt, y, α, t)
    grad = ClimaCore.Operators.Gradient()
    wdiv = ClimaCore.Operators.WeakDivergence()
    ## apply element operators
    @. dydt = α * wdiv(grad(y))

    ## direct stiffness summation (DSS): project to continuous function space
    ClimaCore.Spaces.weighted_dss!(dydt)
    return dydt
end

y0 = exp.(.-(coord.y .^ 2 .+ coord.x .^ 2) ./ 2)

heat_cg_prob = ODEProblem(heat_cg_tendency!, y0, (0.0, 5.0), 0.1)
heat_cg_sol = solve(heat_cg_prob, SSPRK33(), dt = 0.1, saveat = 0.5)
#----------------------------------------------------------------------------

anim = Plots.@animate for u in heat_cg_sol.u
    Plots.plot(u, c = :thermal)
end
mp4(anim)
#----------------------------------------------------------------------------

# ### 3.3 Shallow water equations
#
#
# The shallow water equations in vector invariant form can be written as
# ```math
# \begin{align*}
#     \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho u) &= 0\\
#     \frac{\partial u_i}{\partial t} + \nabla (\Phi + \tfrac{1}{2}\|u\|^2)_i  &= J (u \times (\nabla \times u))_i
# \end{align*}
# ```
# where ``J`` is the Jacobian determinant, and ``\Phi = g \rho``.
#
# Note that the velocity ``u`` is specified in _covariant_ coordinates ``u_i``.
#
# For vizualization purposes, we can model a passive tracer $\theta$ as
# ```math
# \frac{\partial \rho \theta}{\partial t} + \nabla \cdot (\rho \theta u) = 0
# ```

using ClimaCore.Geometry

parameters = (
    ϵ = 0.1,  ## perturbation size for initial condition
    l = 0.5, ## Gaussian width
    k = 0.5, ## Sinusoidal wavenumber
    ρ₀ = 1.0, ## reference density
    c = 2,
    g = 10,
    D₄ = 1e-4, ## hyperdiffusion coefficient
)

function init_state(local_geometry, p)
    coord = local_geometry.coordinates
    @unpack x, y = coord
    ## set initial state
    ρ = p.ρ₀

    ## set initial velocity
    U₁ = cosh(y)^(-2)

    ## Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    ## Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    ϕ = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = ϕ * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * ϕ * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * ϕ * sin(p.k * x) * cos(p.k * y)

    u = Geometry.Covariant12Vector(
        Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′),
        local_geometry,
    )

    ## set initial tracer
    θ = sin(p.k * y)
    return (ρ = ρ, u = u, ρθ = ρ * θ)
end


y0 =
    init_state.(
        ClimaCore.Fields.local_geometry_field(rectangle_space),
        Ref(parameters),
    )

## plot initial tracer
Plots.plot(y0.ρθ)
#----------------------------------------------------------------------------

function shallow_water_tendency!(dydt, y, _, t)

    @unpack D₄, g = parameters

    sdiv = ClimaCore.Operators.Divergence()
    wdiv = ClimaCore.Operators.WeakDivergence()
    grad = ClimaCore.Operators.Gradient()
    wgrad = ClimaCore.Operators.WeakGradient()
    curl = ClimaCore.Operators.Curl()
    wcurl = ClimaCore.Operators.WeakCurl()


    ## compute hyperviscosity first
    @. dydt.u =
        wgrad(sdiv(y.u)) -
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))
    @. dydt.ρθ = wdiv(grad(y.ρθ))

    ClimaCore.Spaces.weighted_dss!(dydt)

    @. dydt.u =
        -D₄ * (
            wgrad(sdiv(dydt.u)) - Geometry.Covariant12Vector(
                wcurl(Geometry.Covariant3Vector(curl(dydt.u))),
            )
        )
    @. dydt.ρθ = -D₄ * wdiv(grad(dydt.ρθ))

    ## comute rest of tendency
    @. begin
        dydt.ρ = -wdiv(y.ρ * y.u)
        dydt.u += -grad(g * y.ρ + norm(y.u)^2 / 2) + y.u × curl(y.u)
        dydt.ρθ += -wdiv(y.ρθ * y.u)
    end
    ClimaCore.Spaces.weighted_dss!(dydt)
    return dydt
end

#----------------------------------------------------------------------------

shallow_water_prob = ODEProblem(shallow_water_tendency!, y0, (0.0, 20.0))
@time shallow_water_sol =
    solve(shallow_water_prob, SSPRK33(), dt = 0.05, saveat = 1.0)
anim = Plots.@animate for u in shallow_water_sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
mp4(anim)
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
