using LinearAlgebra
import ClimaCore:
    ClimaCore,
    DataLayouts,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators
using ClimaCore.Geometry
import ClimaCore.Geometry: ⊗
import LazyBroadcast: lazy

# A diffusive flux correction for center- and face-valued quantities, written in
# terms of the gradient operators. Its diffusivity is |velocity| Δz, which is
# the form of numerical diffusion first-order upwinding introduces. Neither
# version carries boundary values of its own: the center-valued one passes no
# correction flux through the boundary faces, and the face-valued one takes its
# outer gradient there to be the one-sided difference with the inner gradient
# set to zero outside of the domain.

function add_flux_correction_c2c!(d_, velocity, quantity)
    FT = Spaces.undertype(axes(quantity))
    # The zero gradient on each boundary face makes the inner gradient vanish
    # there, so no correction flux passes through the face.
    zero_gradient = Operators.SetGradient(
        Geometry.outer(Geometry.Covariant3Vector(zero(FT)), zero(eltype(quantity))),
    )
    quantity_gradc2f =
        Operators.GradientC2F(bottom = zero_gradient, top = zero_gradient)
    lg_field = Fields.local_geometry_field(axes(velocity))
    gradf2c = Operators.GradientF2C()
    @. d_ +=
        adjoint(
            gradf2c(
                adjoint(quantity_gradc2f(quantity)) * Geometry.Contravariant3Vector(
                    abs(Geometry.contravariant3(velocity, lg_field)),
                ),
            ),
        ) * Geometry.Contravariant3Vector(1)
    return
end

function add_flux_correction_f2f!(d_, velocity, quantity)
    gradf2c = Operators.GradientF2C()
    lg_field = Fields.local_geometry_field(axes(velocity))
    inner_grad = @. adjoint(gradf2c(quantity)) * Geometry.Contravariant3Vector(
        abs(Geometry.contravariant3(velocity, lg_field)),
    )
    # The gradient on each boundary face is ±`inner_grad` at the adjacent cell
    # center, which is the one-sided difference with `inner_grad` taken as zero
    # outside of the domain. Each boundary value is a lazy broadcast over the
    # whole column, of which only the level adjacent to its boundary is read.
    top_gradient_extrapolate = Operators.SetGradient(
        @. lazy(Geometry.outer((Geometry.Covariant3Vector(-1),), inner_grad))
    )
    bottom_gradient_extrapolate = Operators.SetGradient(
        @. lazy(Geometry.outer((Geometry.Covariant3Vector(1),), inner_grad))
    )
    gradc2f = Operators.GradientC2F(
        bottom = bottom_gradient_extrapolate,
        top = top_gradient_extrapolate,
    )
    @. d_ += adjoint(gradc2f(inner_grad)) * Geometry.Contravariant3Vector(1)
    return
end
