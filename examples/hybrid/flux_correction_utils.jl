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

function add_flux_correction_c2c(d_, velocity, quantity)
    n_levels = Fields.nlevels(quantity)
    top_level_space = axes(Fields.level(quantity, n_levels))
    bottom_level_space = axes(Fields.level(quantity, 1))
    quantity_top_gradient_extrapolate = Operators.SetGradient(
        Geometry.outer.(
            (Geometry.Covariant3Vector(1),),
            Fields.Field(
                Fields.field_values(Fields.level(quantity, n_levels)) .-
                Fields.field_values(Fields.level(quantity, n_levels - 1)),
                top_level_space,
            ),
        ),
    )
    quantity_bottom_gradient_extrapolate = Operators.SetGradient(
        Geometry.outer.(
            (Geometry.Covariant3Vector(1),),
            Fields.Field(
                (
                    Fields.field_values(Fields.level(quantity, 2)) .-
                    Fields.field_values(Fields.level(quantity, 1))
                ),
                bottom_level_space,
            ),
        ))
    quantity_gradc2f = Operators.GradientC2F(
        bottom = quantity_bottom_gradient_extrapolate,
        top = quantity_top_gradient_extrapolate,
    )
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

function add_flux_correction_f2f(d_, velocity, quantity)
    gradf2c = Operators.GradientF2C()
    lg_field = Fields.local_geometry_field(axes(velocity))
    inner_grad = @. adjoint(gradf2c(quantity)) * Geometry.Contravariant3Vector(
        abs(Geometry.contravariant3(velocity, lg_field)),
    )
    n_levels = Fields.nlevels(inner_grad)
    top_level_space = axes(Fields.level(inner_grad, n_levels))
    bottom_level_space = axes(Fields.level(inner_grad, 1))
    top_gradient_extrapolate = Operators.SetGradient(
        Geometry.outer.(
            (Geometry.Covariant3Vector(1),),
            Fields.Field(
                Fields.field_values(Fields.level(inner_grad, n_levels)) .-
                Fields.field_values(Fields.level(inner_grad, n_levels - 1)),
                top_level_space,
            ),
        ),
    )
    bottom_gradient_extrapolate = Operators.SetGradient(
        Geometry.outer.(
            (Geometry.Covariant3Vector(1),),
            Fields.Field(
                (
                    Fields.field_values(Fields.level(inner_grad, 1)) .-
                    Fields.field_values(Fields.level(inner_grad, 2))
                ),
                bottom_level_space,
            ),
        ))
    gradc2f = Operators.GradientC2F(
        bottom = bottom_gradient_extrapolate,
        top = top_gradient_extrapolate,
    )
    lg_field = Fields.local_geometry_field(axes(velocity))
    @. d_ +=
        adjoint(
            gradc2f(inner_grad
            ),
        ) * Geometry.Contravariant3Vector(1)
    return
end
