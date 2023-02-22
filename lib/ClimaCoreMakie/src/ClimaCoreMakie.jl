module ClimaCoreMakie

export fieldcontourf, fieldcontourf!, fieldheatmap, fieldheatmap!

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f, Observable
import ClimaCore

# line plots
@recipe(FieldLine, field) do scene
    attrs = Attributes()
    merge(a, default_theme(scene, Lines))
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField1D) =
    FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}}

function Makie.plot!(
    plot::FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}},
)

    field = plot.field
    # Only update vertices if space updates
    space = lift(axes, field; ignore_equal_values = true)

    vertices = parent(ClimaCore.Spaces.coordinates_data(space[]))

    # insert NaNs to create gaps between elements
    Nq, _, Nh = size(vertices[])
    vals = parent(field[])
    positions = [
        i <= Nq ? Makie.Point2f(vertices[i, 1, h], vals[i, 1, h]) :
        Makie.Point2f(NaN, NaN) for i in 1:(Nq + 1), h in 1:Nh
    ]

    Makie.lines!(plot, vec(positions))
end

# helper functions

function space_vertices(
    space::ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace,
)
    coords = ClimaCore.Spaces.coordinates_data(space)
    return Makie.Point2f.(
        vec(parent(getproperty(coords, 1))),
        vec(parent(getproperty(coords, 2))),
    )
end
function space_vertices(
    space::ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    coords = ClimaCore.Spaces.coordinates_data(space)
    vertices =
        Makie.Point2f.(
            parent(getproperty(coords, 1)),
            parent(getproperty(coords, 2)),
        )

    # we extend the extrema to the boundaries
    face_space = ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(space)
    face_coords = ClimaCore.Spaces.coordinates_data(face_space)
    nf = ClimaCore.Spaces.nlevels(face_space)
    bottom_coords = ClimaCore.level(face_coords, 1)
    vertices[1, :, :, :] =
        Makie.Point2f.(
            parent(getproperty(bottom_coords, 1)),
            parent(getproperty(bottom_coords, 2)),
        )
    top_coords = ClimaCore.level(face_coords, nf)
    vertices[end, :, :, :] =
        Makie.Point2f.(
            parent(getproperty(top_coords, 1)),
            parent(getproperty(top_coords, 2)),
        )
    return vec(vertices)
end
function space_triangles(space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace)
    (Ni, _, _, Nv, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Nv, Ni, Nh)
    return GLTriangleFace.(a, b, c)
end
function space_triangles_matrix(
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
)
    (Ni, _, _, Nv, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Nv, Ni, Nh)
    return vcat(a', b', c')
end



"""
    fieldheatmap(field)

Plots a heatmap of a field.

## Attributes

Inherited from [`Makie.mesh`](https://docs.makie.org/stable/examples/plotting_functions/mesh/index.html#mesh). 

- `colormap::Union{Symbol, Vector{<:Colorant}} = :viridis`` sets the colormap that is sampled for numeric colors.

- `colorrange::Tuple{<:Real, <:Real}` sets the values representing the start and end points of colormap.

- `nan_color::Union{Symbol, <:Colorant} = RGBAf(0,0,0,0)` sets a replacement color for `color = NaN`.

- `lowclip::Union{Automatic, Symbol, <:Colorant} = automatic` sets a color for any value below the colorrange.

- `highclip::Union{Automatic, Symbol, <:Colorant} = automatic` sets a color for any value above the colorrange.

"""
@recipe(FieldHeatmap, field) do scene
    attrs = Makie.Attributes(; shading = false)
    return merge(attrs, Makie.default_theme(scene, Makie.Mesh))
end

Makie.plottype(::ClimaCore.Fields.ExtrudedFiniteDifferenceField) =
    FieldHeatmap{<:Tuple{ClimaCore.Fields.ExtrudedFiniteDifferenceField}}

function Makie.plot!(plot::FieldHeatmap)
    field = plot.field

    # Only update vertices if axes updates
    space = lift(axes, field; ignore_equal_values = true)
    vertices = lift(space_vertices, space)
    triangles = lift(space_triangles, space)

    plot.color = map(vec ∘ parent, field)

    Makie.mesh!(plot, vertices, triangles; plot.attributes...)
end

Makie.Colorbar(fig_or_scene, plot::FieldHeatmap; kwargs...) =
    Makie.Colorbar(fig_or_scene, plot.plots[1]; kwargs...)

"""
    fieldcontourf(field::Field)

Plots a 2D filled contour plot of a field.

## Attributes

These are inherited from [`Makie.tricontourf`](https://docs.makie.org/stable/examples/plotting_functions/tricontourf/):

- `levels = 10` can be either an `Int` which results in n bands delimited by n+1 equally spaced levels, or it can be an `AbstractVector{<:Real}` that lists n consecutive edges from low to high, which result in n-1 bands.
- `mode = :normal` sets the way in which a vector of levels is interpreted, if it's set to `:relative`, each number is interpreted as a fraction between the minimum and maximum values of `zs`. For example, `levels = 0.1:0.1:1.0` would exclude the lower 10% of data.
- `extendlow = nothing`. This sets the color of an optional additional band from `minimum(zs)` to the lowest value in `levels`. If it's `:auto`, the lower end of the colormap is picked and the remaining colors are shifted accordingly. If it's any color representation, this color is used. If it's `nothing`, no band is added.
- `extendhigh = nothing`. This sets the color of an optional additional band from the highest value of `levels` to `maximum(zs)`. If it's `:auto`, the high end of the colormap is picked and the remaining colors are shifted accordingly. If it's any color representation, this color is used. If it's `nothing`, no band is added.
- `color` sets the color of the plot. It can be given as a named color `Symbol` or a `Colors.Colorant`. Transparency can be included either directly as an alpha value in the `Colorant` or as an additional float in a tuple `(color, alpha)`. The color can also be set for each scattered marker by passing a `Vector` of colors or be used to index the `colormap` by passing a `Real` number or `Vector{<: Real}`.
- `colormap::Union{Symbol, Vector{<:Colorant}} = :viridis` sets the colormap from which the band colors are sampled.

"""
@recipe(FieldContourf, field) do scene
    attrs = Makie.Attributes(;)
    return merge(attrs, Makie.default_theme(scene, Makie.Tricontourf))
end

function Makie.plot!(plot::FieldContourf)

    field = plot.field
    # Only update vertices if axes updates
    space = lift(axes, field; ignore_equal_values = true)
    vertices = lift(space_vertices, space)
    plot.triangulation = lift(space_triangles_matrix, space)

    xs = lift(vertices) do vertices
        map(x -> x[1], vertices)
    end
    ys = lift(vertices) do vertices
        map(x -> x[2], vertices)
    end
    zs = map(vec ∘ parent, field)

    Makie.tricontourf!(plot, xs, ys, zs; plot.attributes...)
end

Makie.Colorbar(fig_or_scene, plot::FieldContourf; kwargs...) =
    Makie.Colorbar(fig_or_scene, plot.plots[1]; kwargs...)




@recipe(Viz, space, scalars) do scene
    return Makie.Attributes(;
        colormap = :balance,
        shading = false,
        colorrange = Makie.automatic,
    )
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField2D) =
    Viz{<:Tuple{ClimaCore.Fields.SpectralElementField2D}}


function Makie.convert_arguments(
    ::Type{<:Viz},
    field::ClimaCore.Fields.SpectralElementField2D,
)
    if !(eltype(field) <: Union{Float64, Float32, ClimaCore.Geometry.WVector})
        error("plotting only implemented for F64, F32 scalar fields")
    end
    return ClimaCore.axes(field), Float32.(vec(parent(field)))
end

function Makie.plot!(
    plot::Viz{
        <:Tuple{<:ClimaCore.Spaces.SpectralElementSpace2D, Vector{Float32}},
    },
)
    space = plot.space
    # Only update vertices if global_geometry updates (is this the correct value to check for changes?)
    global_geometry =
        lift(x -> x.global_geometry, space; ignore_equal_values = true)

    vertices = lift(global_geometry) do gg
        cartesian =
            ClimaCore.Geometry.Cartesian123Point.(
                ClimaCore.Fields.coordinate_field(space[]),
                Ref(gg),
            )
        # Use the OpenGL native Point3f type to avoid conversions
        return Point3f.(
            vec(parent(cartesian.x1)),
            vec(parent(cartesian.x2)),
            vec(parent(cartesian.x3)),
        )
    end

    # Triangles stay static if the size doesn't change, so we lift and only trigger updates if size changes:
    space_size =
        lift(x -> size(x.local_geometry), space; ignore_equal_values = true)

    triangles = lift(space_size) do (Ni, Nj, _, _, Nh)
        a, b, c = ClimaCore.Spaces.triangles(Ni, Nj, Nh)
        # Use the native GPU types to avoid copies/conversions
        return GLTriangleFace.(a, b, c)
    end

    Makie.mesh!(
        plot,
        vertices,
        triangles,
        color = plot.scalars,
        colormap = plot.colormap,
        shading = plot.shading,
        colorrange = plot.colorrange,
    )
end

end # module
