import CommonDataModel
import ClimaCore: slab, column


"""
    def_time_coord(nc::NCDataset, length = Inf, eltype = Float64;
        standard_name = "time", long_name = "time", axis = "T", kwargs...)

Define a time coordinate (dimension and variable) `"time"` in the NetCDF dataset `nc` and
return the variable. By default its length is unlimited.

The keyword arguments `standard_name`, `long_name`, and `axis` set the corresponding
attributes; any further keyword arguments are added as attributes as well.

# Examples

```julia
timevar = def_time_coord(nc; units = "seconds since 2020-01-01 00:00:00")
timevar[:] = collect(0.0:0.5:60)
```
"""
function def_time_coord(
    nc::NCDataset,
    length = Inf,
    eltype = Float64;
    standard_name = "time",
    long_name = "time",
    axis = "T",
    kwargs...,
)
    defDim(nc, "time", length)
    time = defVar(nc, "time", eltype, ("time",))
    time.attrib["standard_name"] = standard_name
    time.attrib["long_name"] = long_name
    time.attrib["axis"] = axis
    for (k, v) in kwargs
        time.attrib[String(k)] = v
    end
    return time
end


"""
    def_space_coord(nc::NCDataset, space::Spaces.AbstractSpace; type = "dgll")

Define the spatial dimensions and coordinate variables for `space` in the NetCDF dataset
`nc` and return the coordinate variables as a tuple.

For horizontal and extruded spaces, `type` is the node type used by
[`remap_weights`](@ref), `"cgll"` (unique nodes) or `"dgll"` (all nodes); it is recorded
in the `node_type` attribute of `nc`. Horizontal coordinates are `X`/`Y` [m] on
rectilinear meshes and `lat`/`lon` [degrees] otherwise; vertical coordinates are `z`
for center spaces and `z_half` for face spaces [m]. If a compatible dimension already
exists, it is reused; if one with a different size exists, an error is thrown.
"""
def_space_coord(
    nc::NCDataset,
    space::Spaces.SpectralElementSpace2D;
    type = "dgll",
) = def_space_coord(nc, space, Spaces.topology(space).mesh; type)

function def_space_coord(
    nc::NCDataset,
    space::Spaces.SpectralElementSpace2D,
    ::Meshes.RectilinearMesh;
    type = "dgll",
)
    if type == "cgll"
        nodes = Spaces.unique_nodes(space)
    elseif type == "dgll"
        nodes = Spaces.all_nodes(space)
    else
        error("Unsupported type: $type")
    end
    ncol = length(nodes)

    if haskey(nc, "Y")
        # dimension already exists: check correct size
        if size(nc["Y"]) != (ncol,)
            error("incompatible horizontal dimension already exists")
        end
        return (nc["X"], nc["Y"])
    end

    # # dimensions
    defDim(nc, "ncol", ncol)

    # variables
    ## X
    X = defVar(nc, "X", Float64, ("ncol",))
    X.attrib["units"] = "m"
    X.attrib["axis"] = "X"
    X.attrib["long_name"] = "x-coordinate in Cartesian system"

    ## lat
    Y = defVar(nc, "Y", Float64, ("ncol",))
    Y.attrib["units"] = "m"
    Y.attrib["axis"] = "Y"
    Y.attrib["long_name"] = "y-coordinate in Cartesian system"

    coords = Spaces.coordinates_data(space)

    for (col, ((i, j), e)) in enumerate(nodes)
        coord = slab(coords, e)[1, i, j, 1]
        X[col] = coord.x
        Y[col] = coord.y
    end
    nc.attrib["node_type"] = type
    return (X, Y)
end

function def_space_coord(
    nc::NCDataset,
    space::Spaces.SpectralElementSpace2D,
    mesh::Meshes.AbstractMesh2D;
    type = "dgll",
)
    if type == "cgll"
        nodes = Spaces.unique_nodes(space)
    elseif type == "dgll"
        nodes = Spaces.all_nodes(space)
    else
        error("Unsupported type: $type")
    end
    ncol = length(nodes)

    if haskey(nc, "lon")
        # dimension already exists: check correct size
        if size(nc["lon"]) != (ncol,)
            error("incompatible horizontal dimension already exists")
        end
        return (nc["lat"], nc["lon"])
    end

    # dimensions
    defDim(nc, "ncol", ncol)

    # variables
    ## lon
    lon = defVar(nc, "lon", Float64, ("ncol",))
    lon.attrib["units"] = "degrees_east"
    lon.attrib["axis"] = "X"
    lon.attrib["long_name"] = "longitude"
    lon.attrib["standard_name"] = "longitude"

    ## lat
    lat = defVar(nc, "lat", Float64, ("ncol",))
    lat.attrib["units"] = "degrees_north"
    lat.attrib["axis"] = "Y"
    lat.attrib["long_name"] = "latitude"
    lat.attrib["standard_name"] = "latitude"

    coords = Spaces.coordinates_data(space)

    for (col, ((i, j), e)) in enumerate(nodes)
        coord = slab(coords, e)[1, i, j, 1]
        lon[col] = coord.long
        lat[col] = coord.lat
    end
    nc.attrib["node_type"] = type
    return (lat, lon)
end

function def_space_coord(
    nc::NCDataset,
    space::Spaces.CenterFiniteDifferenceSpace,
)
    nlevels = Spaces.nlevels(space)

    if haskey(nc, "z")
        if size(nc["z"]) != (nlevels,)
            error("incompatible vertical dimension already exists")
        end
        return (nc["z"],)
    end

    # dimensions
    defDim(nc, "z", nlevels)
    defDim(nc, "nv", 2)

    # variables
    ## z
    z = defVar(nc, "z", Float64, ("z",))
    z.attrib["units"] = "meters"
    z.attrib["axis"] = "Z"
    z.attrib["positive"] = "up"
    z.attrib["long_name"] = "height"
    z.attrib["standard_name"] = "height"
    z.attrib["bounds"] = "z_bnds"

    z_bnds = defVar(nc, "z_bnds", Float64, ("nv", "z"))

    coords = Spaces.coordinates_data(space)
    z .= parent(coords)
    fcoords = Fields.coordinate_field(Spaces.FaceFiniteDifferenceSpace(space))
    z_bnds[1, :] .= parent(fcoords)[1:(end - 1)]
    z_bnds[2, :] .= parent(fcoords)[2:end]
    return (z,)
end


function def_space_coord(nc::NCDataset, space::Spaces.FaceFiniteDifferenceSpace)
    nlevels = Spaces.nlevels(space)

    if haskey(nc, "z_half")
        if size(nc["z_half"]) != (nlevels,)
            error("incompatible vertical dimension already exists")
        end
        return (nc["z_half"],)
    end

    # dimensions
    defDim(nc, "z_half", nlevels)

    # variables
    ## z_half
    z_half = defVar(nc, "z_half", Float64, ("z_half",))
    z_half.attrib["units"] = "meters"
    z_half.attrib["axis"] = "Z"
    z_half.attrib["positive"] = "up"
    z_half.attrib["long_name"] = "height"
    z_half.attrib["standard_name"] = "height"

    coords = Spaces.coordinates_data(space)
    z_half .= parent(coords)
    return (z_half,)
end

function def_space_coord(
    nc::NCDataset,
    space::Spaces.ExtrudedFiniteDifferenceSpace;
    type = "dgll",
)
    staggering = Spaces.staggering(space)
    hvar = def_space_coord(nc, Spaces.horizontal_space(space); type = type)
    vvar = def_space_coord(
        nc,
        Spaces.FiniteDifferenceSpace(
            Spaces.vertical_topology(space),
            staggering,
        ),
    )
    (hvar..., vvar...)
end

"""
    space_dims(space::Spaces.AbstractSpace)

Return the names of the NetCDF dimensions used by `space`, as defined by
[`def_space_coord`](@ref).
"""
space_dims(space::Spaces.SpectralElementSpace2D) = ("ncol",)
space_dims(space::Spaces.CenterFiniteDifferenceSpace) = ("z",)
space_dims(space::Spaces.FaceFiniteDifferenceSpace) = ("z_half",)
space_dims(space::Spaces.CenterExtrudedFiniteDifferenceSpace) = ("ncol", "z")
space_dims(space::Spaces.FaceExtrudedFiniteDifferenceSpace) = ("ncol", "z_half")

"""
    NCDatasets.defVar(nc::NCDataset, name, T::DataType, space::AbstractSpace, extradims = ())

Define a new variable in `nc` named `name`, suitable for storing a field with element
type `T <: Real` on `space`, along with any further dimensions named in `extradims`, and
return it. The variable is stored as `Float64` regardless of `T`.
"""
function NCDatasets.defVar(
    nc::NCDataset,
    name::NCDatasets.SymbolOrString,
    T::DataType,
    space::Spaces.AbstractSpace,
    extradims = (),
)
    @assert T <: Real
    defVar(nc, name, Float64, (space_dims(space)..., extradims...))
end

"""
    NCDatasets.defVar(nc::NCDataset, name, field::Field, extradims = ())

Define a new variable in `nc` named `name`, suitable for storing `field`, along with any
further dimensions named in `extradims`, and return it.

!!! note

    This does not write any data to the variable; assign `var[:] = field` to do so.
"""
function NCDatasets.defVar(
    nc::NCDataset,
    name::NCDatasets.SymbolOrString,
    field::Fields.Field,
    extradims = (),
)
    defVar(nc, name, eltype(field), axes(field), extradims)
end

"""
    var[:, extraidx...] = field

Write the data in `field` to the NetCDF variable `var` and return `var`. `extraidx` are
the indices of `var` along any extra dimensions, e.g. the time index.

`var` must have been defined by [`defVar`](@ref) on a dataset whose spatial dimensions
were defined by [`def_space_coord`](@ref); the dataset's `node_type` attribute selects
the unique or all-node layout.

# Examples

```julia
# Given a collection of fields U, write them as a single array to a NetCDF file.
def_space_coord(nc, space)
nc_time = def_time_coord(nc)
nc_u = defVar(nc, "u", Float64, space, ("time",))
for (i, t) in enumerate(times)
    nc_time[i] = t
    nc_u[:, i] = U[i]
end
```
"""
Base.setindex!(
    var::Union{NCDatasets.CFVariable, CommonDataModel.CFVariable},
    ::Fields.Field,
    ::Colon,
)

function Base.setindex!(
    var::Union{NCDatasets.CFVariable, CommonDataModel.CFVariable},
    field::Fields.SpectralElementField2D,
    ::Colon,
    extraidx::Int...,
)
    space = axes(field)
    nc = NCDatasets.dataset(var)
    if nc.attrib["node_type"] == "cgll"
        nodes = Spaces.unique_nodes(space)
    elseif nc.attrib["node_type"] == "dgll"
        nodes = Spaces.all_nodes(space)
    else
        error("unsupported node type")
    end
    data = Fields.field_values(field)
    for (col, ((i, j), e)) in enumerate(nodes)
        var[col, extraidx...] = slab(data, e)[1, i, j, 1]
    end
    return var
end
function Base.setindex!(
    var::Union{NCDatasets.CFVariable, CommonDataModel.CFVariable},
    field::Fields.ExtrudedFiniteDifferenceField,
    ::Colon,
    extraidx::Int...,
)
    nc = NCDatasets.dataset(var)
    space = axes(field)
    hspace = Spaces.horizontal_space(space)
    if nc.attrib["node_type"] == "cgll"
        nodes = Spaces.unique_nodes(hspace)
    elseif nc.attrib["node_type"] == "dgll"
        nodes = Spaces.all_nodes(hspace)
    else
        error("unsupported node type")
    end
    data = Fields.field_values(field)
    for (col, ((i, j), h)) in enumerate(nodes)
        var[col, :, extraidx...] = parent(column(data, i, j, h))
    end
    return var
end
