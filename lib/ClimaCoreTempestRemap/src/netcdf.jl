
import ClimaCore: slab, column

"""
    def_time_coord(nc::NCDataset, length=Inf, eltype=Float64;
        units = "seconds since 2020-01-01 00:00:00"
        kwargs...
    )

Deine a time coordinate (dimension + variable) `"time"` in the NetCDF dataset
`nc`. By default its length is set to be unlimited. The variable corresponding
to the coordinate is returned.

Additional attributes can be added as keyword arguments.

# Example

```julia
timevar = add_time_coord!(nc; units = "seconds since 2020-01-01 00:00:00",)
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
    def_space_coord(nc::NCDataset, space::Spaces.AbstractSpace; type = "cgll")

Add spatial dimensions for `space` in the NetCDF dataset `nc`, compatible with
the type used by [`remap_weights`](@ref).

If a compatible dimension already exists, it will be reused.
"""
function def_space_coord(
    nc::NCDataset,
    space::Spaces.SpectralElementSpace2D;
    type = "cgll",
)
    if type == "cgll"
        nodes = Spaces.unique_nodes(space)
        ncol = length(nodes)
    else
        error("Unsupported type: $type")
    end

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
        coord = slab(coords, e)[i, j]
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
    space::Spaces.ExtrudedFiniteDifferenceSpace{S};
    type = "cgll",
) where {S <: Spaces.Staggering}
    hvar = def_space_coord(nc, space.horizontal_space; type = type)
    vvar = def_space_coord(
        nc,
        Spaces.FiniteDifferenceSpace{S}(space.vertical_topology),
    )
    (hvar..., vvar...)
end

"""
    space_dims(space::Spaces.AbstractSpace)

The names of the NetCDF dimensions used by `space`.
"""
space_dims(space::Spaces.SpectralElementSpace2D) = ("ncol",)
space_dims(space::Spaces.CenterFiniteDifferenceSpace) = ("z",)
space_dims(space::Spaces.FaceFiniteDifferenceSpace) = ("z_half",)
space_dims(space::Spaces.CenterExtrudedFiniteDifferenceSpace) = ("ncol", "z")
space_dims(space::Spaces.FaceExtrudedFiniteDifferenceSpace) = ("ncol", "z_half")

"""
    NCDatasets.defVar(nc::NCDataset, name::String, T, space::AbstractSpace, extradims=())

Define a new variable in `nc` named `name` of element type `T` suitable for
storing a field on `space`, along with any further dimensions specified in
`extradims`. The new variable is returned.
"""
function NCDatasets.defVar(
    nc::NCDataset,
    name,
    T::DataType,
    space::Spaces.AbstractSpace,
    extradims = (),
)
    @assert T <: Real
    defVar(nc, name, Float64, (space_dims(space)..., extradims...))
end

"""
    NCDatasets.defVar(nc::NCDataset, name, field::Field, extradims=())

Define a new variable in `nc` named `name` of suitable for storing `field`,
along with any further dimensions specified in `extradims`. The new variable is
returned.

!!! note
    This does not write any data to the variable.
"""
function NCDatasets.defVar(
    nc::NCDataset,
    name,
    field::Fields.Field,
    extradims = (),
)
    defVar(nc, name, eltype(field), axes(field), extradims)
end

"""
    var[:, extraidx...] = field

Write the data in `field` to a NetCDF variable `var`. `extraidx` are any extra indices of `var`.

Appropriate spatial dimensions should already be defined by [`defVar`](@ref).

```julia
# Given a collection of fields U, write them as a single array to a NetCDF file.
def_space_coord(nc, space)
nc_time = def_time_coord(nc)
nc_u = defVar(nc, "u", Float64, space, ("time",))
for (i,t) in enumerate(times)
    nc_time[i] = t
    nc_u[:,i] = U[i]
end
```
"""
Base.setindex!(::NCDatasets.CFVariable, ::Fields.Field, ::Colon)

function Base.setindex!(
    var::NCDatasets.CFVariable,
    field::Fields.SpectralElementField2D,
    ::Colon,
    extraidx::Int...,
)
    space = axes(field)
    nc = NCDataset(var)
    if nc.attrib["node_type"] == "cgll"
        nodes = Spaces.unique_nodes(space)
    else
        error("node_type not supported")
    end

    nodes = Spaces.unique_nodes(space)
    data = Fields.field_values(field)
    for (col, ((i, j), e)) in enumerate(nodes)
        var[col, extraidx...] = slab(data, e)[i, j]
    end
    return var
end
function Base.setindex!(
    var::NCDatasets.CFVariable,
    field::Fields.ExtrudedFiniteDifferenceField,
    ::Colon,
    extraidx::Int...,
)
    nc = NCDataset(var)
    space = axes(field)
    hspace = space.horizontal_space
    if nc.attrib["node_type"] == "cgll"
        nodes = Spaces.unique_nodes(hspace)
    else
        error("node_type not supported")
    end
    data = Fields.field_values(field)
    for (col, ((i, j), h)) in enumerate(nodes)
        var[col, :, extraidx...] = parent(column(data, i, j, h))
    end
    return var
end
