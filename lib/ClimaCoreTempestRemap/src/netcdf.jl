
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
    def_space_coord(nc::NCDataset, space::Spaces.AbstractSpace; type = "dgll")

Add spatial dimensions for `space` in the NetCDF dataset `nc`, compatible with
the type used by [`remap_weights`](@ref).

If a compatible dimension already exists, it will be reused.
"""
function def_space_coord(
    nc::NCDataset,
    space::Spaces.SpectralElementSpace2D;
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
    type = "dgll",
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
    elseif nc.attrib["node_type"] == "dgll"
        nodes = Spaces.all_nodes(space)
    else
        error("unsupported node type")
    end
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


function first_center_space(fv::Fields.FieldVector)
    for prop_chain in Fields.property_chains(fv)
        f = Fields.single_field(fv, prop_chain)
        space = axes(f)
        if space isa Spaces.CenterExtrudedFiniteDifferenceSpace
            return space
        end
    end
    error("Unfound space")
end

function first_face_space(fv::Fields.FieldVector)
    for prop_chain in Fields.property_chains(fv)
        f = Fields.single_field(fv, prop_chain)
        space = axes(f)
        if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
            return space
        end
    end
    error("Unfound space")
end

function process_name_default(s::AbstractString)
    s = replace(s, "components_data_" => "")
    return s
end

function remap2latlon(
    Y::Fields.FieldVector;
    t_now = 0.0,
    nc_dir = pwd(),
    nlat = 90,
    nlon = 180,
    filename = "tempremaptest.nc",
    process_name = process_name_default,
    center_space = first_center_space,
    face_space = first_face_space,
)
    cspace = center_space(Y)
    fspace = face_space(Y)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(hspace.quadrature_style)

    # create a temporary dir for intermediate data
    remap_tmpdir = joinpath(nc_dir, "remaptmp")
    mkpath(remap_tmpdir)
    FT = eltype(Y)

    ### create an nc file to store raw cg data
    # create data
    datafile_cc = joinpath(remap_tmpdir, "test.nc")

    varname(pc::Tuple) = process_name(join(pc, "_"))

    NCDatasets.NCDataset(datafile_cc, "c") do nc
        # defines the appropriate dimensions and variables for a space coordinate
        # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
        nc_time = def_time_coord(nc)
        def_space_coord(nc, cspace, type = "cgll")
        def_space_coord(nc, fspace, type = "cgll")
        # define variables for the prognostic states
        for prop_chain in Fields.property_chains(Y)
            f = Fields.single_field(Y, prop_chain)
            space = axes(f)
            nc_var = defVar(nc, varname(prop_chain), FT, space, ("time",))
            nc_var[:, 1] = f
        end
        # TODO: interpolate w onto center space and save it the same way as the other vars
        nc_time[1] = t_now
    end
    varnames = varname.(Fields.property_chains(Y))

    # write out our cubed sphere mesh
    meshfile_cc = joinpath(remap_tmpdir, "mesh_cubedsphere.g")
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = joinpath(remap_tmpdir, "mesh_rll.g")
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = joinpath(remap_tmpdir, "mesh_overlap.g")
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = joinpath(remap_tmpdir, "remap_weights.nc")
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    datafile_latlon = joinpath(nc_dir, filename)

    # TODO: add an option to silence tempest?
    open(tempname(), "w") do io # silence tempest
        redirect_stdout(io) do
            apply_remap(datafile_latlon, datafile_cc, weightfile, varnames)
        end
    end
    # Cleanup
    all_files = [
        joinpath(root, f) for
        (root, dirs, files) in Base.Filesystem.walkdir(nc_dir) for
        f in files
    ]
    for f in all_files
        if endswith(f, ".g")
            rm(f; force = true)
        end
    end
    # rm(datafile_cc; force=true) # TODO: should we clean this up?
    # rmpath(remap_tmpdir; force=true) # TODO: should we clean this up?
    # TODO: Two NC files are created, do we need both?
end
