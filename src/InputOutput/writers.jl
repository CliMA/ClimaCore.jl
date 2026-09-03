abstract type AbstractWriter end

"""
    layout_string(values)

Return the layout string of the `DataLayout` `values` (`"DataF"`, `"VIJFH"`, or
`"VIJHF"`), stored as the `data_layout` attribute of a dataset in an HDF5 file.
The strings match the layout names used by earlier versions of ClimaCore, so
files written by [`HDF5Writer`](@ref) remain readable by them. Other layouts
throw an error.
"""
layout_string(values) =
    values isa DataLayouts.DataF ? "DataF" :
    values isa DataLayouts.VIJFH ? "VIJFH" :
    values isa DataLayouts.VIJHF ? "VIJHF" :
    error("Cannot write layout $(typeof(values)) to an HDF5 file")

# Axis of the `H` dimension in the parent array of a `VIJHWithF` layout
parent_h_dim(values::DataLayouts.VIJHWithF) =
    something(DataLayouts.f_dim(values), 5) == 5 ? 4 : 5

"""
    HDF5Writer(filename::AbstractString,
               context::ClimaComms.AbstractCommsContext;
               overwrite::Bool = true)
    HDF5Writer(::Function,
               filename::AbstractString,
               context::ClimaComms.AbstractCommsContext;
               overwrite::Bool = true)

Open `filename` for writing ClimaCore objects with the ClimaCore HDF5 storage
conventions. Objects are written with [`write!`](@ref) and read back with
[`HDF5Reader`](@ref). The writer caches the domains, meshes, topologies, and
grids it has written, so each is stored once per file.

# Arguments

  - `filename`: Path of the HDF5 file.
  - `context`: The `ClimaComms` context of the run (`ClimaComms.context()`). For
    distributed fields it is the `MPICommsContext` the fields are distributed
    with; the file is then opened with MPI-IO.

# Keyword Arguments

  - `overwrite = true`: Replace an existing file. With `overwrite = false`, an
    existing file is opened for appending, and a missing file is created.

The `do`-block form passes the writer to the function and closes the file when
the function returns. Both forms require `context`.

!!! note

    The default Julia HDF5 binaries are built without MPI support. Writing with
    an `MPICommsContext` requires HDF5.jl configured with an MPI-enabled HDF5
    library; see [the HDF5.jl
    documentation](https://juliaio.github.io/HDF5.jl/stable/#Parallel-HDF5).

# Examples

```julia
InputOutput.HDF5Writer(filename, ClimaComms.context()) do writer
    InputOutput.write!(writer, Y, "Y")
end
```
"""
struct HDF5Writer{C <: ClimaComms.AbstractCommsContext} <: AbstractWriter
    file::HDF5.File
    context::C
    # written object => its group name; identity-keyed so that two distinct
    # objects sharing a default name (e.g. a CG and a DG grid, both
    # "horizontal_grid") get distinct groups instead of silently aliasing
    cache::IdDict{Any, String}
    # requested name => times requested, for uniquifying repeated names
    namecounts::Dict{String, Int}
end

function HDF5Writer(
    f::Function,
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext;
    overwrite::Bool = true,
)

    writer = HDF5Writer(filename, context; overwrite)
    try
        f(writer)
    finally
        Base.close(writer)
    end
end

function HDF5Writer(
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext;
    overwrite::Bool = true,
)
    mode = overwrite ? "w" : "cw"

    if context isa ClimaComms.SingletonCommsContext
        file = h5open(filename, mode)
    else
        file = h5open(filename, mode, context.mpicomm)
    end
    # Add an attribute to the file if it doesn't already exist
    if haskey(attributes(file), "ClimaCore version")
        file_version = VersionNumber(attrs(file)["ClimaCore version"])
        current_version = VERSION
        if file_version != current_version
            @warn "$filename was written using a different version of ClimaCore than is currently loaded" file_version current_version
        end
    else
        write_attribute(file, "ClimaCore version", string(VERSION))
    end
    return HDF5Writer(file, context, IdDict{Any, String}(), Dict{String, Int}())
end

function Base.close(hdfwriter::HDF5Writer)
    empty!(hdfwriter.cache)
    empty!(hdfwriter.namecounts)
    close(hdfwriter.file)
    return nothing
end

"""
    write_attributes!(writer::HDF5Writer, name::AbstractString, data::Dict)

Write the key-value pairs of `data` as attributes of the object at path `name`
in the file of `writer`.
"""
write_attributes!(writer::HDF5Writer, name::AbstractString, data::Dict) =
    h5writeattr(writer.file.filename, name, data)

function cartesianindices_to_matrix(elemorder)
    m, n = length(elemorder), length(eltype(elemorder))
    elemordermatrix = zeros(Int, m, n)
    for (i, order) in enumerate(elemorder)
        for j in 1:n
            elemordermatrix[i, j] = order[j]
        end
    end
    return elemordermatrix
end

"""
    write!(writer::HDF5Writer, obj[, name])

Write a domain, mesh, topology, or grid `obj` to the file of `writer` and return
the name it is stored under. `name` defaults to [`defaultname`](@ref).

Each object is written once per file: writing an object that is already in the
cache of `writer` leaves the file unchanged and returns the name it was first
stored under. Distinct objects that request the same name (e.g. two
spectral-element grids that differ only in a constructor flag, both named
`"horizontal_grid"`) are stored in distinct groups; the second and later ones
get a `_2`, `_3`, ... suffix. References between objects use the returned name.
`Field`s and `FieldVector`s are written with the three-argument method below,
are not cached, and require an explicit `name`.
"""
function write!(writer::HDF5Writer, obj, name = defaultname(obj))
    get!(writer.cache, obj) do
        write_new!(writer, obj, unique_name!(writer, name))
    end
end

# First request for a name keeps it; later requests (necessarily for distinct
# objects, since identical objects hit the cache) get a numbered suffix.
function unique_name!(writer::HDF5Writer, name::AbstractString)
    n = get(writer.namecounts, name, 0) + 1
    writer.namecounts[name] = n
    return n == 1 ? String(name) : string(name, "_", n)
end

# Domains
"""
    defaultname(obj)

Return the default name under which [`write!`](@ref) stores a domain, mesh,
topology, or grid, e.g. `"sphere"`, `"z-interval"`, `"cubedsphere"`, or
`"horizontal_grid"`. `Field`s and `FieldVector`s have no default name.
"""
function defaultname end
defaultname(::Domains.SphereDomain) = "sphere"
function defaultname(domain::Domains.IntervalDomain)
    Domains.coordinate_type(domain) <: Geometry.XPoint && return "x-interval"
    Domains.coordinate_type(domain) <: Geometry.YPoint && return "y-interval"
    Domains.coordinate_type(domain) <: Geometry.ZPoint && return "z-interval"
    return "interval"
end

"""
    write_new!(writer::HDF5Writer, obj, name::AbstractString = defaultname(obj))

Write `obj` to the file of `writer` under `name`, bypassing the cache, and
return `name`. Methods exist for `Domains.IntervalDomain`,
`Domains.SphereDomain`, `Meshes.IntervalMesh`, `Meshes.RectilinearMesh`,
`Meshes.AbstractCubedSphere`, `Topologies.IntervalTopology`,
`Topologies.Topology2D`, and the grid types `Grids.SpectralElementGrid1D`,
`Grids.SpectralElementGrid2D`, `Grids.FiniteDifferenceGrid`,
`Grids.MultiPointGrid`, `Grids.ExtrudedFiniteDifferenceGrid`, and
`Grids.LevelGrid`. Each method creates a group under `domains/`, `meshes/`,
`topologies/`, or `grids/`, stores the object's parameters as attributes, and
writes the objects it depends on (e.g. the mesh of a topology) with
[`write!`](@ref), storing their names as attributes.

Called from [`write!`](@ref), which supplies a unique `name`.
"""
function write_new!(
    writer::HDF5Writer,
    domain::Domains.IntervalDomain,
    name::AbstractString = defaultname(domain),
)
    group = create_group(writer.file, "domains/$name")
    write_attribute(group, "type", "IntervalDomain")
    write_attribute(
        group,
        "coord_type",
        string(nameof(typeof(domain.coord_min))),
    )
    write_attribute(group, "coord_min", Geometry.component(domain.coord_min, 1))
    write_attribute(group, "coord_max", Geometry.component(domain.coord_max, 1))
    !isnothing(domain.boundary_names) && write_attribute(
        group,
        "boundary_names",
        [String(bname) for bname in domain.boundary_names],
    )
    return name
end

function write_new!(
    writer::HDF5Writer,
    domain::Domains.SphereDomain,
    name::AbstractString = defaultname(domain),
)
    group = create_group(writer.file, "domains/$name")
    write_attribute(group, "type", "SphereDomain")
    write_attribute(group, "radius", domain.radius)
    return name
end

# Meshes
defaultname(mesh::Meshes.IntervalMesh) = defaultname(mesh.domain)
defaultname(::Meshes.RectilinearMesh) = "rectilinear"
defaultname(::Meshes.AbstractCubedSphere) = "cubedsphere"

function write_new!(
    writer::HDF5Writer,
    mesh::Meshes.IntervalMesh,
    name::AbstractString = defaultname(mesh),
)
    domainname = write!(writer, mesh.domain)
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", "IntervalMesh")
    write_attribute(group, "domain", domainname)
    write_attribute(group, "nelements", Meshes.nelements(mesh))
    write_attribute(group, "reverse_mode", mesh.reverse_mode)
    if occursin("LinRange", string(typeof(mesh.faces)))
        write_attribute(group, "faces_type", "Range")
    else
        write_attribute(group, "faces_type", "Array")
        write_attribute(
            group,
            "faces",
            [getfield(mesh.faces[i], 1) for i in 1:length(mesh.faces)],
        )
    end
    (; stretch) = mesh
    write_attribute(group, "stretch_type", string(nameof(typeof(stretch))))
    fns = fieldnames(typeof(stretch))
    if !isempty(fns)
        vals = map(fns) do fn
            getfield(stretch, fn)
        end
        write_attribute(group, "stretch_params", [vals...])
    end
    return name
end

function write_new!(
    writer::HDF5Writer,
    mesh::Meshes.RectilinearMesh,
    name::AbstractString = defaultname(mesh),
)
    domainname1 = write!(writer, mesh.intervalmesh1)
    domainname2 = write!(writer, mesh.intervalmesh2)
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", "RectilinearMesh")
    write_attribute(group, "intervalmesh1", domainname1)
    write_attribute(group, "intervalmesh2", domainname2)
    return name
end

function write_new!(
    writer::HDF5Writer,
    mesh::Meshes.AbstractCubedSphere,
    name::AbstractString = defaultname(mesh),
)
    domainname = write!(writer, mesh.domain)
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", string(nameof(typeof(mesh))))
    write_attribute(group, "ne", Meshes.n_elements_per_panel_direction(mesh))
    write_attribute(
        group,
        "localelementmap",
        string(nameof(typeof(mesh.localelementmap))),
    )
    write_attribute(group, "domain", domainname)
    return name
end

# Topologies
defaultname(::Topologies.Topology2D) = "2d"
defaultname(topology::Topologies.IntervalTopology) = defaultname(topology.mesh)

function write_new!(
    writer::HDF5Writer,
    topology::Topologies.IntervalTopology,
    name::AbstractString = defaultname(topology),
)
    meshname = write!(writer, topology.mesh)
    group = create_group(writer.file, "topologies/$name")
    write_attribute(group, "type", "IntervalTopology")
    write_attribute(group, "mesh", meshname)
    return name
end

function write_new!(
    writer::HDF5Writer,
    topology::Topologies.Topology2D,
    name::AbstractString = defaultname(topology),
)
    @assert writer.context == topology.context

    group = create_group(writer.file, "topologies/$name")
    write_attribute(group, "type", "Topology2D")
    write_attribute(group, "mesh", write!(writer, topology.mesh))
    if !(topology.elemorder isa CartesianIndices)
        elemorder_matrix = reinterpret(reshape, Int, topology.elemorder)
        if writer.context isa ClimaComms.SingletonCommsContext
            write_dataset(group, "elemorder", elemorder_matrix)
        else
            elemorder_dataset = create_dataset(
                group,
                "elemorder",
                datatype(eltype(elemorder_matrix)),
                dataspace(size(elemorder_matrix));
                dxpl_mpio = :collective,
            )
            elemorder_dataset[:, topology.local_elem_gidx] =
                elemorder_matrix[:, topology.local_elem_gidx]
        end
    end
    return name
end

# Grids
#
defaultname(::DataLayouts.IJHMask) = "IJHMask"
defaultname(::Grids.SpectralElementGrid1D) = "horizontal_grid"
defaultname(::Grids.SpectralElementGrid2D) = "horizontal_grid"
defaultname(::Grids.ExtrudedFiniteDifferenceGrid) = "extruded_finite_difference_grid"
defaultname(grid::Grids.FiniteDifferenceGrid) = defaultname(grid.topology)
defaultname(::Grids.MultiPointGrid) = "multi_point_grid"
defaultname(grid::Grids.LevelGrid) = "$(defaultname(grid.full_grid)): level $(grid.level)"

function write_new!(
    writer::HDF5Writer,
    grid::Grids.SpectralElementGrid1D,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "SpectralElementGrid1D")
    write_attribute(
        group,
        "quadrature_type",
        string(nameof(typeof(Spaces.quadrature_style(grid)))),
    )
    write_attribute(
        group,
        "quadrature_num_points",
        Quadratures.degrees_of_freedom(Spaces.quadrature_style(grid)),
    )
    write_attribute(
        group,
        "discretization",
        grid.discretization isa Grids.DG ? "DG" : "CG",
    )
    write_attribute(group, "topology", write!(writer, Spaces.topology(grid)))
    return name
end

function write_new!(
    writer::HDF5Writer,
    grid::Grids.SpectralElementGrid2D,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "SpectralElementGrid2D")
    write_attribute(
        group,
        "quadrature_type",
        string(nameof(typeof(Spaces.quadrature_style(grid)))),
    )
    write_attribute(
        group,
        "quadrature_num_points",
        Quadratures.degrees_of_freedom(Spaces.quadrature_style(grid)),
    )
    write_attribute(group, "bubble", grid.enable_bubble ? "true" : "false")
    write_attribute(
        group,
        "discretization",
        grid.discretization isa Grids.DG ? "DG" : "CG",
    )
    write_attribute(group, "topology", write!(writer, Spaces.topology(grid)))
    if !(grid.mask isa DataLayouts.NoMask)
        write_attribute(
            group,
            "grid_mask",
            write!(writer, grid.mask, Spaces.topology(grid)),
        )
    end
    return name
end

function write_new!(
    writer::HDF5Writer,
    grid::Grids.FiniteDifferenceGrid,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "FiniteDifferenceGrid")
    write_attribute(group, "topology", write!(writer, Spaces.topology(grid)))
    return name
end

function write_new!(
    writer::HDF5Writer,
    grid::Grids.MultiPointGrid,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "MultiPointGrid")
    write_attribute(group, "radius", grid.global_geometry.radius)
    # 2×N matrix of (lat, long) coordinates, one column per point
    coords = Array(parent(grid.local_geometry.coordinates))
    write_dataset(group, "points", coords[1, 1, 1, :, :])
    return name
end


function write_new!(
    writer::HDF5Writer,
    grid::Grids.ExtrudedFiniteDifferenceGrid,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "ExtrudedFiniteDifferenceGrid")
    write_attribute(
        group,
        "horizontal_grid",
        write!(writer, grid.horizontal_grid),
    )
    write_attribute(group, "vertical_grid", write!(writer, grid.vertical_grid))
    hypsography = grid.hypsography
    if hypsography isa Hypsography.LinearAdaption
        write_attribute(group, "hypsography_type", "LinearAdaption")
        write_attribute(
            group,
            "hypsography_surface",
            write!(writer, hypsography.surface, "_z_surface/$name"), # Change to save "space.hyps"
        )
    elseif hypsography isa Hypsography.SLEVEAdaption
        write_attribute(group, "hypsography_type", "SLEVEAdaption")
        write_attribute(group, "hypsography_ηₕ", hypsography.ηₕ)
        write_attribute(group, "hypsography_s", hypsography.s)
        write_attribute(
            group,
            "hypsography_surface",
            write!(writer, hypsography.surface, "_z_surface/$name"),
        )
    end
    write_attribute(
        group,
        "deep",
        grid.global_geometry isa Geometry.DeepSphericalGlobalGeometry,
    )
    return name
end


function write_new!(
    writer::HDF5Writer,
    grid::Grids.LevelGrid,
    name::AbstractString = defaultname(grid),
)
    group = create_group(writer.file, "grids/$name")
    write_attribute(group, "type", "LevelGrid")
    write_attribute(group, "full_grid", write!(writer, grid.full_grid))
    if grid.level isa PlusHalf
        write_attribute(group, "level_half", grid.level - half)
    else
        write_attribute(group, "level", grid.level)
    end
    return name
end

function write!(
    writer::HDF5Writer,
    mask::DataLayouts.IJHMask,
    topology::Topologies.AbstractTopology,
    name::AbstractString = defaultname(mask),
)
    get!(writer.cache, mask) do
        uname = unique_name!(writer, name)
        group = create_group(writer.file, "grid_mask/$uname")
        write!(writer, group, mask.is_active, "is_active", topology)
        uname
    end
end

# write fields
"""
    write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)
    write!(writer::HDF5Writer, fieldvector::Fields.FieldVector, name::AbstractString)

Write `field` or `fieldvector` to the file of `writer` under `name` and return
`name`.

A `Field` is stored as the dataset `fields/<name>`, with its data layout,
element type, grid name, and staggering as attributes; the grid is written
with [`write!`](@ref) if it is not in the file yet. A `FieldVector` is stored
as the group `fields/<name>`, and each component is written as a `Field` (or a
nested `FieldVector`) under `<name>/<key>`, so the component `Y.c` of a
`FieldVector` named `"Y"` is stored as `"Y/c"`. `Field`s and `FieldVector`s are
not cached and can be written more than once under different names.
"""
function write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)
    write!(writer, field, name, axes(field))
end

"""
    write!(
        writer::HDF5Writer,
        field::Fields.Field,
        name::AbstractString,
        space::Spaces.AbstractPointSpace,
    )

Write a `Field` on a `PointSpace` to the file of `writer`. The field data is
stored as the dataset `fields/<name>` and the local geometry data of the space
as the dataset `local_geometry_data/<name>`, since a `PointSpace` has no grid
to reference.
"""
function write!(
    writer::HDF5Writer,
    field::Fields.Field,
    name::AbstractString,
    space::Spaces.AbstractPointSpace,
)
    array = parent(field)
    lg_data = Grids.local_geometry_data(space)
    lg_type = Grids.local_geometry_type(typeof(space))
    lg_array = parent(lg_data)
    dataset = create_dataset(
        writer.file,
        "fields/$name",
        datatype(eltype(array)),
        dataspace(size(array)),
    )
    dataset[:] = array
    write_attribute(dataset, "type", "Field")
    write_attribute(
        dataset,
        "data_layout",
        layout_string(Fields.field_values(field)),
    )
    write_attribute(dataset, "field_eltype", string(eltype(field)))
    local_geometry_dataset = create_dataset(
        writer.file,
        "local_geometry_data/$name",
        datatype(eltype(array)),
        dataspace(size(lg_array)),
    )
    local_geometry_dataset[:] = lg_array
    write_attribute(
        local_geometry_dataset,
        "local_geometry_type",
        string(lg_type),
    )
end

"""
    write!(
        writer::HDF5Writer,
        group,
        values::DataLayouts.DataLayout,
        name::AbstractString,
        topology::Topologies.AbstractTopology,
    )

Write the `DataLayout` `values` as the dataset `name` in the HDF5 `group`.
`topology` is the horizontal topology the data is laid out on; for a
`Topology2D` with a distributed `writer` context, each rank writes its own
elements with `_write_mpi!`, otherwise the whole array is written with
`_write!`. Used for grid masks.
"""
function write!(
    writer::HDF5Writer,
    group,
    values::DataLayouts.DataLayout,
    name::AbstractString,
    topology::Topologies.AbstractTopology,
)
    if topology isa Topologies.Topology2D &&
       !(writer.context isa ClimaComms.SingletonCommsContext)
        nelems = Topologies.nelems(topology)
        (; local_elem_gidx) = topology
        _write_mpi!(group, values, name; nelems, local_elem_gidx)
    else
        _write!(group, values, name)
    end
end

function write_plain_array!(group, array::AbstractArray, name::AbstractString)
    array_cpu = array isa Array ? array : Array(array)
    nd = ndims(array_cpu)
    dims = size(array_cpu)
    localidx = ntuple(d -> (:), nd)
    dataset =
        create_dataset(group, name, datatype(eltype(array_cpu)), dataspace(dims))
    dataset[localidx...] = array_cpu
    return dataset
end

"""
    _write_mpi!(group, values::DataLayouts.DataLayout, name; nelems, local_elem_gidx)

Write the distributed `DataLayout` `values` as the dataset `data/<name>` in the
HDF5 `group` with a collective MPI write. The dataset holds all `nelems` global
elements along the `H` axis; this rank writes the elements at the global
indices `local_elem_gidx`. Return `name`.
"""
function _write_mpi!(
    group,
    values::DataLayouts.DataLayout,
    name::AbstractString;
    nelems,
    local_elem_gidx,
)
    h_dim = parent_h_dim(values)
    array = parent(values)
    nd = ndims(array)
    dims = ntuple(d -> d == h_dim ? nelems : size(array, d), nd)
    localidx = ntuple(d -> d == h_dim ? local_elem_gidx : (:), nd)
    dataset = create_dataset(
        group,
        "data/$name",
        datatype(eltype(array)),
        dataspace(dims);
        dxpl_mpio = :collective,
    )
    dataset[localidx...] = array
    write_attribute(dataset, "data_layout", layout_string(values))
    write_attribute(dataset, "data_eltype", string(eltype(values)))
    return name
end

"""
    _write!(group, values::DataLayouts.DataLayout, name::AbstractString)

Write the whole parent array of `values` as the dataset `name` in the HDF5
`group`, for data that is not distributed. Return `name`.
"""
function _write!(group, values::DataLayouts.DataLayout, name::AbstractString;)
    array = parent(values)
    dataset = write_plain_array!(group, array, name)
    write_attribute(dataset, "type", layout_string(values))
    write_attribute(dataset, "data_eltype", string(eltype(values)))
    return name
end


"""
    write!(
        writer::HDF5Writer,
        field::Fields.Field,
        name::AbstractString,
        space::Spaces.AbstractSpace,
    )

Write a `Field` on `space` as the dataset `fields/<name>` and return `name`.
The grid of `space` is written first with [`write!`](@ref), and its name is
stored in the `grid` attribute of the dataset, together with the data layout,
element type, and staggering of the field. With a distributed `writer` context
on a `Topology2D`, each rank writes its own elements with a collective MPI
write.
"""
function write!(
    writer::HDF5Writer,
    field::Fields.Field,
    name::AbstractString,
    space::Spaces.AbstractSpace,
)
    values = Fields.field_values(field)
    array = parent(field)
    nd = ndims(array)

    staggering = Spaces.staggering(space)
    grid = Spaces.grid(space)
    grid_name = write!(writer, grid)

    # topology is only queried on the distributed path: point-cloud spaces
    # have no topology and are single-process only
    if !(writer.context isa ClimaComms.SingletonCommsContext) &&
       Spaces.topology(space) isa Topologies.Topology2D
        topology = Spaces.topology(space)
        nelems = Topologies.nelems(topology)
        f_dim = DataLayouts.f_dim(values)
        h_dim = isnothing(f_dim) || f_dim == 5 ? 4 : 5
        dims = ntuple(d -> d == h_dim ? nelems : size(array, d), nd)
        localidx = ntuple(d -> d == h_dim ? topology.local_elem_gidx : (:), nd)
        dataset = create_dataset(
            writer.file,
            "fields/$name",
            datatype(eltype(array)),
            dataspace(dims);
            dxpl_mpio = :collective,
        )
    else
        dims = size(array)
        localidx = ntuple(d -> (:), nd)
        dataset = create_dataset(
            writer.file,
            "fields/$name",
            datatype(eltype(array)),
            dataspace(dims),
        )
    end
    dataset[localidx...] = array
    write_attribute(dataset, "type", "Field")
    write_attribute(
        dataset,
        "data_layout",
        layout_string(Fields.field_values(field)),
    )
    write_attribute(dataset, "field_eltype", string(eltype(field)))
    write_attribute(dataset, "grid", grid_name)
    if !isnothing(staggering)
        write_attribute(
            dataset,
            "staggering",
            string(nameof(typeof(staggering))),
        )
    end

    return name
end

function write!(
    writer::HDF5Writer,
    fieldvector::Fields.FieldVector,
    name::AbstractString,
)
    group = create_group(writer.file, "fields/$name")
    write_attribute(group, "type", "FieldVector")
    for (key, component) in pairs(Fields._values(fieldvector))
        write!(writer, component, "$name/$key")
    end
    return name
end


"""
    write!(writer::HDF5Writer, name => value...)

Write one or more `name => value` pairs to `writer`, as
`write!(writer, value, name)` for each pair. Return `nothing`.
"""
function write!(writer::HDF5Writer, pairs::Pair...)
    for (name, value) in pairs
        write!(writer, value, name)
    end
    return nothing
end


"""
    write!(filename::AbstractString, name => value...)

Open an [`HDF5Writer`](@ref) on `filename`, write one or more `name => value`
pairs to it, and close the file.
"""
function write!(filename::AbstractString, pairs::Pair...)
    hdfwriter = HDF5Writer(filename)
    try
        write!(hdfwriter, pairs...)
    finally
        Base.close(hdfwriter)
    end
end
