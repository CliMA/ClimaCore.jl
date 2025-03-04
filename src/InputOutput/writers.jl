abstract type AbstractWriter end

"""
    HDF5Writer(filename::AbstractString[,
               context::ClimaComms.AbstractCommsContext];
               overwrite::Bool = true)
    HDF5Writer(::Function,
               filename::AbstractString[,
               context::ClimaComms.AbstractCommsContext];
               overwrite::Bool = true)

An `AbstractWriter` for writing to HDF5-formatted files using the ClimaCore
storage conventions. An internal cache is used to avoid writing duplicate
domains, meshes, topologies and spaces to the file. Use [`HDF5Reader`](@ref) to
load the data from the file.

The optional `context` can be used for writing distributed fields: in this case,
the `MPICommsContext` used passed as an argument: this must match the context
used for distributing the `Field`.

The writer overwrites or appends to existing files depending on the value of the
`overwrite` keyword argument. When `overwrite` is `false`, the writer appends to
`filename` if the file already exists, otherwise it creates a new one.

!!! note

    The default Julia HDF5 binaries are not built with MPI support. To use the
    distributed functionality, you will need to configure HDF5.jl with an
    MPI-enabled HDF5 library, see [the HDF5.jl
    documentation](https://juliaio.github.io/HDF5.jl/stable/#Parallel-HDF5).

# Interface

[`write!`](@ref)

# Usage

```julia
InputOutput.HDF5Writer(filename) do writer
    InputOutput.write!(writer, Y, "Y")
end
```
"""
struct HDF5Writer{C <: ClimaComms.AbstractCommsContext} <: AbstractWriter
    file::HDF5.File
    context::C
    cache::Dict{String, String}
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
    cache = Dict{String, String}()
    return HDF5Writer(file, context, cache)
end

function Base.close(hdfwriter::HDF5Writer)
    empty!(hdfwriter.cache)
    close(hdfwriter.file)
    return nothing
end

"""
    write_attributes!(writer::AbstractWriter, name::AbstractString, data::Dict)

Write `data` as attributes to the object at `name` in the given HDF5 file.
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
    write!(writer::AbstractWriter, obj[, preferredname])

Write the object `obj` using `writer`. An optional `preferredname` can be
provided, otherwise [`defaultname`](@ref) will be used to generate a name. The
name of the object will be returned.

A cache of domains, meshes, topologies and spaces is kept: if one of these
objects has already been written, then the file will not be modified: instead
the name under which the object was first written will be returned. Note that
`Field`s and `FieldVector`s are _not_ cached, and so can be written multiple
times.
"""
function write!(writer::HDF5Writer, obj, name = defaultname(obj))
    get!(writer.cache, name) do
        write_new!(writer, obj, name)
    end
end

# Domains
"""
    defaultname(obj)

Default name of object for InputOutput writers.
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
    write_new!(writer, domain, name)

Writes an object of type 'IntervalDomain' and name 'name' to the HDF5 file.
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

"""
    write_new!(writer, domain, name)

Writes an object of type 'SphereDomain' and name 'name' to the HDF5 file.
"""
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

"""
    write_new!(writer, mesh, name)

Write `CubedSphereMesh` data to HDF5.
"""
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

"""
    write_new!(writer, topology, name)

Write `IntervalTopology` data to HDF5.
"""
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

"""
    write_new!(writer, topology, name)

Write `Topology2D` data to HDF5.
"""
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
defaultname(::Grids.SpectralElementGrid1D) = "horizontal_grid"
defaultname(::Grids.SpectralElementGrid2D) = "horizontal_grid"
defaultname(::Grids.ExtrudedFiniteDifferenceGrid) =
    "extruded_finite_difference_grid"
defaultname(grid::Grids.FiniteDifferenceGrid) = defaultname(grid.topology)
defaultname(grid::Grids.LevelGrid) =
    "$(defaultname(grid.full_grid)): level $(grid.level)"

"""
    write_new!(writer, space, name)

Write `SpectralElementSpace1D` data to HDF5.
"""
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
    write_attribute(group, "topology", write!(writer, Spaces.topology(grid)))
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


"""
    write_new!(writer, domain, name)

Writes an object of type 'Hypsography' and name 'name' to the HDF5 file.
"""
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

# write fields
"""
    write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)

Write the `field` to the HDF5 in `writer` and assign it the given `name`.
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

Write a `Field`, with `axes` of type `PointSpace`,  to the HDF5 file. The field
is written to the `fields` group in the file, with the name `name`. The local
geometry data of the `PointSpace` is written to the `local_geometry_data` group
with name `name`.
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
        string(nameof(typeof(Fields.field_values(field)))),
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
        field::Fields.Field,
        name::AbstractString,
        space::Spaces.AbstractSpace,
    )

Write an object of type 'Field' and name 'name' to the HDF5 file.
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
    topology = Spaces.topology(space)
    grid = Spaces.grid(space)
    grid_name = write!(writer, grid)

    if topology isa Topologies.Topology2D &&
       !(writer.context isa ClimaComms.SingletonCommsContext)
        nelems = Topologies.nelems(topology)
        h_dim = DataLayouts.h_dim(DataLayouts.singleton(values))
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
        string(nameof(typeof(Fields.field_values(field)))),
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

Write one or more `name => value` pairs to `writer`.
"""
function write!(writer::HDF5Writer, pairs::Pair...)
    for (name, value) in pairs
        write!(writer, value, name)
    end
    return nothing
end


"""
    write!(filename::AbstractString, name => value...)

Write one or more `name => value` pairs to the HDF5 file `filename`.
"""
function write!(filename::AbstractString, pairs::Pair...)
    hdfwriter = HDF5Writer(filename)
    try
        write!(hdfwriter, pairs...)
    finally
        Base.close(hdfwriter)
    end
end
