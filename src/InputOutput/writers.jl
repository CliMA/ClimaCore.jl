abstract type AbstractWriter end

"""
    HDF5Writer(filename::AbstractString[, context::ClimaComms.AbstractCommsContext])

An `AbstractWriter` for writing to HDF5-formatted files using the ClimaCore
storage conventions. An internal cache is used to avoid writing duplicate
domains, meshes, topologies and spaces to the file. Use [`HDF5Reader`](@ref) to
load the data from the file.

The optional `context` can be used for writing distributed fields: in this case,
the `MPICommsContext` used passed as an argument: this must match the context
used for distributing the `Field`.

!!! note

    The default Julia HDF5 binaries are not built with MPI support. To use the
    distributed functionality, you will need to configure HDF5.jl with an
    MPI-enabled HDF5 library, see [the HDF5.jl
    documentation](https://juliaio.github.io/HDF5.jl/stable/#Parallel-HDF5).

# Interface

[`write!`](@ref)

# Usage

```julia
writer = InputOutput.HDF5Writer(filename)
InputOutput.write!(writer, Y, "Y")
close(writer)
```
"""
struct HDF5Writer{C <: ClimaComms.AbstractCommsContext} <: AbstractWriter
    file::HDF5.File
    context::C
    cache::Dict{String, String}
end

function HDF5Writer(
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext = ClimaComms.SingletonCommsContext(),
)
    if context isa ClimaComms.SingletonCommsContext
        file = h5open(filename, "w")
    else
        file = h5open(filename, "w", context.mpicomm)
    end
    write_attribute(file, "ClimaCore version", string(VERSION))
    cache = Dict{String, String}()
    return HDF5Writer(file, context, cache)
end

function Base.close(hdfwriter::HDF5Writer)
    empty!(hdfwriter.cache)
    close(hdfwriter.file)
    return nothing
end


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
    write_attribute(group, "ne", mesh.ne)
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
defaultname(::Topologies.DistributedTopology2D) = "2d"
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
    @assert writer.context isa ClimaComms.SingletonCommsContext

    group = create_group(writer.file, "topologies/$name")
    write_attribute(group, "type", "Topology2D")
    write_attribute(group, "mesh", write!(writer, topology.mesh))
    if !(topology.elemorder isa CartesianIndices)
        elemorder_matrix = reinterpret(reshape, Int, topology.elemorder)
        write_dataset(group, "elemorder", elemorder_matrix)
    end
    return name
end

"""
    write_new!(writer, topology, name)

Write `DistributedTopology2D` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    topology::Topologies.DistributedTopology2D,
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

# Spaces
#
defaultname(::Spaces.SpectralElementSpace1D) = "horizontal_space"
defaultname(::Spaces.SpectralElementSpace2D) = "horizontal_space"
defaultname(::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    "center_extruded_finite_difference_space"
defaultname(::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    "face_extruded_finite_difference_space"
defaultname(space::Spaces.FiniteDifferenceSpace) = defaultname(space.topology)


"""
    write_new!(writer, space, name)

Write `SpectralElementSpace1D` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    space::Spaces.SpectralElementSpace1D,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, "spaces/$name")
    write_attribute(group, "type", "SpectralElementSpace1D")
    write_attribute(
        group,
        "quadrature_type",
        string(nameof(typeof(space.quadrature_style))),
    )
    write_attribute(
        group,
        "quadrature_num_points",
        Spaces.Quadratures.degrees_of_freedom(space.quadrature_style),
    )
    write_attribute(group, "topology", write!(writer, space.topology))
    return name
end

function write_new!(
    writer::HDF5Writer,
    space::Spaces.SpectralElementSpace2D,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, "spaces/$name")
    write_attribute(group, "type", "SpectralElementSpace2D")
    write_attribute(
        group,
        "quadrature_type",
        string(nameof(typeof(space.quadrature_style))),
    )
    write_attribute(
        group,
        "quadrature_num_points",
        Spaces.Quadratures.degrees_of_freedom(space.quadrature_style),
    )
    write_attribute(group, "topology", write!(writer, space.topology))
    return name
end

function write_new!(
    writer::HDF5Writer,
    space::Spaces.FiniteDifferenceSpace,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, "spaces/$name")
    write_attribute(group, "type", "FiniteDifferenceSpace")
    write_attribute(group, "topology", write!(writer, space.topology))
    return group
end

function write_new!(
    writer::HDF5Writer,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, "spaces/$name")
    write_attribute(group, "type", "ExtrudedFiniteDifferenceSpace")
    write_attribute(group, "staggering", "CellCenter")
    write_attribute(
        group,
        "horizontal_space",
        write!(writer, space.horizontal_space),
    )
    write_attribute(
        group,
        "vertical_topology",
        write!(writer, space.vertical_topology),
    )
    if space.hypsography isa Hypsography.LinearAdaption
        write_attribute(group, "hypsography_type", "LinearAdaption")
        write_attribute(
            group,
            "hypsography_surface",
            write!(writer, space.hypsography.surface, "_z_surface/$name"),
        )
    end
    return name
end

function write_new!(
    writer::HDF5Writer,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, name)
    group = create_group(writer.file, "spaces/$name")
    write_attribute(group, "type", "ExtrudedFiniteDifferenceSpace")
    write_attribute(group, "staggering", "CellFace")
    write_attribute(
        group,
        "center_space",
        write!(writer, Spaces.CenterExtrudedFiniteDifferenceSpace(space)),
    )
    return name
end


# write fields
function write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)
    space = axes(field)
    space_name = write!(writer, space)

    array = parent(field)
    topology = Spaces.topology(space)
    nd = ndims(array)
    if topology isa Topologies.DistributedTopology2D &&
       !(writer.context isa ClimaComms.SingletonCommsContext)
        nelems = Topologies.nelems(topology)
        dims = ntuple(d -> d == nd ? nelems : size(array, d), nd)
        localidx = ntuple(d -> d < nd ? (:) : topology.local_elem_gidx, nd)
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
    write_attribute(dataset, "value_type", string(eltype(field)))
    write_attribute(dataset, "space", space_name)
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
    write!(filename, fvpair)

Write 'FieldVector' data, specified by a pair `fieldvectorname => fieldvector`, to HDF5 file 'filename'.
"""
function write!(
    filename::AbstractString,
    fvpair::Pair{String, <:Fields.FieldVector},
)
    hdfwriter = InputOutput.HDF5Writer(filename)
    InputOutput.write!(hdfwriter, fvpair.second, fvpair.first)
    Base.close(hdfwriter)
    return nothing
end
