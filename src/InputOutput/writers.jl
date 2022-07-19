abstract type AbstractWriter end

using PkgVersion
using HDF5
using ..Domains
using ..Meshes
using ..Topologies
using ..Spaces
using ..Fields

"""
    HDF5Writer

A struct containing the `filename` and `cache` for writing data to the HDF5 file.
The `cache` enables writing of relevant data without duplication.
"""
struct HDF5Writer <: AbstractWriter
    file::HDF5.File
    cache::Dict{String, String}
end

function HDF5Writer(filename::String)
    file = h5open(filename, "w")
    attrs(file)["ClimaCore version"] = string(PkgVersion.@Version)
    cache = Dict{String, String}()
    return HDF5Writer(file, cache)
end


function Base.close(hdfwriter::HDF5Writer)
    close(hdfwriter.file)
    empty!(hdfwriter.cache)
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
    write!(reader, obj[, path])

Checks the cache to verify if the object, specified by the `path`, is in the cache.
If it has not been read, this function reads it to the cache and returns the name of the object.
"""
write!(writer::HDF5Writer, obj, name = defaultname(obj)) =
    get!(writer.cache, name) do
        write_new!(writer, obj, name)
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
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", "IntervalMesh")
    write_attribute(group, "domain", write!(writer, mesh.domain))
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
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", "RectilinearMesh")
    write_attribute(group, "intervalmesh1", write!(writer, mesh.intervalmesh1))
    write_attribute(group, "intervalmesh2", write!(writer, mesh.intervalmesh2))
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
    group = create_group(writer.file, "meshes/$name")
    write_attribute(group, "type", string(nameof(typeof(mesh))))
    write_attribute(group, "ne", mesh.ne)
    write_attribute(
        group,
        "localelementmap",
        string(nameof(typeof(mesh.localelementmap))),
    )
    write_attribute(group, "domain", write!(writer, mesh.domain))
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
    group = create_group(writer.file, "topologies/$name")
    write_attribute(group, "type", "IntervalTopology")
    write_attribute(group, "mesh", write!(writer, topology.mesh))
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
    group = create_group(writer.file, "topologies/$name")
    write_attribute(group, "type", "Topology2D")
    write_attribute(group, "mesh", write!(writer, topology.mesh))
    if !(topology.elemorder isa LinearIndices)
        elemorder_matrix = cartesianindices_to_matrix(topology.elemorder)
        write_dataset(group, "elemorder", elemorder_matrix)
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

"""
    write_new!(writer, space, name)

Write `CenterExtrudedFiniteDifferenceSpace` data to HDF5.
"""
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
    return name
end

"""
    write_new!(writer, space, name)

Write `FaceExtrudedFiniteDifferenceSpace` data to HDF5.
"""
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
"""
    write!(writer, field, name)

Write `Field` data to HDF5.
"""
function write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)
    axes_name = write!(writer, axes(field))
    write_dataset(writer.file, "fields/$name", parent(field))
    dataset = writer.file["fields/$name"]
    write_attribute(dataset, "type", "Field")
    write_attribute(
        dataset,
        "data_layout",
        string(nameof(typeof(Fields.field_values(field)))),
    )
    write_attribute(dataset, "value_type", string(eltype(field)))
    write_attribute(dataset, "space", axes_name)
    return name
end
# field vectors
"""
    write_new!(writer, fieldvector, name)

Write `FieldVector` data to HDF5.
"""
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
