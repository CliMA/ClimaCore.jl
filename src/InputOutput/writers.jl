abstract type AbstractWriter end


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
    cache = Dict{String, String}()
    return HDF5Writer(file, cache)
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
defaultname(::Domains.SphereDomain) = "domains/sphere"
defaultname(::Domains.IntervalDomain) = "domains/interval"

"""
    write_new!(writer, domain, name)

Writes an object of type 'IntervalDomain' and name 'name' to the HDF5 file.
"""
function write_new!(
    writer::HDF5Writer,
    domain::Domains.IntervalDomain,
    name::AbstractString = defaultname(domain),
)
    group = create_group(writer.file, name)
    write_attribute(group, "type", "IntervalDomain")
    write_attribute(group, "coord_type", string(typeof(domain.coord_min)))
    write_attribute(group, "coord_min", string(domain.coord_min))
    write_attribute(group, "coord_max", string(domain.coord_max))
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
    group = create_group(writer.file, name)
    write_attribute(group, "type", "SphereDomain")
    write_attribute(group, "radius", domain.radius)
    return name
end

# Meshes
defaultname(::Meshes.IntervalMesh) = "meshes/intervalmesh"
defaultname(::Meshes.RectilinearMesh) = "meshes/mesh2d"
defaultname(::Meshes.AbstractCubedSphere) = "meshes/mesh2d"

"""
    write_new!(writer, mesh, name)

Write `IntervalMesh` data to HDF5.

"""
function write_new!(
    writer::HDF5Writer,
    mesh::Meshes.IntervalMesh,
    name::AbstractString = defaultname(mesh),
)
    meshname = split(name, "/")[end]
    group = create_group(writer.file, name)
    write_attribute(group, "type", "IntervalMesh")
    write!(writer, mesh.domain, "domains/$meshname")
    write_attribute(group, "nelements", length(mesh.faces) - 1)
    FT = Geometry.float_type(mesh.faces[1])
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
    write_attribute(group, "faces_pt_type", string(typeof(mesh.faces[1])))
    write_attribute(group, "FT", string(FT))
    write_attribute(group, "domain", meshname)
    return name
end

"""
    write_new!(writer, mesh, name)

Write `RectilinearMesh` data to HDF5.

"""
function write_new!(
    writer::HDF5Writer,
    mesh::Meshes.RectilinearMesh,
    name::AbstractString = defaultname(mesh),
)
    group = create_group(writer.file, name)
    write_attribute(group, "type", "RectilinearMesh")
    write_attribute(group, "interval1", "interval1")
    write_attribute(group, "interval2", "interval2")
    write!(writer, mesh.intervalmesh1, "meshes/interval1")
    write!(writer, mesh.intervalmesh2, "meshes/interval2")
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
    group = create_group(writer.file, name)
    write!(writer, mesh.domain)
    write_attribute(group, "type", string(nameof(typeof(mesh))))
    write_attribute(group, "ne", mesh.ne)
    write_attribute(
        group,
        "localelementmap",
        string(nameof(typeof(mesh.localelementmap))),
    )
    write_attribute(group, "domain", "sphere")
    return name
end

# Topologies
defaultname(::Topologies.Topology2D) = "topologies/topology2d"
defaultname(::Topologies.IntervalTopology) = "topologies/intervaltopology"

"""
    write_new!(writer, topology, name)

Write `IntervalTopology` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    topology::Topologies.IntervalTopology,
    name::AbstractString = defaultname(topology),
)
    group = create_group(writer.file, name)
    write_attribute(group, "type", "IntervalTopology")
    if occursin("face", name)
        meshname = "meshes/face_interval_mesh"
    elseif occursin("center", name)
        meshname = "meshes/center_interval_mesh"
    else
        meshname = "meshes/IntervalMesh"
    end
    write_attribute(group, "mesh", split(meshname, "/")[end])
    write!(writer, topology.mesh, meshname)
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
    group = create_group(writer.file, name)
    write_attribute(group, "type", "Topology2D")
    write_attribute(group, "mesh", "mesh2d")
    write!(writer, topology.mesh, "meshes/mesh2d")
    if !(topology.elemorder isa LinearIndices)
        elemorder_matrix = cartesianindices_to_matrix(topology.elemorder)
        write_dataset(group, "elemorder", elemorder_matrix)
    end
    return name
end
# Spaces
#
defaultname(::Spaces.SpectralElementSpace1D) = "spaces/horizontal_space"
defaultname(::Spaces.SpectralElementSpace2D) = "spaces/horizontal_space"
defaultname(::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    "spaces/center_extruded_finite_difference_space"
defaultname(::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    "spaces/face_extruded_finite_difference_space"

"""
    write_new!(writer, space, name)

Write `SpectralElementSpace1D` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    space::Spaces.SpectralElementSpace1D,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, name)
    write!(writer, space.topology)
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
    write_attribute(group, "topology", "intervaltopology")
    return name
end

"""
    write_new!(writer, space, name)

Write `SpectralElementSpace2D` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    space::Spaces.SpectralElementSpace2D,
    name::AbstractString = defaultname(space),
)
    group = create_group(writer.file, name)
    write!(writer, space.topology)
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
    write_attribute(group, "topology", "topology2d")
    return name
end

"""
    write_new!(writer, space, name)

Write `FiniteDifferenceSpace` data to HDF5.
"""
function write_new!(
    writer::HDF5Writer,
    space::Spaces.FiniteDifferenceSpace,
    name::AbstractString = "vertical_space",
)
    group = create_group(writer.file, "spaces/$name")
    write(writer, space.topology)
    write_attribute(group, "type", "FiniteDifferenceSpace")
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
    group = create_group(writer.file, name)
    write!(writer, space.horizontal_space, "spaces/horizontal_space")
    write!(
        writer,
        space.vertical_topology,
        "topologies/center_vertical_topology",
    )
    write_attribute(group, "type", "CenterExtrudedFiniteDifferenceSpace")
    write_attribute(group, "vertical_topology", "center_vertical_topology")
    write_attribute(group, "horizontal_space", "horizontal_space")
    write_attribute(group, "staggering", "CellCenter")
    writer.cache[name] = name
    return name#group
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
    write!(writer, space.horizontal_space, "spaces/horizontal_space")
    write!(writer, space.vertical_topology, "topologies/face_vertical_topology")
    write_attribute(group, "type", "FaceExtrudedFiniteDifferenceSpace")
    write_attribute(group, "vertical_topology", "face_vertical_topology")
    write_attribute(group, "horizontal_space", "horizontal_space")
    write_attribute(group, "staggering", "CellFace")
    writer.cache[name] = name
    return name
end


# write fields
"""
    write!(writer, field, name)

Write `Field` data to HDF5.
"""
function write!(writer::HDF5Writer, field::Fields.Field, name::AbstractString)
    group = create_group(writer.file, "fields/$name")
    dataset = write_dataset(writer.file, "fields/$name/data", parent(field))
    write!(writer, axes(field))
    write_attribute(group, "type", "Field")
    write_attribute(
        group,
        "data_layout",
        string(nameof(typeof(Fields.field_values(field)))),
    )
    write_attribute(group, "value_type", string(eltype(field)))
    write_attribute(group, "space", split(defaultname(axes(field)), "/")[end])
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
    group = create_group(writer.file, "fieldvectors/$name")
    write_attribute(group, "type", "FieldVector")
    fields = String[]
    for (key, component) in pairs(Fields._values(fieldvector))
        write!(writer, component, string(key))
        push!(fields, string(key))
    end
    write_attribute(group, "fields", fields)
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

function Base.close(hdfwriter::HDF5Writer)
    close(hdfwriter.file)
    empty!(hdfwriter.cache)
    return nothing
end
