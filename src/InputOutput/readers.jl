abstract type AbstractReader end

using HDF5
using StaticArrays
using ..ClimaCore
using ..Domains: IntervalDomain, SphereDomain
using ..Meshes:
    Meshes,
    NormalizedBilinearMap,
    IntervalMesh,
    RectilinearMesh,
    EquiangularCubedSphere
using ..Topologies: Topologies, IntervalTopology, Topology2D
using ..Spaces:
    Spaces,
    Spaces.Quadratures,
    Spaces.Quadratures.GLL,
    Spaces.CellCenter,
    Spaces.CellFace,
    SpectralElementSpace1D,
    SpectralElementSpace2D,
    CenterExtrudedFiniteDifferenceSpace,
    FaceExtrudedFiniteDifferenceSpace,
    ExtrudedFiniteDifferenceSpace
using ..Fields: Field, FieldVector
import ..Geometry:
    Geometry,
    XPoint,
    YPoint,
    ZPoint,
    Covariant1Vector,
    Covariant12Vector,
    Covariant3Vector
using ..DataLayouts

"""
    HDF5Reader(filename)

An `AbstractReader` for reading from HDF5 files created by [`HDF5Writer`](@ref).
The reader object contains an internal cache of domains, meshes, topologies and spaces
that are read so that duplicate objects are not created.

# Interface
- [`read_domain`](@ref)
- [`read_mesh`](@ref)
- [`read_topology`](@ref)
- [`read_space`](@ref)
- [`read_field`](@ref)

# Usage

```julia
reader = InputOutput.HDF5Reader(filename)
field = read_field(reader, "Y")
close(reader)
```
"""
struct HDF5Reader
    file::HDF5.File
    domain_cache::Dict{Any, Any}
    mesh_cache::Dict{Any, Any}
    topology_cache::Dict{Any, Any}
    space_cache::Dict{Any, Any}
end

function HDF5Reader(filename::AbstractString)
    file = h5open(filename, "r")
    if !haskey(attrs(file), "ClimaCore version")
        error("Not a ClimaCore HDF5 file")
    end
    file_version = VersionNumber(attrs(file)["ClimaCore version"])
    package_version = PkgVersion.@Version
    if file_version > package_version
        @warn "$filename was written using a newer version of ClimaCore than is currently loaded" file_version package_version
    end
    return HDF5Reader(file, Dict(), Dict(), Dict(), Dict())
end


function Base.close(hdfreader::HDF5Reader)
    close(hdfreader.file)
    empty!(hdfreader.domain_cache)
    empty!(hdfreader.mesh_cache)
    empty!(hdfreader.topology_cache)
    empty!(hdfreader.space_cache)
    return nothing
end

function _scan_coord_type(coordstring::AbstractString)
    coordstring == "XPoint" && return XPoint
    coordstring == "YPoint" && return YPoint
    coordstring == "ZPoint" && return ZPoint
    error("Invalid coord type $coordstring")
end

function _scan_quadrature_style(quadraturestring::AbstractString, npts)
    @assert quadraturestring ∈ ("GLL", "GL", "Uniform", "ClosedUniform")
    quadraturestring == "GLL" && return GLL{npts}()
    quadraturestring == "GL" && return GL{npts}()
    quadraturestring == "Uniform" && return Uniform{npts}()
    return ClosedUniform{npts}()
end

function _scan_data_layout(layoutstring::AbstractString)
    @assert layoutstring ∈ ("IJFH", "IJF", "IFH", "IF", "VIJFH", "VIFH")
    layoutstring == "IJFH" && return IJFH
    layoutstring == "IJF" && return IJF
    layoutstring == "IFH" && return IFH
    layoutstring == "IF" && return IF
    layoutstring == "VIJFH" && return VIJFH
    return VIFH
end

"""
    matrix_to_cartesianindices(elemorder_matrix)

Converts the `elemorder_matrix` to cartesian indices.
"""
function matrix_to_cartesianindices(elemorder_matrix)
    m, ndims = size(elemorder_matrix)
    dims = [maximum(elemorder_matrix[:, dim]) for dim in 1:ndims]
    elemorder = Vector{CartesianIndex{ndims}}(undef, m)
    for i in 1:m
        elemorder[i] = CartesianIndex(elemorder_matrix[i, :]...)
    end
    return reshape(elemorder, dims...)
end


"""
    read_domain(reader::AbstractReader, name)

Reads a domain named `name` from `reader`. Domain objects are cached in the
reader to avoid creating duplicate objects.
"""
function read_domain(reader, name)
    Base.get!(reader.domain_cache, name) do
        read_domain_new(reader, name)
    end
end

function read_domain_new(reader::HDF5Reader, name::AbstractString)
    group = reader.file["domains/$name"]
    type = attrs(group)["type"]
    if type == "IntervalDomain"
        CT = _scan_coord_type(attrs(group)["coord_type"])
        coord_min = CT(attrs(group)["coord_min"])
        coord_max = CT(attrs(group)["coord_max"])
        if haskey(attributes(group), "boundary_names")
            boundary_names =
                tuple(map(Symbol, attrs(group)["boundary_names"])...)
            return Domains.IntervalDomain(coord_min, coord_max; boundary_names)
        else
            return Domains.IntervalDomain(coord_min, coord_max; periodic = true)
        end
    elseif type == "SphereDomain"
        radius = attrs(group)["radius"]
        return Domains.SphereDomain(radius)
    else
        error("Unsupported domain type $type")
    end
end


"""
    read_mesh(reader::AbstractReader, name)

Reads a mesh named `name` from `reader`, or from the reader cache if it has
already been read.
"""
function read_mesh(reader, name)
    Base.get!(reader.mesh_cache, name) do
        read_mesh_new(reader, name)
    end
end

function read_mesh_new(reader::HDF5Reader, name::AbstractString)
    group = reader.file["meshes/$name"]
    type = attrs(group)["type"]
    if type == "IntervalMesh"
        domain = read_domain(reader, attrs(group)["domain"])
        nelements = attrs(group)["nelements"]
        faces_type = attrs(group)["faces_type"]
        if faces_type == "Range"
            return Meshes.IntervalMesh(
                domain,
                Meshes.Uniform(),
                nelems = nelements,
            )
        else
            CT = Domains.coordinate_type(domain)
            faces = [CT(coords) for coords in attrs(group)["faces"]]
            return Meshes.IntervalMesh(domain, faces)
        end
    elseif type == "RectilinearMesh"
        intervalmesh1 = read_mesh(reader, attrs(group)["intervalmesh1"])
        intervalmesh2 = read_mesh(reader, attrs(group)["intervalmesh2"])
        return Meshes.RectilinearMesh(intervalmesh1, intervalmesh2)
    elseif type == "EquiangularCubedSphere"
        domain = read_domain(reader, attrs(group)["domain"])
        localelementmap =
            attrs(group)["localelementmap"] == "NormalizedBilinearMap" ?
            NormalizedBilinearMap() : IntrinsicMap()
        ne = attrs(group)["ne"]
        return Meshes.EquiangularCubedSphere(domain, ne, localelementmap)
    end
end

"""
    read_topology(reader::AbstractReader, name)

Reads a topology named `name` from `reader`, or from the reader cache if it has
already been read.
"""
function read_topology(reader, name)
    Base.get!(reader.topology_cache, name) do
        read_topology_new(reader, name)
    end
end


function read_topology_new(reader::HDF5Reader, name::AbstractString)
    group = reader.file["topologies/$name"]
    type = attrs(group)["type"]
    if type == "IntervalTopology"
        mesh = read_mesh(reader, attrs(group)["mesh"])
        return Topologies.IntervalTopology(mesh)
    elseif type == "Topology2D"
        mesh = read_mesh(reader, attrs(group)["mesh"])
        if haskey(group, "elemorder")
            elemorder =
                matrix_to_cartesianindices(HDF5.read(group, "elemorder"))
            return Topologies.Topology2D(mesh, elemorder)
        else
            return Topologies.Topology2D(mesh)
        end
    else
        error("Unsupported type $type")
    end
end


"""
    read_space(reader::AbstractReader, name)

Reads a space named `name` from `reader`, or from the reader cache if it has
already been read.
"""
function read_space(reader, name)
    Base.get!(reader.space_cache, name) do
        read_space_new(reader, name)
    end
end

function read_space_new(reader, name)
    group = reader.file["spaces/$name"]
    type = attrs(group)["type"]
    if type in ("SpectralElementSpace1D", "SpectralElementSpace2D")
        npts = attrs(group)["quadrature_num_points"]
        quadrature_style =
            _scan_quadrature_style(attrs(group)["quadrature_type"], npts)
        topology = read_topology(reader, attrs(group)["topology"])
        if type == "SpectralElementSpace1D"
            Spaces.SpectralElementSpace1D(topology, quadrature_style)
        else
            Spaces.SpectralElementSpace2D(topology, quadrature_style)
        end
    elseif type == "ExtrudedFiniteDifferenceSpace"
        if attrs(group)["staggering"] == "CellFace"
            center_space = read_space(reader, attrs(group)["center_space"])
            return Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
        else
            vertical_topology =
                read_topology(reader, attrs(group)["vertical_topology"])
            horizontal_space =
                read_space(reader, attrs(group)["horizontal_space"])
            return Spaces.ExtrudedFiniteDifferenceSpace(
                horizontal_space,
                Spaces.CenterFiniteDifferenceSpace(vertical_topology),
            )
        end
    end
end

"""
    read_field(reader, name)

Reads a `Field` or `FieldVector` named `name` from `reader`. Fields are _not_
cached, so that reading the same field multiple times will create multiple
distinct objects.
"""
function read_field(reader::HDF5Reader, name::AbstractString)
    obj = reader.file["fields/$name"]
    type = attrs(obj)["type"]
    if type == "Field"
        data = read(obj)
        data_layout = attrs(obj)["data_layout"]
        Nij = size(data, findfirst("I", data_layout)[1])
        DataLayout = _scan_data_layout(data_layout)
        ElType = eval(Meta.parse(attrs(obj)["value_type"]))
        values = DataLayout{ElType, Nij}(data)
        space = read_space(reader, attrs(obj)["space"])
        return Fields.Field(values, space)
    elseif type == "FieldVector"
        FieldVector(;
            [
                Symbol(sub) => read_field(reader, "$name/$sub") for
                sub in keys(obj)
            ]...,
        )
    else
        error("Unsupported type $type")
    end
end
