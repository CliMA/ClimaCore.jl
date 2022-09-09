abstract type AbstractReader end

# these need to be here for to make the eval work
# TODO: figure out a better way to represent types
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
    HDF5Reader(filename::AbstractString[, context::ClimaComms.AbstractCommsContext])

An `AbstractReader` for reading from HDF5 files created by [`HDF5Writer`](@ref).
The reader object contains an internal cache of domains, meshes, topologies and
spaces that are read so that duplicate objects are not created.

The optional `context` can be used for reading distributed fields: in this case,
the `MPICommsContext` used passed as an argument: resulting `Field`s will be
distributed using this context. As with [`HDF5Writer`](@ref), this requires a
HDF5 library with MPI support.

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
struct HDF5Reader{C <: ClimaComms.AbstractCommsContext}
    file::HDF5.File
    context::C
    file_version::VersionNumber
    domain_cache::Dict{Any, Any}
    mesh_cache::Dict{Any, Any}
    topology_cache::Dict{Any, Any}
    space_cache::Dict{Any, Any}
end

function HDF5Reader(
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext = ClimaComms.SingletonCommsContext(),
)
    if context isa ClimaComms.SingletonCommsContext
        file = h5open(filename, "r")
    else
        file = h5open(filename, "r", context.mpicomm)
    end
    if !haskey(attrs(file), "ClimaCore version")
        error("Not a ClimaCore HDF5 file")
    end
    file_version = VersionNumber(attrs(file)["ClimaCore version"])
    if file_version > VERSION
        @warn "$filename was written using a newer version of ClimaCore than is currently loaded" file_version package_version
    end
    return HDF5Reader(
        file,
        context,
        file_version,
        Dict(),
        Dict(),
        Dict(),
        Dict(),
    )
end


function Base.close(hdfreader::HDF5Reader)
    empty!(hdfreader.domain_cache)
    empty!(hdfreader.mesh_cache)
    empty!(hdfreader.topology_cache)
    empty!(hdfreader.space_cache)
    close(hdfreader.file)
    return nothing
end

function _scan_coord_type(coordstring::AbstractString)
    coordstring == "XPoint" && return Geometry.XPoint
    coordstring == "YPoint" && return Geometry.YPoint
    coordstring == "ZPoint" && return Geometry.ZPoint
    error("Invalid coord type $coordstring")
end

function _scan_quadrature_style(quadraturestring::AbstractString, npts)
    @assert quadraturestring ∈ ("GLL", "GL", "Uniform", "ClosedUniform")
    quadraturestring == "GLL" && return Spaces.Quadratures.GLL{npts}()
    quadraturestring == "GL" && return Spaces.Quadratures.GL{npts}()
    quadraturestring == "Uniform" && return Spaces.Quadratures.Uniform{npts}()
    return Spaces.Quadratures.ClosedUniform{npts}()
end

function _scan_data_layout(layoutstring::AbstractString)
    @assert layoutstring ∈ ("IJFH", "IJF", "IFH", "IF", "VIJFH", "VIFH")
    layoutstring == "IJFH" && return DataLayouts.IJFH
    layoutstring == "IJF" && return DataLayouts.IJF
    layoutstring == "IFH" && return DataLayouts.IFH
    layoutstring == "IF" && return DataLayouts.IF
    layoutstring == "VIJFH" && return DataLayouts.VIJFH
    return DataLayouts.VIFH
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
            Meshes.NormalizedBilinearMap() : Meshes.IntrinsicMap()
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
            elemorder_matrix = HDF5.read(group, "elemorder")
            if reader.file_version < v"0.10.9"
                elemorder = collect(
                    reinterpret(
                        reshape,
                        CartesianIndex{size(elemorder_matrix, 2)},
                        elemorder_matrix',
                    ),
                )
            else
                elemorder = collect(
                    reinterpret(
                        reshape,
                        CartesianIndex{size(elemorder_matrix, 1)},
                        elemorder_matrix,
                    ),
                )
            end
        else
            elemorder = Meshes.elements(mesh)
        end

        if reader.context isa ClimaComms.SingletonCommsContext
            return Topologies.Topology2D(mesh, elemorder)
        else
            return Topologies.DistributedTopology2D(
                reader.context,
                mesh,
                elemorder,
            )
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
            hypsography_type = get(attrs(group), "hypsography_type", "Flat")
            if hypsography_type == "Flat"
                hypsography = Spaces.Flat()
            elseif hypsography_type == "LinearAdaption"
                hypsography = Hypsography.LinearAdaption(
                    read_field(reader, attrs(group)["hypsography_surface"]),
                )
            else
                error("Unsupported hypsography type $hypsography_type")
            end
            return Spaces.ExtrudedFiniteDifferenceSpace(
                horizontal_space,
                Spaces.CenterFiniteDifferenceSpace(vertical_topology),
                hypsography,
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
        space = read_space(reader, attrs(obj)["space"])
        topology = Spaces.topology(space)
        if topology isa Topologies.DistributedTopology2D
            nd = ndims(obj)
            localidx = ntuple(d -> d < nd ? (:) : topology.local_elem_gidx, nd)
            data = obj[localidx...]
        else
            data = read(obj)
        end
        data_layout = attrs(obj)["data_layout"]
        Nij = size(data, findfirst("I", data_layout)[1])
        DataLayout = _scan_data_layout(data_layout)
        ElType = eval(Meta.parse(attrs(obj)["value_type"]))
        values = DataLayout{ElType, Nij}(data)
        return Fields.Field(values, space)
    elseif type == "FieldVector"
        Fields.FieldVector(;
            [
                Symbol(sub) => read_field(reader, "$name/$sub") for
                sub in keys(obj)
            ]...,
        )
    else
        error("Unsupported type $type")
    end
end
