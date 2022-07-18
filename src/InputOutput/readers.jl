abstract type AbstractReader end

using HDF5
using StaticArrays
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
    HDF5Reader

A struct containing the `filename` and `cache` for reading data from the HDF5 file.
The `cache` enables reading of relevant data without duplication.
"""
struct HDF5Reader
    file::HDF5.File
    cache::Dict{Any, Any}
end

function HDF5Reader(filename::AbstractString)
    file = h5open(filename, "r")
    cache = Dict()
    return HDF5Reader(file, cache)
end

"""
    Base.read!(reader, path)

Checks the cache to verify if the object, specified by the `path`, is in the cache.
If it has not been read, this function reads it to the cache and returns the object.
"""
function Base.read!(reader::HDF5Reader, path::AbstractString)
    (; file, cache) = reader
    name = split(path, "/")[end]
    if !haskey(cache, name)
        @assert haskey(file, path)
        read_new!(reader, name, _type(reader, path))
    end
    return cache[name]
end

"""
    _type(reader, path)

Extracts the type of the object, specified by the `path`.
"""
_type(reader::HDF5Reader, path::AbstractString) =
    eval(Meta.parse(attrs(reader.file[path])["type"]))

function _scan_primitive_type_string(typestring::AbstractString)
    @assert typestring ∈ (
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Int128",
        "UInt128",
        "Float16",
        "Float32",
        "Float64",
        "BigFloat",
    )
    typestring == "Int8" && return Int8
    typestring == "UInt8" && return UInt8
    typestring == "Int16" && return Int16
    typestring == "UInt16" && return UInt16
    typestring == "Int32" && return Int32
    typestring == "UInt32" && return UInt32
    typestring == "Int64" && return Int64
    typestring == "UInt64" && return UInt64
    typestring == "Int128" && return Int128
    typestring == "UInt128" && return UInt128
    typestring == "Float64" && return Float64
    typestring == "Float32" && return Float32
    typestring == "Float16" && return Float16
    return BigFloat
end

function _scan_coord_string(coordstring::AbstractString)
    @assert coordstring ∈ ("XPoint", "YPoint", "ZPoint")
    coordstring == "ZPoint" && return ZPoint
    coordstring == "YPoint" && return YPoint
    return XPoint
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
    read_new!(reader, name, ::Type{IntervalDomain})

Reads an object named 'name' of type 'IntervalDomain'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{IntervalDomain},
)
    g = reader.file["domains/" * name]
    coord_type = attrs(g)["coord_type"]
    FT = _scan_primitive_type_string(split(split(coord_type, "{")[end], "}")[1])
    CT = _scan_coord_string(split(split(coord_type, "{")[1], ".")[end])
    coord_min = parse(FT, split(split(attrs(g)["coord_min"], "(")[end], ")")[1])
    coord_max = parse(FT, split(split(attrs(g)["coord_max"], "(")[end], ")")[1])
    boundary_names =
        haskey(attributes(g), "boundary_names") ?
        tuple(map(Symbol, attrs(g)["boundary_names"])...) : nothing
    periodic = boundary_names === nothing ? true : false
    reader.cache[name] = Domains.IntervalDomain(
        CT{FT}(coord_min),
        CT{FT}(coord_max),
        periodic = periodic,
        boundary_names = boundary_names,
    )
    return nothing
end

"""
    read_new!(reader, name, ::Type{SphereDomain})

Reads an object named 'name' of type 'SphereDomain'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{SphereDomain},
)
    g = reader.file["domains/" * name]
    reader.cache[name] = Domains.SphereDomain(attrs(g)["radius"])
    return nothing
end

"""
    read_new!(reader, name, ::Type{IntervalMesh})

Reads an object named 'name' of type 'IntervalMesh'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{IntervalMesh},
)
    (; file, cache) = reader
    g = file["meshes/" * name]
    domain = attrs(g)["domain"]
    FT = _scan_primitive_type_string(attrs(g)["FT"]) # Float type
    nelements = attrs(g)["nelements"]
    faces_type = attrs(g)["faces_type"]
    if faces_type == "Range"
        cache[name] = Meshes.IntervalMesh(
            Base.read!(reader, "domains/" * domain),
            Meshes.Uniform(),
            nelems = nelements,
        )
    else
        CT = _scan_coord_string(
            split(
                replace(
                    attrs(g)["faces_pt_type"],
                    "ClimaCore." => "",
                    "Geometry." => "",
                ),
                "{",
            )[1],
        )
        faces = [CT{FT}(coords) for coords in attrs(g)["faces"]]
        cache[name] =
            Meshes.IntervalMesh(Base.read!(reader, "domains/" * domain), faces)
    end
    return nothing
end

"""
    read_new!(reader, name, ::Type{RectilinearMesh})

Reads an object named 'name' of type 'RectilinearMesh'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{RectilinearMesh},
)
    (; file, cache) = reader
    g = file["meshes/" * name]
    interval1 = attrs(g)["interval1"]
    interval2 = attrs(g)["interval2"]
    cache[name] = Meshes.RectilinearMesh(
        Base.read!(reader, "meshes/$interval1"),
        Base.read!(reader, "meshes/$interval2"),
    )
    return nothing
end

"""
    read_new!(reader, name, ::Type{EquiangularCubedSphere})

Reads an object named 'name' of type 'EquiangularCubedSphere'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{EquiangularCubedSphere},
)
    (; file, cache) = reader
    g = file["meshes/" * name]
    localelementmap =
        occursin("NormalizedBilinearMap", attrs(g)["localelementmap"]) ?
        NormalizedBilinearMap() : IntrinsicMap()
    ne = attrs(g)["ne"]
    domain = attrs(g)["domain"]
    cache[name] = Meshes.EquiangularCubedSphere(
        Base.read!(reader, "domains/" * domain),
        ne,
        localelementmap,
    )
    return nothing
end

"""
    read_new!(reader, name, ::Type{IntervalTopology})

Reads an object named 'name' of type 'IntervalTopology'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{IntervalTopology},
)
    (; file, cache) = reader
    g = file["topologies/" * name]
    mesh = attrs(g)["mesh"]
    cache[name] =
        Topologies.IntervalTopology(Base.read!(reader, "meshes/" * mesh))
    return nothing
end

"""
    read_new!(reader, name, ::Type{Topology2D})

Reads an object named 'name' of type 'Topology2D'.
"""
function read_new!(reader::HDF5Reader, name::AbstractString, ::Type{Topology2D})
    (; file, cache) = reader
    g = file["topologies/" * name]
    elemorder_matrix = HDF5.read(g, "elemorder")
    elemorder = matrix_to_cartesianindices(elemorder_matrix)
    mesh = attrs(g)["mesh"]
    cache[name] =
        Topologies.Topology2D(Base.read!(reader, "meshes/" * mesh), elemorder)
    return nothing
end

"""
    read_new!(reader, name, ::Type{SpectralElementSpace1D})

Reads an object named 'name' of type 'SpectralElementSpace1D'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{SpectralElementSpace1D},
)
    (; file, cache) = reader
    g = file["spaces/" * name]
    npts = attrs(g)["quadrature_num_points"]
    quadrature_style = _scan_quadrature_style(attrs(g)["quadrature_type"], npts)
    topology = attrs(g)["topology"]
    cache[name] = Spaces.SpectralElementSpace1D(
        Base.read!(reader, "topologies/" * topology),
        quadrature_style,
    )
    return nothing
end

"""
    read_new!(reader, name, ::Type{SpectralElementSpace2D})

Reads an object named 'name' of type 'SpectralElementSpace2D'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{SpectralElementSpace2D},
)
    (; file, cache) = reader
    g = file["spaces/" * name]
    npts = attrs(g)["quadrature_num_points"]
    quadrature_style = _scan_quadrature_style(attrs(g)["quadrature_type"], npts)
    topology = attrs(g)["topology"]
    cache[name] = Spaces.SpectralElementSpace2D(
        Base.read!(reader, "topologies/" * topology),
        quadrature_style,
    )
    return nothing
end

"""
    read_new!(reader, name, ::Type{ExtrudedFiniteDifferenceSpace})

Reads an object named 'name' of type 'ExtrudedFiniteDifferenceSpace'.
"""
function read_new!(
    reader::HDF5Reader,
    name::AbstractString,
    ::Type{T},
) where {T <: ExtrudedFiniteDifferenceSpace}
    (; file, cache) = reader
    g = file["spaces/" * name]
    staggering = attrs(g)["staggering"] == "CellCenter" ? CellCenter : CellFace
    vertical_topology = attrs(g)["vertical_topology"]
    horizontal_space_name = attrs(g)["horizontal_space"]
    vertical_space = Spaces.FiniteDifferenceSpace{staggering}(
        Base.read!(reader, "topologies/" * vertical_topology),
    )
    horizontal_space = Base.read!(reader, "spaces/" * horizontal_space_name)
    cache[name] =
        Spaces.ExtrudedFiniteDifferenceSpace(horizontal_space, vertical_space)
    return nothing
end

"""
    read_new!(reader, name, ::Type{Field})

Reads an object named 'name' of type 'Field'.
"""
function read_new!(reader::HDF5Reader, name::AbstractString, ::Type{Field})
    (; file, cache) = reader
    @assert name ∈ keys(file["fields"])
    g = file["fields/" * name]
    data = read_dataset(g, "data")
    data_layout = attrs(g)["data_layout"]
    Nij = size(data, findfirst("I", data_layout)[1])
    DataLayout = _scan_data_layout(data_layout)
    ElType =
        eval(Meta.parse(replace(attrs(g)["value_type"], "ClimaCore." => "")))
    values = DataLayout{ElType, Nij}(data)
    spacename = attrs(g)["space"]
    return Fields.Field(values, Base.read!(reader, "spaces/" * spacename))
end

"""
    read_fieldvector!(reader, name)

Reads a 'FieldVector' with name 'name` from the HDFReader.
"""
function read_fieldvector!(reader::HDF5Reader, name::AbstractString)
    (; file, cache) = reader
    @assert name ∈ keys(file["fieldvectors"])
    g = file["fieldvectors/" * name]
    fields = attrs(g)["fields"]
    return FieldVector(;
        [
            Symbol(name) =>
                read_new!(reader, name, _type(reader, "fields/" * name)) for
            name in fields
        ]...,
    )
end

"""
    read!(filename, fvname)

Reads a 'FieldVector' with name 'fvname` from the HDF5 file 'filename'.
"""
function read(filename::AbstractString, fvname::AbstractString)
    hdfreader = InputOutput.HDF5Reader(filename)
    fv = InputOutput.read_fieldvector!(hdfreader, fvname)
    Base.close(hdfreader)
    return fv
end

function Base.close(hdfreader::HDF5Reader)
    close(hdfreader.file)
    empty!(hdfreader.cache)
    return nothing
end
