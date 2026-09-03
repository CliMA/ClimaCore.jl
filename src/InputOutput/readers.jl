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
using ..Topologies: Topologies, IntervalTopology
using ..Spaces:
    Spaces,
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

function get_key(file, key)
    if haskey(file, key)
        return file[key]
    else
        msg = "Key `$key` not found in HDF5Reader.\n"
        msg *= "Available keys:\n"
        msg *= string(keys(file))
        error(msg)
    end
end

"""
    read_type(ts::AbstractString)

Parse a string `ts` into an expression, and then evaluate it into a type if it is a valid type expression.
"""
function read_type(ts::AbstractString)
    type_expr = Meta.parse(ts)
    if is_type_expr(type_expr)
        return eval(type_expr)
    end
    error("$ts cannot be parsed into a valid type")
end

"""
    is_type_expr(expr)

Return whether `expr` is a type expression, containing no function calls or other
non-type expressions, with the exception of `@NamedTuple`. Based on `is_valid_type_exp`
in [JLD.jl](https://github.com/JuliaIO/JLD.jl/blob/80ac89643e3ad87545e48f4d361a00a29cdf4e2f/src/JLD.jl#L922).
"""
is_type_expr(s::Symbol) = true
is_type_expr(q::QuoteNode) = is_type_expr(q.value)
function is_type_expr(expr::Expr)
    if expr.head == :macrocall && expr.args[1] == Symbol("@NamedTuple")
        # skip the LineNumberNode, as eval does nothing for it
        if length(expr.args) > 2
            return all(map(is_type_expr, expr.args[3:end]))
        end
        return true
    end
    return expr.head in (:curly, :., :tuple, :braces, Symbol("::")) &&
           all(map(is_type_expr, expr.args))
end
is_type_expr(t) = isbits(t)

"""
    HDF5Reader(filename::AbstractString, context::ClimaComms.AbstractCommsContext)
    HDF5Reader(::Function, filename::AbstractString, context::ClimaComms.AbstractCommsContext)

An `AbstractReader` for reading from HDF5 files created by [`HDF5Writer`](@ref).
The reader caches the domains, meshes, topologies, grids, and spaces it reads so that
duplicate objects are not created.

`context` is the `ClimaComms` context of the run (`ClimaComms.context()`); with
an `MPICommsContext` the resulting `Field`s are distributed over its ranks,
which requires an HDF5 library with MPI support, as for [`HDF5Writer`](@ref).
The `do`-block form closes the file when the block returns. Opening a file written by a
newer version of ClimaCore logs a warning.

Objects are read with [`read_domain`](@ref), [`read_mesh`](@ref), [`read_topology`](@ref),
[`read_space`](@ref), and [`read_field`](@ref).

# Examples

```julia
InputOutput.HDF5Reader(filename, ClimaComms.context()) do reader
    Y = read_field(reader, "Y")          # the whole FieldVector
    Y.c |> propertynames
    Y.f |> propertynames
    c = read_field(reader, "Y/c")        # one component, by its slash path
    ρ_field = c.ρ
end
```

A `FieldVector` named `"Y"` with components `c` and `f` is stored as the fields
`"Y/c"` and `"Y/f"`; the members of a `NamedTuple`-valued field (`Y.c.ρ`) are
not addressable separately.

The caches expose what has been read so far, e.g. `reader.space_cache` is a `Dict` from
space names such as `"horizontal_space"` to the space objects. Fields read this way can be
plotted by loading a Makie backend (e.g. `CairoMakie`), which activates ClimaCore's
plotting extension (`ClimaCore.Visualize`), and interpolated onto other grids with
`ClimaCoreTempestRemap`.
"""
struct HDF5Reader{C <: ClimaComms.AbstractCommsContext}
    file::HDF5.File
    context::C
    file_version::VersionNumber
    domain_cache::Dict{Any, Any}
    mesh_cache::Dict{Any, Any}
    topology_cache::Dict{Any, Any}
    grid_cache::Dict{Any, Any}
    space_cache::Dict{Any, Any}
end

function HDF5Reader(
    f::Function,
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext,
)
    reader = HDF5Reader(filename, context)
    try
        f(reader)
    finally
        Base.close(reader)
    end
end

function HDF5Reader(
    filename::AbstractString,
    context::ClimaComms.AbstractCommsContext,
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
    current_version = VERSION
    if file_version > current_version
        @warn "$filename was written using a newer version of ClimaCore than is currently loaded" file_version current_version
    end
    return HDF5Reader(
        file,
        context,
        file_version,
        Dict(),
        Dict(),
        Dict(),
        Dict(),
        Dict(),
    )
end


function Base.close(hdfreader::HDF5Reader)
    @debug "closing file" hdfreader.context pid =
        ClimaComms.mypid(hdfreader.context)

    empty!(hdfreader.domain_cache)
    empty!(hdfreader.mesh_cache)
    empty!(hdfreader.topology_cache)
    empty!(hdfreader.grid_cache)
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
    quadraturestring == "GLL" && return Quadratures.GLL{npts}()
    quadraturestring == "GL" && return Quadratures.GL{npts}()
    quadraturestring == "Uniform" && return Quadratures.Uniform{npts}()
    return Quadratures.ClosedUniform{npts}()
end

"""
    _scan_data_layout(layoutstring, T, array)

Construct a `DataLayout` with element type `T` from an `array` read out of an
HDF5 file, given the `data_layout` string stored in the file. Layout strings
written by older versions of ClimaCore (e.g. "IJFH" or "VF") are mapped to
either `VIJFH` or `VIJHF`, depending on the relative order of their `F` and `H`
axes. Prepending or inserting axes of length 1 does not change how the entries
of `array` are ordered in memory, so this mapping preserves all data.
"""
function _scan_data_layout(
    layoutstring::AbstractString,
    ::Type{T},
    array,
) where {T}
    layoutstring == "DataF" && return DataLayouts.DataF{T}(array)
    f_pos = findfirst('F', layoutstring)
    h_pos = findfirst('H', layoutstring)
    i_pos = findfirst('I', layoutstring)
    j_pos = findfirst('J', layoutstring)
    Nv = startswith(layoutstring, "V") ? size(array, 1) : 1
    Ni = isnothing(i_pos) ? 1 : size(array, i_pos)
    Nj = isnothing(j_pos) ? 1 : size(array, j_pos)
    Nh = isnothing(h_pos) ? 1 : size(array, h_pos)
    Nf = DataLayouts.num_basetypes(eltype(array), T)
    F = !isnothing(h_pos) && !isnothing(f_pos) && h_pos < f_pos ? 5 : 4
    VIJH = F == 5 ? DataLayouts.VIJHF : DataLayouts.VIJFH
    array_size = DataLayouts.add_f_dim((Nv, Ni, Nj, Nh), Nf, Val(F))
    return VIJH{T, Nv, Ni, Nj, nothing}(reshape(array, array_size))
end

# Axis of the `H` dimension in an array stored with the given layout string.
_scan_h_dim(layoutstring::AbstractString) = findfirst('H', layoutstring)

# Reconstruct the Nh type parameter used by fields on `space`, so that reading
# a field is an exact inverse of writing it.
function _match_space_layout(values, space)
    lg_data = Spaces.local_geometry_data(space)
    lg_data isa DataLayouts.DataLayout{<:Any, 0} && return values
    Nh = DataLayouts.shape_params(lg_data).Nh
    return DataLayouts.layout_constructor(values; Nh)(parent(values))
end

"""
    matrix_to_cartesianindices(elemorder_matrix)

Convert the `elemorder_matrix`, whose rows are element indices, to an array of
`CartesianIndex` with one dimension per column, sized by the maximum index in each.
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

Read the domain named `name` from `reader`, or from the reader cache if it has already
been read.
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

Read the mesh named `name` from `reader`, or from the reader cache if it has already been
read.
"""
function read_mesh(reader, name)
    Base.get!(reader.mesh_cache, name) do
        read_mesh_new(reader, name)
    end
end

function read_mesh_new(reader::HDF5Reader, name::AbstractString)
    group = get_key(reader.file, "meshes/$name")
    type = attrs(group)["type"]
    if type == "IntervalMesh"
        domain = read_domain(reader, attrs(group)["domain"])
        nelements = attrs(group)["nelements"]
        faces_type = get(attrs(group), "faces_type", nothing)
        stretch_type = get(attrs(group), "stretch_type", nothing)
        reverse_mode = get(attrs(group), "reverse_mode", false)
        if stretch_type == "Uniform" || faces_type == "Range"
            return Meshes.IntervalMesh(
                domain,
                Meshes.Uniform();
                nelems = nelements,
                reverse_mode,
            )
        end
        stretch_params = get(attrs(group), "stretch_params", nothing)
        if stretch_type ≠ "UnknownStretch" && !isnothing(stretch_params)
            CT = Domains.coordinate_type(domain)
            stretch =
                getproperty(Meshes, Symbol(stretch_type))(stretch_params...)
            return Meshes.IntervalMesh(
                domain,
                stretch;
                nelems = nelements,
                reverse_mode,
            )
        end
        # Fallback: read from array
        @assert faces_type == "Array"
        CT = Domains.coordinate_type(domain)
        faces = [CT(coords) for coords in attrs(group)["faces"]]
        return Meshes.IntervalMesh(domain, faces)
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

Read the topology named `name` from `reader`, or from the reader cache if it has already
been read. A `Topology2D` is created with the reader's context, so it is distributed when
the context is an `MPICommsContext`.
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
        device = ClimaComms.device(reader.context)
        context = ClimaComms.SingletonCommsContext(device)
        return Topologies.IntervalTopology(device, mesh)
    elseif type == "Topology2D"
        mesh = read_mesh(reader, attrs(group)["mesh"])
        if haskey(group, "elemorder")
            elemorder_matrix = read(group, "elemorder")
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

        return Topologies.Topology2D(reader.context, mesh, elemorder)
    else
        error("Unsupported type $type")
    end
end



"""
    read_grid(reader::AbstractReader, name)

Read the grid named `name` from `reader`, or from the reader cache if it has already been
read.
"""
function read_grid(reader, name)
    Base.get!(reader.grid_cache, name) do
        read_grid_new(reader, name)
    end
end

"""
    read_data_layout(dataset, topology)

Read a `DataLayout` from the HDF5 `dataset` for the given `topology`, selecting the
elements local to this process when `topology` is a `Topology2D`. This is the inverse of
the [`write!`](@ref) method for data layouts.
"""
function read_data_layout(dataset, topology)
    ArrayType = ClimaComms.array_type(topology)
    data_layout = HDF5.read_attribute(dataset, "type")
    array = read(dataset)
    if topology isa Topologies.Topology2D
        h_dim = _scan_h_dim(data_layout)
        nd = ndims(array)
        localidx = ntuple(d -> d == h_dim ? topology.local_elem_gidx : (:), nd)
        data = ArrayType(array[localidx...])
    else
        data = ArrayType(read(array))
    end
    ElType = read_type(HDF5.read_attribute(dataset, "data_eltype"))
    return _scan_data_layout(data_layout, ElType, data)
end

function read_grid_new(reader, name)
    group = reader.file["grids/$name"]
    type = attrs(group)["type"]
    if type in ("SpectralElementGrid1D", "SpectralElementGrid2D")
        npts = attrs(group)["quadrature_num_points"]
        quadrature_style =
            _scan_quadrature_style(attrs(group)["quadrature_type"], npts)
        topology = read_topology(reader, attrs(group)["topology"])
        enable_bubble = get(attrs(group), "bubble", "false") == "true"
        # A grid group without a "discretization" attribute reads back as CG,
        # unless it has the legacy "discontinuous" attribute (files written
        # before "discretization" replaced it).
        legacy_default =
            get(attrs(group), "discontinuous", "false") == "true" ? "DG" : "CG"
        discretization =
            get(attrs(group), "discretization", legacy_default) == "DG" ?
            Grids.DG() : Grids.CG()
        if type == "SpectralElementGrid1D"
            return Grids.SpectralElementGrid1D(
                topology,
                quadrature_style;
                discretization,
            )
        else
            enable_mask = haskey(attrs(group), "grid_mask")
            grid = Grids.SpectralElementGrid2D(
                topology,
                quadrature_style;
                enable_bubble,
                enable_mask,
                discretization,
            )
            if enable_mask
                # the attribute stores the mask's group name ("IJHMask", with
                # a numbered suffix when a file holds several masks)
                mask_name = attrs(group)["grid_mask"]
                @assert startswith(mask_name, "IJHMask")
                ds_is_active = reader.file["grid_mask"][mask_name]["is_active"]
                is_active = read_data_layout(ds_is_active, topology)
                Grids.set_mask!(grid, is_active)
            end
            return grid
        end
    elseif type == "FiniteDifferenceGrid"
        topology = read_topology(reader, attrs(group)["topology"])
        return Grids.FiniteDifferenceGrid(topology)
    elseif type == "MultiPointGrid"
        radius = attrs(group)["radius"]
        coords = read(group, "points")
        points = [
            Geometry.LatLongPoint(coords[1, i], coords[2, i]) for
            i in 1:size(coords, 2)
        ]
        device = ClimaComms.device(reader.context)
        return Grids.MultiPointGrid(points; radius, device)
    elseif type == "ExtrudedFiniteDifferenceGrid"
        vertical_grid = read_grid(reader, attrs(group)["vertical_grid"])
        horizontal_grid = read_grid(reader, attrs(group)["horizontal_grid"])
        hypsography_type = get(attrs(group), "hypsography_type", "Flat")
        deep = get(attrs(group), "deep", false)
        if hypsography_type == "Flat"
            hypsography = Grids.Flat()
        elseif hypsography_type == "LinearAdaption"
            hypsography = Hypsography.LinearAdaption(
                read_field(reader, attrs(group)["hypsography_surface"]),
            )
        elseif hypsography_type == "SLEVEAdaption"
            # Store hyps object for general use ?
            hypsography = Hypsography.SLEVEAdaption(
                read_field(reader, attrs(group)["hypsography_surface"]),
                get(attrs(group), "hypsography_ηₕ", "hypsography_surface_ηₕ"),
                get(attrs(group), "hypsography_s", "hypsography_surface_s"),
            )
        else
            error("Unsupported hypsography type $hypsography_type")
        end
        return Grids.ExtrudedFiniteDifferenceGrid(
            horizontal_grid,
            vertical_grid,
            hypsography;
            deep,
        )
    elseif type == "LevelGrid"
        full_grid = read_grid(reader, attrs(group)["full_grid"])
        if haskey(attrs(group), "level")
            level = attrs(group)["level"]
        else
            level = attrs(group)["level_half"] + half
        end

        return Grids.LevelGrid(full_grid, level)
    else
        error("Unsupported grid type $type")
    end
end

"""
    read_space(reader::AbstractReader, name)

Read the space named `name` from `reader`, or from the reader cache if it has already
been read.
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
                hypsography = Grids.Flat()
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
    else
        error("Unsupported space type $type")
    end
end

"""
    read_field(reader, name)

Read the `Field` or `FieldVector` named `name` from `reader`. Fields are not cached, so
reading the same field multiple times creates distinct objects. The components of a
`FieldVector` are addressed by their slash path, e.g. `"Y/c"`.
"""
function read_field(reader::HDF5Reader, name::AbstractString)
    key = "fields/$name"
    obj = get_key(reader.file, key)
    type = attrs(obj)["type"]
    if type == "Field"
        if haskey(attrs(obj), "grid")
            grid = read_grid(reader, attrs(obj)["grid"])
            staggering = get(attrs(obj), "staggering", nothing)
            if staggering == "CellCenter"
                staggering = Grids.CellCenter()
            elseif staggering == "CellFace"
                staggering = Grids.CellFace()
            end
            space = Spaces.space(grid, staggering)
        elseif haskey(attrs(obj), "space")
            space = read_space(reader, attrs(obj)["space"])
        else
            # if the there is no grid, then the field is on a PointSpace
            lg_obj = reader.file["local_geometry_data/$name"]
            ArrayType = ClimaComms.array_type(ClimaComms.device(reader.context))
            lg_data = ArrayType(read(lg_obj))
            # For a point space, the data layout of local_geometry_data is always DataF.
            lg_type = read_type(attrs(lg_obj)["local_geometry_type"])
            local_geometry_data = DataLayouts.DataF{lg_type}(lg_data)
            space = Spaces.PointSpace(local_geometry_data)
            topology = nothing
        end
        if space isa Union{
            Spaces.MultiPointSpace,
            Spaces.MultiColumnFiniteDifferenceSpace,
        }
            topology = nothing
            ArrayType = ClimaComms.array_type(ClimaComms.device(reader.context))
        elseif !(space isa Spaces.AbstractPointSpace)
            topology = Spaces.topology(space)
            ArrayType = ClimaComms.array_type(topology)
        end
        data_layout = attrs(obj)["data_layout"]
        if topology isa Topologies.Topology2D
            h_dim = _scan_h_dim(data_layout)
            nd = ndims(obj)
            localidx =
                ntuple(d -> d == h_dim ? topology.local_elem_gidx : (:), nd)
            data = ArrayType(obj[localidx...])
        else
            data = ArrayType(read(obj))
        end
        # The `value_type` attribute is deprecated. here we mantain backwards compatibility
        ElType = read_type(
            haskey(attrs(obj), "field_eltype") ?
            attrs(obj)["field_eltype"] : attrs(obj)["value_type"],
        )
        values = _scan_data_layout(data_layout, ElType, data)
        return Fields.Field(_match_space_layout(values, space), space)
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

"""
    read_attributes(reader::AbstractReader, name::AbstractString)

Return a `Dict` of the attributes of the object at path `name` in the HDF5 file of
`reader`.
"""
read_attributes(reader::HDF5Reader, name::AbstractString) =
    h5readattr(reader.file.filename, name)
