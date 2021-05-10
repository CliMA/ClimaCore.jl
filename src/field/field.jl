module Fields

import ..slab, ..column
import ..DataLayouts
import ..DataLayouts: AbstractData, DataStyle
import ..Meshes: AbstractMesh, Quadratures
import ..Operators
import ..Geometry
import ..Geometry: Cartesian12Vector

import LinearAlgebra

using Requires

function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        function OrdinaryDiffEq.calculate_residuals!(
            out::Fields.Field,
            ũ::Fields.Field,
            u₀::Fields.Field,
            u₁::Fields.Field,
            α,
            ρ,
            internalnorm,
            t,
        )
            OrdinaryDiffEq.calculate_residuals!(
                parent(out),
                parent(ũ),
                parent(u₀),
                parent(u₁),
                α,
                ρ,
                internalnorm,
                t,
            )
        end
    end
end


"""
    Field(values, mesh)

A set of `values` defined at each point of a `mesh`.
"""
struct Field{V <: AbstractData, M <: AbstractMesh}
    values::V
    mesh::M
    # add metadata/attributes?
    function Field{V, M}(values::V, mesh::M) where {V, M}
        # need to enforce that the data size matches the mesh
        # @assert support(values) === support(mesh.coordinates)
        @assert size(values) == size(mesh.coordinates)
        return new{V, M}(values, mesh)
    end
end
Field(values::V, mesh::M) where {V <: AbstractData, M <: AbstractMesh} =
    Field{V, M}(values, mesh)

Base.propertynames(field::Field) = propertynames(getfield(field, :values))
field_values(field::Field) = getfield(field, :values)
mesh(field::Field) = getfield(field, :mesh)

# need to define twice to avoid ambiguities
Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(field_values(field), name), mesh(field))
Base.getproperty(field::Field, name::Integer) =
    Field(getproperty(field_values(field), name), mesh(field))

Base.eltype(field::Field) = eltype(field_values(field))
Base.parent(field::Field) = parent(field_values(field))
Base.size(field::Field) = () # to play nice with DifferentialEquations; may want to revisit this

function _show_compact_field(io, field, prefix, isfirst = false)
    #print(io, prefix1)
    if eltype(field) <: Number
        if isfirst
            print(io, "\n", prefix)
        end
        print(
            IOContext(io, :compact => true, :limit => true),
            vec(parent(field)),
        )
    else
        names = propertynames(field)
        for name in names
            print(io, "\n", prefix)
            print(io, name, ": ")
            _show_compact_field(io, getproperty(field, name), prefix * "  ")
        end
    end
end

function Base.show(io::IO, field::Field)
    print(io, eltype(field), "-valued Field:")
    _show_compact_field(io, field, "  ", true)
    print(io, "\non ", mesh(field))
end


# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/DiffOperators.jl#L5
# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/FieldsInterfaces.jl#L70

# TODO: nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html


# Broadcasting
struct FieldStyle{DS <: DataStyle} <: Base.BroadcastStyle end
FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()

Base.Broadcast.BroadcastStyle(::Type{Field{V, M}}) where {V, M} =
    FieldStyle(DataStyle(V))


Base.Broadcast.BroadcastStyle(
    a::Base.Broadcast.AbstractArrayStyle{0},
    b::FieldStyle,
) = b

Base.Broadcast.broadcastable(field::Field{V, M}) where {V, M} = field

Base.axes(field::Field) = (mesh(field),)

# TODO: we may want to rethink how we handle this to allow for more lazy operations
todata(obj) = obj
todata(field::Field) = field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    Base.Broadcast.Broadcasted{DS}(bc.f, map(todata, bc.args))
end

# we specialize handling of +, *, muladd, so that we can support broadcasting over NamedTuple element types
# TODO: it may be more efficient to handle this at the array level?
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(+), args...) =
    Base.Broadcast.broadcasted(fs, Operators.:⊞, args...)

Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(-), args...) =
    Base.Broadcast.broadcasted(fs, Operators.:⊟, args...)

Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(*), args...) =
    Base.Broadcast.broadcasted(fs, Operators.:⊠, args...)

# required for adaptive timestepping
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(/), args...) =
    Base.Broadcast.broadcasted(fs, Operators.rdiv, args...)


Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(muladd), args...) =
    Base.Broadcast.broadcasted(fs, Operators.rmuladd, args...)


function mesh(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    return axes(bc)[1]
end

function Base.similar(field::Field, ::Type{Eltype}) where {Eltype}
    return Field(similar(field_values(field), Eltype), mesh(field))
end
function Base.similar(field::Field)
    return similar(field, eltype(field))
end
function Base.similar(field::F, ::Type{F}) where {F <: Field}
    return similar(field)
end

function Base.copy(field::Field)
    Field(copy(field_values(field)), mesh(field))
end
# we implement our own to avoid the type-widening code, and throw a more useful error
@inline function Base.copy(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: FieldStyle}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    if Base.isconcretetype(ElType)
        # We can trust it and defer to the simpler `copyto!`
        return copyto!(similar(bc, ElType), bc)
    end
    error("cannot infer concrete eltype of $(bc.f) on $(map(eltype, bc.args))")
end

function Base.copyto!(dest::Field{V, M}, src::Field{V, M}) where {V, M}
    #@assert mesh(dest) == mesh(src)
    copyto!(field_values(dest), field_values(src))
    return dest
end


function Base.zero(field::Field)
    zfield = similar(field::Field)
    zarray = parent(Fields.field_values(zfield))
    fill!(zarray, zero(eltype(zarray)))
    return zfield
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
    ::Type{Eltype},
) where {DS, Eltype}
    return Field(similar(todata(bc), Eltype), mesh(bc))
end

function Base.copyto!(
    dest::Field{V, M},
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
) where {V, M, DS}
    copyto!(field_values(dest), todata(bc))
    return dest
end

function Base.deepcopy(::Field)
    error("oh no")
end


function Base.Broadcast.check_broadcast_shape(
    (mesh1,)::Tuple{AbstractMesh},
    (mesh2,)::Tuple{AbstractMesh},
)
    if mesh1 !== mesh2
        error("Mismatched meshes\n$mesh1\n$mesh2")
    end
    return nothing
end
function Base.Broadcast.check_broadcast_shape(
    ax1::Tuple{AbstractMesh},
    ax2::Tuple{},
)
    return nothing
end
function Base.Broadcast.check_broadcast_shape(
    ax1::Tuple{AbstractMesh},
    ax2::Tuple,
)
    error("$ax2 is not a Mesh")
end


# useful operations
Base.map(fn, field::Field) = Base.broadcast(fn, field)

# sum will give the integral over the field
function Base.sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}})
    Base.reduce(
        Operators.:⊞,
        Base.Broadcast.broadcasted(
            Operators.:⊠,
            mesh(field).local_geometry.WJ,
            todata(field),
        ),
    )
end
function Base.sum(fn, field::Field)
    # Can't just call mapreduce as we need to weight _after_ applying the function
    Base.sum(Base.Broadcast.broadcasted(fn, field))
end

function LinearAlgebra.norm(field::Field, p::Real = 2)
    if p == 2
        # currently only one which supports structured types
        sqrt(sum(LinearAlgebra.norm_sqr, field))
    elseif p == 1
        sum(abs, field)
    elseif p == Inf
        error("Inf norm not yet supported")
    else
        sum(x -> x^p, field)^(1 / p)
    end
end

function Base.isapprox(
    x::Field,
    y::Field;
    atol::Real = 0,
    rtol::Real = Base.rtoldefault(eltype(parent(x)), eltype(parent(y)), atol),
    nans::Bool = false,
    norm::Function = LinearAlgebra.norm,
)
    d = norm(x .- y)
    return isfinite(d) && d <= max(atol, rtol * max(norm(x), norm(y)))
end



# for compatibility with OrdinaryDiffEq
# Based on ApproxFun definitions
#  https://github.com/SciML/RecursiveArrayTools.jl/blob/6e779acb321560c75e27739a89ae553cd0f332f1/src/init.jl#L8-L12
Base.isnan(field::Field) = any(isnan, parent(field))
Base.any(f, field::Field) = any(f, parent(field))
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/181dcf265351ed3c02437c89a8d2af3f6967fa85/src/initdt.jl#L80

import RecursiveArrayTools

RecursiveArrayTools.recursive_unitless_eltype(field::Field) = typeof(field)
RecursiveArrayTools.recursive_unitless_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_unitless_bottom_eltype(parent(field))
RecursiveArrayTools.recursive_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_bottom_eltype(parent(field))

function RecursiveArrayTools.recursivecopy!(dest::F, src::F) where {F <: Field}
    copy!(parent(dest), parent(src))
    return dest
end
RecursiveArrayTools.recursivecopy(field::Field) = copy(field)
RecursiveArrayTools.recursivecopy(
    a::AbstractArray{F, N},
) where {F <: Field, N} = map(RecursiveArrayTools.recursivecopy, a)

"""
    coordinate_field(mesh::AbstractMesh)

Construct a `Field` of the coordinates of the mesh.
"""
coordinate_field(mesh::AbstractMesh) = Field(mesh.coordinates, mesh)
coordinate_field(field::Field) = coordinates(mesh(field))

function interpcoord(elemrange, x::Real)
    n = length(elemrange) - 1
    z = x == elemrange[end] ? n : searchsortedlast(elemrange, x) # element index
    @assert 1 <= z <= n
    lo = elemrange[z]
    hi = elemrange[z + 1]
    # Find ξ ∈ [-1,1] such that
    # x = (1-ξ)/2 * lo + (1+ξ)/2 * hi
    #   = (lo + hi) / 2 + ξ * (hi - lo) / 2
    ξ = (2x - (lo + hi)) / (hi - lo)
    return z, ξ
end

import ..Operators

import ..Meshes

function matrix_interpolate(
    field::Field,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    fieldmesh = mesh(field)
    discretization = fieldmesh.topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2

    interp_data =
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))

    M = Quadratures.interpolation_matrix(
        Float64,
        Q_interp,
        fieldmesh.quadrature_style,
    )
    Operators.tensor_product!(interp_data, field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())



function Operators.slab_gradient!(∇field::Field, field::Field)
    @assert mesh(∇field) === mesh(field)
    Operators.slab_gradient!(
        field_values(∇field),
        field_values(field),
        mesh(field),
    )
    return ∇field
end
function Operators.slab_divergence!(divflux::Field, flux::Field)
    @assert mesh(divflux) === mesh(flux)
    Operators.slab_divergence!(
        field_values(divflux),
        field_values(flux),
        mesh(flux),
    )
    return divflux
end
function Operators.slab_weak_divergence!(divflux::Field, flux::Field)
    @assert mesh(divflux) === mesh(flux)
    Operators.slab_weak_divergence!(
        field_values(divflux),
        field_values(flux),
        mesh(flux),
    )
    return divflux
end

function Operators.slab_gradient(field::Field)
    S = eltype(field)
    ∇S = Operators.rmaptype(T -> Cartesian12Vector{T}, S)
    Operators.slab_gradient!(similar(field, ∇S), field)
end

function Operators.slab_divergence(field::Field)
    S = eltype(field)
    divS = Operators.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_divergence!(similar(field, divS), field)
end
function Operators.slab_weak_divergence(field::Field)
    S = eltype(field)
    divS = Operators.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_weak_divergence!(similar(field, divS), field)
end

include("plots.jl")


end # module
