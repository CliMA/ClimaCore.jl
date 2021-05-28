
# face  center
#   i
#         i
#  i+1

abstract type FiniteDifferenceOperator end

struct CenteredDerivative <: FiniteDifferenceOperator end


# should CenterFiniteDifferenceSpace store the c2c Δh, or the f2f?

function result_space(::CenteredDerivative, space::CenterFiniteDifferenceSpace)
    space.facefield
end

function result_space(::CenteredDerivative, space::FaceFiniteDifferenceSpace)
    space.centerfield
end

function stencil(::CenteredDerivative, field::CenterFiniteDifferenceField, idx)
    data = Fields.field_values(field)
    space = Fields.space(field)
    return (data[idx] ⊟ data[idx-1]) / space.Δh_c2c[idx]
end

function stencil(::CenteredDerivative, field::FaceFiniteDifferenceField, idx)
    data = Fields.field_values(field)
    space = Fields.space(field)
    return (data[idx+1] ⊟ data[idx]) / space.Δh_f2f[idx]
end

"""
    AdvectionOperator

Computes an advection of `A` by `v`, where
- `A` is a field stored at cell centers 
- `v` is a vector field stored at cell faces

Returns a field stored at cell faces
"""
abstract type AdvectionOperator <: FiniteDifferenceOperator end

function result_space(::AdvectionOperator, scalar_space::CenterFiniteDifferenceSpace, vector_space::FaceFiniteDifferenceSpace)
    vector_space.facefield
end

struct LeftAdvectionOperator <: FiniteDifferenceOperator end
struct RightAdvectionOperator <: FiniteDifferenceOperator end
struct UpwindAdvectionOperator <: FiniteDifferenceOperator end

function stencil(::LeftAdvectionOperator, scalar_field::CenterFiniteDifferenceField, vector_field::FaceFiniteDifferenceField, idx)
    scalar_data = Fields.field_values(scalar_field)
    vector_data = Fields.field_values(vector_field)
    return scalar_data[idx-1] * vector_data[idx]
end

# FiniteDifferenceSpace
#  full indices (what we store) 1:N
#  valid indices ()
#  real indices  (non-halo k:n+k)

#=
#  n-element mesh + h halo on each scalar_field
#   - n + 2h center values
#   - n + 2h + 1 face values
#  apply upwind advection operator (defines values at faces)   
#   - face values 2:n+2h are valid
#  apply centered FiniteDifferenceSpace (gives values at center)
#   - center values 2:n+2h-1 are valid

# CenterFiniteDifferenceSpace
# Halo{CenterFiniteDifferenceSpace}

=#
mutable struct CenterFiniteDifferenceSpace{FT,C,H,FS}
    domain::IntervalDomain{FT}
    coordinates::C
    Δh::H
    facespace::FS
end

mutable struct FaceFiniteDifferenceSpace{FT,C,H,CS}
    domain::IntervalDomain{FT}
    coordinates::C
    Δh::H
    centerspace::CS
end

function CenterFiniteDifferenceSpace(domain::IntervalDomain, n::Integer)
    


#=

struct CenterFiniteDifferenceSpace <: FiniteDifferenceSpace
    fullspace::CenterFiniteDifferenceSpace
    validrange::UnitRange
end

function tendency!(dθdt, θ, v, t)
    # θ = scalar # centers 1:n+2
    # v = vector # faces 1:n+3
    # 1. fill in ghost values θ[1], θ[n+2]
    # 2. apply upwind operator faces at face indices 2:n+2
    #    U = Upwind(θ,v)
    # 3. apply difference operator at centere indices 2:n+1 
    #    dθdt = CenteredDerivative(U)   

    # Prescribed flux form
    # θ = scalar # centers 1:n
    # v = vector # faces 1:n+1
    # 1. apply upwind operator faces at face indices 2:n
    #    F = Upwind(θ,v)
    # 2. Set the F[1] and F[n+1] to the desired fluxes
    # 3. apply difference operator at centere indices 1:n
    #    dθdt = CenteredDerivative(F)
end

=#




function stencil(::RightAdvectionOperator, scalar_field::CenterFiniteDifferenceSpace, vector_field::FaceFiniteDifferenceSpace, idx)
    scalar_data = Fields.field_values(scalar_field)
    vector_data = Fields.field_values(vector_field)
    return scalar_data[idx] * vector_data[idx]
end

function stencil(::UpwindAdvectionOperator, scalar_field::CenterFiniteDifferenceSpace, vector_field::FaceFiniteDifferenceSpace, idx)
    (v + abs(v))/2 * stencil(LeftOperator(), scalar_field, vector_field, idx) + # v > 0
    (v - abs(v))/2 * stencil(RightOperator(), scalar_field, vector_field, idx)  # v < 0
end

struct StencilStyle <: Base.BroadcastStyle end

function Base.BroadcastStyle(::Type{<:FiniteDifferenceOperator})
    return StencilStyle()
end

function Base.getindex(bc::Broadcasted{StencilStyle}, I::Integer)
    stencil(bc.fn, bc.args..., I)
end

(op::AdvectionOperator)(fields) = Base.Broadcast.broadcasted(StencilStyle(), op, fields)
(op::CenteredDerivative)(fields) = Base.Broadcast.broadcasted(StencilStyle(), op, fields)

function Base.Broadcast.instantiate(bc::Broadcasted{StencilStyle})
    spaces = map(arg -> axes(arg)[1], bc.args)
    axes = (result_space(bc.fn, spaces...),)            
    if !isnothing(bc.axes)
        @assert axes == bc.axes
    end
    return Broadcasted{Style}(bc.f, bc.args, axes)
end




function apply_stencil!(field_to, operator, field_from)
    space_from = Fields.space(field_from)
    space_to = Fields.space(field_to)
    data_to = Fields.field_values(field_to)
    for i in eachrealindex(space_to)
        data_to[i] = stencil(operator, space_to, space_from, i)
    end
    return field_to
end


function apply_stencil!(field_to, operator, field_from1, field_from2)
    data_from1 = Fields.field_values(field_from1)
    space_from1  = Fields.space(field_from1)
    data_from2 = Fields.field_values(field_from2)
    space_from2  = Fields.space(field_from2)

    data_to = Fields.field_values(field_to)
    space_to = Fields.space(field_to)

    for i in eachrealindex(space_to)
        data_to[i] = stencil(operator, space_to, data_from1, space_from1, data_from2, space_from2, i)
    end
    return field_to
end

#=
centered.(upwind.(a, v) 

a :: CenterFiniteDifferenceField
axes(a) = CenterFiniteDifferenceMesh(..., nghost=1)
v :: FaceFiniteDifferenceField
axes(v) = FaceFiniteDifferenceField(..., nghost=1)

F = Upwind(a,v) :: ?
axes(F) = FaceFiniteDifferenceField(..., nghost=0)

D = CenteredDerivative(F)
axes(D) = CenterFiniteDifferenceField(..., nghost=0)

# A on centers, v on faces ∂(Av)/∂z
# CenteredDerivative(Upwind(A,v))


function stencil(::UpwindOperator, data, v,
    space_from::FaceFiniteDifferenceSpace, space_to::CenterFiniteifferenceSpace, i)
    (data[i+1] ⊟ data[i]) / space_from.Δh_f2f
end


# stencil function
# stencil to compute ith center value
δf2c(data, space, i) = (data[i+1] ⊟ data[i])
# stencil to compute ith face value
δc2f(data, space, i) = (data[i] ⊟ data[i-1])


# TODO: Next steps
# be able to lookup or dispatch on column mesh,center / faces when doing operations easily
# reformat this as a map across vertical levels with fields (so we don't have to call parent()
# everywhere, get rid of direct indexing)
# CellFace -> CellCenter method for easily getting local operator (could just return a contant)
# we need to make it more generic so that 2nd order is not hard coded (check at lest 1 different order of stencil)
# clean up the interface in general (rmatrixmultiply?), get rid of LinearAlgebra.dot()
# vertical only fields? (split h and v)
# make local_operators static vectors
=#

function vertical_interp!(
    field_to,
    field_from,
    cm::Spaces.FiniteDifferenceSpace,
)
    len_to = length(parent(field_to))
    len_from = length(parent(field_from))
    FT = Spaces.undertype(cm)
    if len_from == Spaces.n_cells(cm) && len_to == Spaces.n_faces(cm) # CellCent -> CellFace
        # TODO: should we extrapolate?
        n_faces = Spaces.n_faces(cm)
        for j in Spaces.column(cm, Spaces.CellCent())
            if j > 1 && j < n_faces
                i = j - 1
                local_operator = parent(cm.interp_cent_to_face)[i]
                local_stencil = parent(field_from)[i:(i + 1)] # TODO: remove hard-coded stencil size 2
                parent(field_to)[j] = convert(
                    FT,
                    LinearAlgebra.dot(parent(local_operator), local_stencil),
                )
            end
        end
    elseif len_from == Spaces.n_faces(cm) && len_to == Spaces.n_cells(cm) # CellFace -> CellCent
        local_operator = SVector{2, FT}(0.5, 0.5)
        for i in Spaces.column(cm, Spaces.CellCent())
            local_stencil = parent(field_from)[i:(i + 1)] # TODO: remove hard-coded stencil size 2
            parent(field_to)[i] =
                convert(FT, LinearAlgebra.dot(local_operator, local_stencil))
            # field_to[i] = local_operator * local_stencil
        end
        # No extrapolation needed
    elseif len_to == len_from # no interp needed
        return copyto!(field_to, field_from)
    else
        error("Cannot interpolate colocated fields")
    end
end


function vertical_gradient!(
    field_to,
    field_from,
    cm::Spaces.FiniteDifferenceSpace,
)
    len_to = length(parent(field_to))
    len_from = length(parent(field_from))
    FT = Spaces.undertype(cm)
    if len_from == Spaces.n_cells(cm) && len_to == Spaces.n_faces(cm) # CellCent -> CellFace
        # TODO: should we extrapolate?
        n_faces = Spaces.n_faces(cm)
        for j in Spaces.column(cm, Spaces.CellCent())
            if j > 1 && j < n_faces
                i = j - 1
                local_operator = parent(cm.∇_cent_to_face)[j]
                local_stencil = parent(field_from)[i:(i + 1)] # TODO: remove hard-coded stencil size 2
                parent(field_to)[j] = convert(
                    FT,
                    LinearAlgebra.dot(parent(local_operator), local_stencil),
                )
            end
        end
    elseif len_from == Spaces.n_faces(cm) && len_to == Spaces.n_cells(cm) # CellFace -> CellCent
        for i in Spaces.column(cm, Spaces.CellCent())
            local_operator = parent(cm.∇_face_to_cent)[i + 1]
            local_stencil = parent(field_from)[i:(i + 1)] # TODO: remove hard-coded stencil size 2
            parent(field_to)[i] = convert(
                FT,
                LinearAlgebra.dot(parent(local_operator), local_stencil),
            )
            # field_to[i] = local_operator * local_stencil
        end
        # No extrapolation needed
    elseif len_to == len_from # collocated derivative
        if len_from == Spaces.n_faces(cm) # face->face gradient
            # TODO: should we extrapolate?
            n_faces = Spaces.n_faces(cm)
            for i in Spaces.column(cm, Spaces.CellFace())
                if i > 1 && i < n_faces
                    local_operator = parent(cm.∇_face_to_face)[i]
                    local_stencil = parent(field_from)[(i - 1):(i + 1)] # TODO: remove hard-coded stencil size 2
                    parent(field_to)[i] = convert(
                        FT,
                        LinearAlgebra.dot(
                            parent(local_operator),
                            local_stencil,
                        ),
                    )
                end
            end
        elseif len_from == Spaces.n_cells(cm) # cent->cent gradient
            # TODO: should we extrapolate?
            n_cells = Spaces.n_cells(cm)
            for i in Spaces.column(cm, Spaces.CellCent())
                if i > 1 && i < n_cells
                    local_operator = parent(cm.∇_cent_to_cent)[i]
                    local_stencil = parent(field_from)[(i - 1):(i + 1)] # TODO: remove hard-coded stencil size 2
                    parent(field_to)[i] = convert(
                        FT,
                        LinearAlgebra.dot(
                            parent(local_operator),
                            local_stencil,
                        ),
                    )
                end
            end
        else
            error("Bad field") # need to implement collocated operators
        end
    end
    return field_to
end

function apply_dirichlet!(
    field,
    value,
    cm::Spaces.FiniteDifferenceSpace,
    boundary::Spaces.ColumnMinMax,
)
    arr = parent(field)
    len = length(arr)
    if len == Spaces.n_faces(cm)
        boundary_index = Spaces.boundary_index(cm, Spaces.CellFace(), boundary)
        arr[boundary_index] = value
    elseif len == Spaces.n_cells(cm)
        ghost_index = Spaces.ghost_index(cm, Spaces.CellCent(), boundary)
        interior_index = Spaces.interior_index(cm, Spaces.CellCent(), boundary)
        arr[ghost_index] = 2 * value - arr[interior_index]
    else
        error("Bad field")
    end
    return field
end

function apply_neumann!(
    field,
    value,
    cm::Spaces.FiniteDifferenceSpace,
    boundary::Spaces.ColumnMinMax,
)
    arr = parent(field)
    len = length(arr)
    if len == Spaces.n_faces(cm)
        # ∂_x ϕ = n̂ * (ϕ_g - ϕ_i) / (2Δx) # second order accurate Neumann for CellFace data
        # boundry_index = Spaces.boundary_index(cm, Spaces.CellFace(), boundary)
        n̂ = Spaces.n_hat(boundary)
        ghost_index = Spaces.ghost_index(cm, Spaces.CellFace(), boundary)
        interior_index = Spaces.interior_index(cm, Spaces.CellFace(), boundary)
        Δvert = Spaces.Δcoordinates(cm, Spaces.CellFace(), boundary)
        arr[ghost_index] = arr[interior_index] + n̂ * 2 * Δvert * value
    elseif len == Spaces.n_cells(cm)
        # ∂_x ϕ = n̂ * (ϕ_g - ϕ_i) / Δx # second order accurate Neumann for CellCent data
        n̂ = Spaces.n_hat(boundary)
        ghost_index = Spaces.ghost_index(cm, Spaces.CellCent(), boundary)
        interior_index = Spaces.interior_index(cm, Spaces.CellCent(), boundary)
        Δvert = Spaces.Δcoordinates(cm, Spaces.CellCent(), boundary)
        arr[ghost_index] = arr[interior_index] + n̂ * Δvert * value
    else
        error("Bad field")
    end
    return field
end
