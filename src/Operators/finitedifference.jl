#=
1) face -> center
 - Dirichlet: overriding the current face value, continue as usual
 - Neumann: does this actually do anything?
 
2) center -> face
 - Dirichlet: 2 point stencil (prescribed boundary + center)
 - Neumann: set the value of the result

1) Face data `f`
 - Dirichlet bcs: set the value directly
 - Neumann bcs: one-sided stencil, set value of gradient

2) Center data `f`
 - Dirichlet bcs: invert interpolation stencil, such that `f_boundary = f_bvalue`
 - Neumann bcs: one-sided (modified) stencil

 A) Have field `f`, `∇f`
 B) Solving A(x_interior) = b - Ax_bc
 - Dirichlet BCS: Ax = b
 - A(x_interior) = b - Ax_bc

 e.g. c2c laplacian, n centers, n+1 faces

1) take gradient to faces
  - apply stencil to interior (i = 2:n)
  - Dirichlet: 2 point (center - boundary stencil)
  - Neumann: setting the value at the face 

 [ x          ;   + something on the RHS if inhomogeneous Dirichlet
   x x .      ;
   . x x      ;
          x x ;
            . ]   + something on the RHS if inhomogeneous Neumann

2) gradient back to center (no boundary adjustment required)
[ x x      ;
    x x    ;
           ;
       x x ]
=#

#=
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

# face  center
#   i
#         i
#  i+1

abstract type FiniteDifferenceOperator end

abstract type InterpolationOperator <: FiniteDifferenceOperator end

# unweighted interpolation operators
struct FaceToCenterInterpolation <: InterpolationOperator end
struct CenterToFaceInterpolation <: InterpolationOperator end

# TODO: type stability of return type based on data undertype
function stencil(::FaceToCenterInterpolation, data, space::Spaces.FaceFiniteDifferenceSpace, idx)
    return (data[idx] + data[idx-1]) / 2
end

function stencil(::CenterToFaceInterpolation, data, space::Spaces.CenterFiniteDifferenceSpace, idx)
    return (data[idx] + data[idx+1]) / 2
end

# weighted interpolation operators
struct WeightedInterpolationOperator <: FiniteDifferenceOperator end

struct WeightedFaceToCenterInterpolation <: InterpolationOperator end
struct WeightedCenterToFaceInterpolation <: InterpolationOperator end

function stencil(::Weighted 
end

function stencil(op::WeightedInterpolationOperator, field, weight, idx)
    field_data= Fields.field_values(field)
    field_space = Fields.space(field)
    weight_data = Fields.field_values(weight)
    weight_space = Fields.space(weight)
    return stencil(op, field_data, field_space, weight_data, weight_space, idx)
end

# derivitive operators
struct CenteredDerivative <: FiniteDifferenceOperator end

function stencil(::CenteredDerivative, data, space::Spaces.CenterFiniteDifferenceSpace, idx)
    return RecursiveApply.rdiv(data[idx] ⊟ data[idx-1], Spaces.Δcoordinate(space, idx))
end

function stencil(::CenteredDerivative, data, space::Spaces.FaceFiniteDifferenceSpace, idx)
    return RecursiveApply.rdiv(data[idx+1] ⊟ data[idx], Spaces.Δcoordinate(space, idx))
end

function stencil(op::FiniteDifferenceOperator, field, idx)
    data = Fields.field_values(field)
    space = Fields.space(field)
    return stencil(op, data, space, idx)
end

"""
    AdvectionOperator

Computes an advection of `A` by `v`, where
- `A` is a field stored at cell centers 
- `v` is a vector field stored at cell faces

Returns a field stored at cell faces
"""
abstract type AdvectionOperator <: FiniteDifferenceOperator end

struct LeftAdvectionOperator <: AdvectionOperator end
struct RightAdvectionOperator <: AdvectionOperator end
struct UpwindAdvectionOperator <: AdvectionOperator end

function stencil(::LeftAdvectionOperator, 
                scalar_data, scalar_space::Spaces.CenterFiniteDifferenceSpace, 
                vector_data, vector_space::Spaces.FaceFiniteDifferenceSpace, 
                idx)
    return scalar_data[idx-1] ⊠ vector_data[idx]
end

function stencil(::RightAdvectionOperator,
                 scalar_data, scalar_space::Spaces.CenterFiniteDifferenceSpace,
                 vector_data, vector_space::Spaces.FaceFiniteDifferenceSpace,
                 idx)
    return scalar_data[idx] ⊠ vector_data[idx]
end

function stencil(::UpwindAdvectionOperator, 
                 scalar_data, scalar_space::Spaces.CenterFiniteDifferenceSpace,
                 vector_field, vector_space::Spaces.FaceFiniteDifferenceSpace,
                 idx)
    # v > 0
    l = ((v ⊞ abs.(v)) ⊠ 0.5) ⊠ stencil(LeftOperator(), scalar_data, scalar_space, vector_data, vector_space, idx)
    # v < 0
    r = ((v ⊟ abs.(v)) ⊠ 0.5) ⊠ stencil(RightOperator(), scalar_data, scalar_space, vector_data, vector_space, idx)
    return l ⊞ r
end

function stencil(op::AdvectionOperator, scalar_field, vector_field, idx)
    scalar_data = Fields.field_values(scalar_field)
    scalar_space = Fields.space(scalar_field) 
    vector_data = Fields.field_values(vector_field)
    vector_space = Fields.space(vector_field)
    return stencil(op, scalar_data, scalar_space, vector_data, vector_space, idx)
end

#=
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
    space_from1 = Fields.space(field_from1)

    data_from2 = Fields.field_values(field_from2)
    space_from2  = Fields.space(field_from2)

    data_to = Fields.field_values(field_to)
    space_to = Fields.space(field_to)

    for i in eachrealindex(space_to)
        data_to[i] = stencil(operator, space_to, space_from1, space_from2, i)
    end
    return field_to
end
=#

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
