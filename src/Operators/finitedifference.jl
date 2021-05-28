
# TODO: Next steps
# be able to lookup or dispatch on column mesh,center / faces when doing operations easily
# reformat this as a map across vertical levels with fields (so we don't have to call parent()
# everywhere, get rid of direct indexing)
# CellFace -> CellCenter method for easily getting local operator (could just return a contant)
# we need to make it more generic so that 2nd order is not hard coded (check at lest 1 different order of stencil)
# clean up the interface in general (rmatrixmultiply?), get rid of LinearAlgebra.dot()
# vertical only fields? (split h and v)
# make local_operators static vectors
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
