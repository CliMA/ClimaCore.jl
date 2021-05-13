module Operators

import ..slab
import ..DataLayouts: Data2D, DataSlab2D
import ..DataLayouts
import ..Geometry
import ..Geometry: Cartesian12Vector, Covariant12Vector, Contravariant12Vector
import ..Meshes
import ..Meshes: Quadratures, AbstractMesh
import ..Topologies
import ..Fields
import ..Fields: Field
using ..RecursiveOperators

import LinearAlgebra
using StaticArrays


# function stubs: definitions are in Fields
function slab_gradient end
function slab_divergence end
function slab_weak_divergence end


# TODO:
#  - convenience operations for fields
#  - determine output element type
#  - let inputs be Broadcasted objects
#    - make sure that
"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product!(
    out::Data2D{S, Nij_out},
    in::Data2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    Nh = length(in)
    @assert Nh == length(out)

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    for h in 1:Nh
        in_slab = slab(in, h)
        out_slab = slab(out, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[i, j] = RecursiveOperators.rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = RecursiveOperators.rmatmul2(M, temp, i, j)
        end
    end
    return out
end
"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product!(
    out_slab::DataLayouts.DataSlab2D{S, Nij_out},
    in_slab::DataLayouts.DataSlab2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    for j in 1:Nij_in, i in 1:Nij_out
        temp[i, j] = RecursiveOperators.rmatmul1(M, in_slab, i, j)
    end
    for j in 1:Nij_out, i in 1:Nij_out
        out_slab[i, j] = RecursiveOperators.rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::Data2D{S, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    tensor_product!(inout, inout, M)
end

abstract type Operator end

struct Gradient <: Operator end

"""
    slab_gradient!(∇data, data, mesh)

Compute the gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
function slab_gradient!(∇data, data, mesh)
    # all derivatives calculated in the reference local geometry FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(data)
    for h in 1:Nh
        ∇data_slab = slab(∇data, h)
        # TODO: can we just call materialize(slab(data,h))
        data_slab = slab(data, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        for i in 1:Nq, j in 1:Nq
            # TODO: materialize data_slab once
            # on GPU this would be done to shared memory, then synchronize threads
            local_geometry = local_geometry_slab[i, j]

            # compute covariant derivatives
            ∂f∂ξ₁ = RecursiveOperators.rmatmul1(D, data_slab, i, j)
            ∂f∂ξ₂ = RecursiveOperators.rmatmul2(D, data_slab, i, j)
            ∂f∂ξ = RecursiveOperators.rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)

            # convert to desired basis
            ∇data_slab[i, j] = RecursiveOperators.rmap(
                x -> Cartesian12Vector(x, local_geometry),
                ∂f∂ξ,
            )
        end
    end
    return ∇data
end


"""
    slab_divergence!(divflux, flux, mesh)

Compute the divergence of `flux`, storing the result in `divflux`.

Given a vector field `v`, we compute the divergence as

    [∂(Jv¹)/∂ξ¹ + ∂(Jv²)/∂ξ²] / J

where `J` is the Jacobian determinant, `vⁱ` is the `i`th contravariant component of `v`.

This is discretized at the quadrature points as

    I{[∂(I{Jv¹})/∂ξ¹ + ∂(I{Jv²})/∂ξ²] / J}

where `I{x}` is the interpolation operator applied to a field `x`.

## References
 - Taylor and Fournier (2010), equation 15
"""
function slab_divergence!(divflux, flux, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        Jv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        Jv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]
            Jv¹[i, j] = RecursiveOperators.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            Jv²[i, j] = RecursiveOperators.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize

        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁Jv¹ = RecursiveOperators.rmatmul1(D, Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            ∂₂Jv² = RecursiveOperators.rmatmul2(D, Jv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = inv(local_geometry.J) ⊠ (∂₁Jv¹ ⊞ ∂₂Jv²)
        end
    end
    return divflux
end



"""
    slab_weak_divergence!(divflux, flux, mesh)

Compute the right-hand side of the divergence of `flux` weakly, storing the result in `divflux`.

This computes the right-hand side of the variational problem of finding θ such that

    - ⟨ϕ, J θ⟩ = ⟨∂ϕ/∂ξ¹, J u¹⟩ + ⟨∂ϕ/∂ξ², J u²⟩

for all `ϕ` (which arises by integration by parts).

Discretely it is equivalent to

    (D₁' * W * J * u¹ + D₂' * W * J * u²)

where
 - `J` is the diagonal Jacobian matrix
 - `W` is the diagonal matrix of quadrature weights
 - `D₁` and `D₂` are the discrete derivative matrices along the first and second dimensions.
"""
function slab_weak_divergence!(divflux, flux, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]
            WJv¹[i, j] = RecursiveOperators.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            WJv²[i, j] = RecursiveOperators.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize

        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            Dᵀ₁WJv¹ = RecursiveOperators.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            Dᵀ₂WJv² = RecursiveOperators.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
        end
    end
    return divflux
end

function slab_weak_divergence!(
    divflux_slab::DataLayouts.DataSlab2D,
    flux_slab::DataLayouts.DataSlab2D,
    mesh_slab::Meshes.MeshSlab,
)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh_slab)
    D = Quadratures.differentiation_matrix(FT, mesh_slab.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh_slab.quadrature_style)

    # for each element in the element stack
    local_geometry_slab = mesh_slab.local_geometry

    ST = eltype(divflux_slab)
    # Shared on GPU
    WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    for i in 1:Nq, j in 1:Nq
        local_geometry = local_geometry_slab[i, j]
        # compute flux in contravariant coordinates (v¹,v²)
        # alternatively we could do this conversion _after_ taking the derivatives
        # may have an effect on the accuracy
        # materialize if lazy
        F = flux_slab[i, j]
        WJv¹[i, j] = RecursiveOperators.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant1(x, local_geometry),
            F,
        )
        WJv²[i, j] = RecursiveOperators.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    # GPU synchronize

    for i in 1:Nq, j in 1:Nq
        # compute spectral deriv along first dimension
        Dᵀ₁WJv¹ = RecursiveOperators.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
        # compute spectral deriv along second dimension
        Dᵀ₂WJv² = RecursiveOperators.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
        divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
    end
    return divflux_slab
end


# TODO: Next steps
# be able to lookup or dispatch on column mesh,center / faces when doing operations easily
# reformat this as a map across vertical levels with fields (so we don't have to call parent()
# everywhere, get rid of direct indexing)
# CellFace -> CellCenter method for easily getting local operator (could just return a contant)
# we need to make it more generic so that 2nd order is not hard coded (check at lest 1 different order of stencil)
# clean up the interface in general (rmatrixmultiply?), get rid of LinearAlgebra.dot()
# vertical only fields? (split h and v)
# make local_operators static vectors
function vertical_interp!(field_to, field_from, cm::Meshes.ColumnMesh)
    len_to = length(parent(field_to))
    len_from = length(parent(field_from))
    FT = Meshes.undertype(cm)
    if len_from == Meshes.n_cells(cm) && len_to == Meshes.n_faces(cm) # CellCent -> CellFace
        # TODO: should we extrapolate?
        n_faces = Meshes.n_faces(cm)
        for j in Meshes.column(cm, Meshes.CellCent())
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
    elseif len_from == Meshes.n_faces(cm) && len_to == Meshes.n_cells(cm) # CellFace -> CellCent
        local_operator = SVector{2, FT}(0.5, 0.5)
        for i in Meshes.column(cm, Meshes.CellCent())
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

function slab_gradient!(∇field::Field, field::Field)
    @assert Fields.mesh(∇field) === Fields.mesh(field)
    Operators.slab_gradient!(
        Fields.field_values(∇field),
        Fields.field_values(field),
        Fields.mesh(field),
    )
    return ∇field
end
function slab_divergence!(divflux::Field, flux::Field)
    @assert Fields.mesh(divflux) === Fields.mesh(flux)
    Operators.slab_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        Fields.mesh(flux),
    )
    return divflux
end
function slab_weak_divergence!(divflux::Field, flux::Field)
    @assert Fields.mesh(divflux) === Fields.mesh(flux)
    Operators.slab_weak_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        Fields.mesh(flux),
    )
    return divflux
end

function slab_gradient(field::Field)
    S = eltype(field)
    ∇S = RecursiveOperators.rmaptype(T -> Cartesian12Vector{T}, S)
    Operators.slab_gradient!(similar(field, ∇S), field)
end

function slab_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveOperators.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_divergence!(similar(field, divS), field)
end
function slab_weak_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveOperators.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_weak_divergence!(similar(field, divS), field)
end


function interpolate(mesh_to::AbstractMesh, field_from::Field)
    field_to = similar(field_from, (mesh_to,), eltype(field_from))
    interpolate!(field_to, field_from)
end
function interpolate!(field_to::Field, field_from::Field)
    mesh_to = Fields.mesh(field_to)
    mesh_from = Fields.mesh(field_from)
    # @assert mesh_from.topology == mesh_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        mesh_to.quadrature_style,
        mesh_from.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M,
    )
    return field_to
end

function restrict!(field_to::Field, field_from::Field)
    mesh_to = Fields.mesh(field_to)
    mesh_from = Fields.mesh(field_from)
    # @assert mesh_from.topology == mesh_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        mesh_from.quadrature_style,
        mesh_to.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M',
    )
    return field_to
end

function matrix_interpolate(
    field::Field,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    fieldmesh = Fields.mesh(field)
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
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())



function vertical_gradient!(field_to, field_from, cm::Meshes.ColumnMesh)
    len_to = length(parent(field_to))
    len_from = length(parent(field_from))
    FT = Meshes.undertype(cm)
    if len_from == Meshes.n_cells(cm) && len_to == Meshes.n_faces(cm) # CellCent -> CellFace
        # TODO: should we extrapolate?
        n_faces = Meshes.n_faces(cm)
        for j in Meshes.column(cm, Meshes.CellCent())
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
    elseif len_from == Meshes.n_faces(cm) && len_to == Meshes.n_cells(cm) # CellFace -> CellCent
        for i in Meshes.column(cm, Meshes.CellCent())
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
        if len_from == Meshes.n_faces(cm) # face->face gradient
            # TODO: should we extrapolate?
            n_faces = Meshes.n_faces(cm)
            for i in Meshes.column(cm, Meshes.CellFace())
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
        elseif len_from == Meshes.n_cells(cm) # cent->cent gradient
            # TODO: should we extrapolate?
            n_cells = Meshes.n_cells(cm)
            for i in Meshes.column(cm, Meshes.CellCent())
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
    cm::Meshes.ColumnMesh,
    boundary::Meshes.ColumnMinMax,
)
    arr = parent(field)
    len = length(arr)
    if len == Meshes.n_faces(cm)
        boundary_index = Meshes.boundary_index(cm, Meshes.CellFace(), boundary)
        arr[boundary_index] = value
    elseif len == Meshes.n_cells(cm)
        ghost_index = Meshes.ghost_index(cm, Meshes.CellCent(), boundary)
        interior_index = Meshes.interior_index(cm, Meshes.CellCent(), boundary)
        arr[ghost_index] = 2 * value - arr[interior_index]
    else
        error("Bad field")
    end
    return field
end

function apply_neumann!(
    field,
    value,
    cm::Meshes.ColumnMesh,
    boundary::Meshes.ColumnMinMax,
)
    arr = parent(field)
    len = length(arr)
    if len == Meshes.n_faces(cm)
        # ∂_x ϕ = n̂ * (ϕ_g - ϕ_i) / (2Δx) # second order accurate Neumann for CellFace data
        # boundry_index = Meshes.boundary_index(cm, Meshes.CellFace(), boundary)
        n̂ = Meshes.n_hat(boundary)
        ghost_index = Meshes.ghost_index(cm, Meshes.CellFace(), boundary)
        interior_index = Meshes.interior_index(cm, Meshes.CellFace(), boundary)
        Δvert = Meshes.Δcoordinates(cm, Meshes.CellFace(), boundary)
        arr[ghost_index] = arr[interior_index] + n̂ * 2 * Δvert * value
    elseif len == Meshes.n_cells(cm)
        # ∂_x ϕ = n̂ * (ϕ_g - ϕ_i) / Δx # second order accurate Neumann for CellCent data
        n̂ = Meshes.n_hat(boundary)
        ghost_index = Meshes.ghost_index(cm, Meshes.CellCent(), boundary)
        interior_index = Meshes.interior_index(cm, Meshes.CellCent(), boundary)
        Δvert = Meshes.Δcoordinates(cm, Meshes.CellCent(), boundary)
        arr[ghost_index] = arr[interior_index] + n̂ * Δvert * value
    else
        error("Bad field")
    end
    return field
end

include("numericalflux.jl")
include("plots.jl")

end # module
