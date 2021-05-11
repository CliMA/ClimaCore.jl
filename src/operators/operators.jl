module Operators

import ..slab
import ..DataLayouts: Data2D, DataSlab
import ..DataLayouts
import ..Geometry
import ..Geometry: Cartesian12Vector, Covariant12Vector, Contravariant12Vector
import ..Meshes
import ..Meshes.Quadratures
import ..Topologies
using ..RecursiveOperators

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
    out_slab::DataLayouts.DataSlab{S, Nij_out},
    in_slab::DataLayouts.DataSlab{S, Nij_in},
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
    divflux_slab::DataLayouts.DataSlab,
    flux_slab::DataLayouts.DataSlab,
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


end # module
