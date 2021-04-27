module Operators

import ..slab
import ..Geometry: Cartesian2DVector, Covariant12Vector, Contravariant12Vector
import ..Fields
import ..Meshes
import ..Meshes.Quadratures
import ..Topologies

using StaticArrays

include("rop.jl")

# TODO:
#  - convenience operations for fields
#  - determine output element type
#  - let inputs be Broadcasted objects
#    - make sure that


"""
    volume_gradient!(∇data, data, mesh)

Compute the gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
function volume_gradient!(∇data, data, mesh)
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
            ∂f∂ξ₁ = rmatmul1(D, data_slab, i, j)
            ∂f∂ξ₂ = rmatmul2(D, data_slab, i, j)
            ∂f∂ξ = rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)

            # convert to desired basis
            ∇data_slab[i, j] =
                rmap(x -> Cartesian2DVector(x, local_geometry), ∂f∂ξ)
        end
    end
    return ∇data
end


"""
    volume_divergence!(divflux, flux, mesh)

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
function volume_divergence!(divflux, flux, mesh)
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
            vⁱ = rmap(
                v -> Contravariant12Vector(v, local_geometry),
                flux_slab[i, j],
            ) # materialize if lazy
            # extract each coordinate and multiply by J (Jv¹,Jv²)
            Jv¹[i, j] = rmap(x -> local_geometry.J * x.u¹, vⁱ)
            Jv²[i, j] = rmap(x -> local_geometry.J * x.u², vⁱ)
        end
        # GPU synchronize

        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁Jv¹ = rmatmul1(D, Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            ∂₂Jv² = rmatmul2(D, Jv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = inv(local_geometry.J) ⊠ (∂₁Jv¹ ⊞ ∂₂Jv²)
        end
    end
    return divflux
end

"""
    horizontal_dss!(data, mesh)

Apply direct stiffness summation (DSS) to `data` horizontally.
"""
function horizontal_dss!(data, mesh)
    topology = mesh.topology
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # iterate over the interior faces for each element of the mesh
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(topology)
        # iterate over non-vertex nodes
        for q in 2:(Nq - 1)
            slab1 = slab(data, elem1)
            slab2 = slab(data, elem2)
            i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
            i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
            value = slab1[i1, j1] ⊞ slab2[i2, j2]
            slab1[i1, j1] = slab2[i2, j2] = value
        end
    end

    # iterate over all vertices
    for vertex in Topologies.vertices(topology)
        # gather: compute sum over shared vertices
        sum_data = mapreduce(⊞, vertex) do (elem, vertex_num)
            data_slab = slab(data, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            data_slab[i, j]
        end

        # scatter: assign sum to shared vertices
        for (elem, vertex_num) in vertex
            data_slab = slab(data, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            data_slab[i, j] = sum_data
        end
    end
    return data
end

end # module
