
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
            temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
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
        temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
    end
    for j in 1:Nij_out, i in 1:Nij_out
        out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
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
    slab_gradient!(∇data, data, space)

Compute the gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
function slab_gradient!(∇data, data, space)
    # all derivatives calculated in the reference local geometry FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(data)
    for h in 1:Nh
        ∇data_slab = slab(∇data, h)
        # TODO: can we just call materialize(slab(data,h))
        data_slab = slab(data, h)
        local_geometry_slab = slab(space.local_geometry, h)

        for i in 1:Nq, j in 1:Nq
            # TODO: materialize data_slab once
            # on GPU this would be done to shared memory, then synchronize threads
            local_geometry = local_geometry_slab[i, j]

            # compute covariant derivatives
            ∂f∂ξ₁ = RecursiveApply.rmatmul1(D, data_slab, i, j)
            ∂f∂ξ₂ = RecursiveApply.rmatmul2(D, data_slab, i, j)
            ∂f∂ξ = RecursiveApply.rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)

            # convert to desired basis
            ∇data_slab[i, j] = RecursiveApply.rmap(
                x -> Cartesian12Vector(x, local_geometry),
                ∂f∂ξ,
            )
        end
    end
    return ∇data
end


"""
    slab_divergence!(divflux, flux, space)

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
function slab_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

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
            Jv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            Jv²[i, j] = RecursiveApply.rmap(
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
            ∂₁Jv¹ = RecursiveApply.rmatmul1(D, Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            ∂₂Jv² = RecursiveApply.rmatmul2(D, Jv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = inv(local_geometry.J) ⊠ (∂₁Jv¹ ⊞ ∂₂Jv²)
        end
    end
    return divflux
end



"""
    slab_weak_divergence!(divflux, flux, space)

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
function slab_weak_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

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
            WJv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            WJv²[i, j] = RecursiveApply.rmap(
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
            Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
        end
    end
    return divflux
end

function slab_weak_divergence!(
    divflux_slab::DataLayouts.DataSlab2D,
    flux_slab::DataLayouts.DataSlab2D,
    space_slab::Spaces.SpectralElementSpaceSlab,
)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space_slab)
    D = Quadratures.differentiation_matrix(FT, space_slab.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space_slab.quadrature_style)

    # for each element in the element stack
    local_geometry_slab = space_slab.local_geometry

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
        WJv¹[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant1(x, local_geometry),
            F,
        )
        WJv²[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    # GPU synchronize

    for i in 1:Nq, j in 1:Nq
        # compute spectral deriv along first dimension
        Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
        # compute spectral deriv along second dimension
        Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
        divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
    end
    return divflux_slab
end





function slab_gradient!(∇field::Field, field::Field)
    @assert Fields.space(∇field) === Fields.space(field)
    Operators.slab_gradient!(
        Fields.field_values(∇field),
        Fields.field_values(field),
        Fields.space(field),
    )
    return ∇field
end
function slab_divergence!(divflux::Field, flux::Field)
    @assert Fields.space(divflux) === Fields.space(flux)
    Operators.slab_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        Fields.space(flux),
    )
    return divflux
end
function slab_weak_divergence!(divflux::Field, flux::Field)
    @assert Fields.space(divflux) === Fields.space(flux)
    Operators.slab_weak_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        Fields.space(flux),
    )
    return divflux
end

function slab_gradient(field::Field)
    S = eltype(field)
    ∇S = RecursiveApply.rmaptype(T -> Cartesian12Vector{T}, S)
    Operators.slab_gradient!(similar(field, ∇S), field)
end

function slab_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_divergence!(similar(field, divS), field)
end
function slab_weak_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_weak_divergence!(similar(field, divS), field)
end



function interpolate(space_to::AbstractSpace, field_from::Field)
    field_to = similar(field_from, (space_to,), eltype(field_from))
    interpolate!(field_to, field_from)
end
function interpolate!(field_to::Field, field_from::Field)
    space_to = Fields.space(field_to)
    space_from = Fields.space(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_to.quadrature_style,
        space_from.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M,
    )
    return field_to
end

function restrict!(field_to::Field, field_from::Field)
    space_to = Fields.space(field_to)
    space_from = Fields.space(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_from.quadrature_style,
        space_to.quadrature_style,
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
    space = Fields.space(field)
    mesh = space.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2

    interp_data =
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))

    M = Quadratures.interpolation_matrix(
        Float64,
        Q_interp,
        space.quadrature_style,
    )
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())
