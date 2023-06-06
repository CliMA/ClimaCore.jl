"""
    compute_christoffel_symbols(gⁱʲ, gᵢⱼ)
2D spectral element configuration. Compute Christoffel symbols 
given metric tensor `g` (contravariant and covariant). 
The underlying topology can be inferred from `g`. 
ClimaCore datastructures store information in the 
`VIJFH` layout, assuming 3D systems - thus the Christoffel symbol 
calculation amounts to populating the `F` entry in this layout with
column-major indexing. 
TODO: Eval in additional unit tests and demonstrate operations on 
Fields with these changes.
"""
function compute_christoffel_symbols(gⁱʲ, gᵢⱼ)
    n = 2
    FT = eltype(parent(gⁱʲ))
    Γ = zeros(FT, (n, n, n))
    grad = Operators.Gradient()
    gⁱʲ_field = Fields.Field(gⁱʲ, space)
    gᵢⱼ_field = Fields.Field(gᵢⱼ, space)
    Nij, Nij, Nf, Nh = size(parent(gᵢⱼ))
    ∇gⁱʲ_field = @. grad(gⁱʲ_field.components.data)
    ∇gᵢⱼ_field = @. grad(gᵢⱼ_field.components.data)
    Γⁱⱼₖ = Fields.Field(
        IJFH{NTuple{n^3, FT}, Nij}(zeros(Nij, Nij, n^3, Nh)),
        space,
    )
    for i in 1:n, j in 1:n, k in 1:n
        Σᵢⱼₖ = zero(gⁱʲ_field).components.data.:1
        for l in 1:n
            il = linear_index(i, l)
            lj = linear_index(l, j)
            lk = linear_index(l, k)
            jk = linear_index(j, k)
            Σᵢⱼₖ .+=
                1 / 2 .* gⁱʲ_field.components.data.:($il) .* (
                    ∇gᵢⱼ_field.:($lj).components.data.:($k) .+
                    ∇gᵢⱼ_field.:($lk).components.data.:($j) .-
                    ∇gᵢⱼ_field.:($jk).components.data.:($l)
                )
        end
        ijk = linear_index(i, j, k, nrows = 2, ncols = 2)
        Γⁱⱼₖ.:($ijk) .= Σᵢⱼₖ
        # Test symmetry w.r.t lower indices Γⁱⱼₖ == Γⁱₖⱼ 
        # Convert to column major indices and assert requirement.
        @assert Γⁱⱼₖ.:($3) == Γⁱⱼₖ.:($5)
        @assert Γⁱⱼₖ.:($4) == Γⁱⱼₖ.:($6)
    end
    return Γⁱⱼₖ
end

"""
    linear_index(i,j,k; nrow=2, ncol=2)
Compute the linear index from a 3 term indexing system, for a system that is 
represented in 2 dimensions (nrow=2, ncol=2). Modifying the appropriate
keyword arguments for nrow and ncol can provide the appropriate linear_index
in a 3 dimensional system.  
"""
function linear_index(i, j; nrows = 2)
    return (j - 1) * nrows + i
end
function linear_index(i, j, k; nrows = 2, ncols = 2)
    return (k - 1) * (nrows * ncols) + (j - 1) * nrows + i
end
