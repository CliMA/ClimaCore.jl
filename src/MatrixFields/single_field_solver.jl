import ..DataLayouts
dual_type(::Type{A}) where {A} = typeof(Geometry.dual(A.instance))

inv_return_type(::Type{X}) where {X} = error(
    "Cannot solve linear system because a diagonal entry in A contains the \
     non-invertible type $X",
)
inv_return_type(::Type{X}) where {X <: Union{Number, SMatrix}} = X
inv_return_type(::Type{X}) where {T, X <: Geometry.Tensor{2, T}} =
    tensor_type(T, Tuple{dual_type(basis2(X)), dual_type(basis1(X))})

x_eltype(A::ScalingFieldMatrixEntry, b) =
    x_type(eltype(A), eltype(Base.broadcastable(b)))
x_eltype(A::ColumnwiseBandMatrixField, b) =
    x_type(eltype(eltype(A)), eltype(Base.broadcastable(b)))
x_type(::Type{T_A}, ::Type{T_b}) where {T_A, T_b} =
    mul_return_type(inv_return_type(T_A), T_b)

################################################################################

check_single_field_solver(::ScalingFieldMatrixEntry, _) = nothing
function check_single_field_solver(A, b)
    matrix_shape(A) == Square() || error(
        "Cannot solve linear system because a diagonal entry in A is not a \
         square matrix",
    )
    axes(A) === axes(b) || error(
        "Cannot solve linear system because a diagonal entry in A is not on \
         the same space as the corresponding entry in b",
    )
end

single_field_solver_cache(::ScalingFieldMatrixEntry, b) = similar(b, Tuple{})
function single_field_solver_cache(A::ColumnwiseBandMatrixField, b)
    ud = outer_diagonals(eltype(A))[2]
    ud == 0 && return similar(b, Tuple{})
    T_U = mul_return_type(inv_return_type(eltype(eltype(A))), eltype(eltype(A)))
    return similar(b, Tuple{x_eltype(A, b), ntuple(Returns(T_U), Val(ud))...})
end

single_field_solve!(_, x, A::ScalingFieldMatrixEntry, b) =
    x .= (inv(scaling_value(A)),) .* b
single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b) =
    if eltype(A) <: MatrixFields.DiagonalMatrixRow
        A₀ = A.entries.:1
        @. x = inv(A₀) * b
    else
        x_bc = Base.broadcastable(x)
        b_bc = Base.broadcastable(b)
        single_field_solve!(ClimaComms.device(axes(A)), cache, x_bc, A, b_bc)
    end

single_field_solve!(::ClimaComms.AbstractCPUDevice, cache, x, A, b) =
    _single_field_solve!(ClimaComms.device(axes(A)), cache, x, A, b)

# CPU (GPU has already called Spaces.column on arg)
function _single_field_solve!(
    device::ClimaComms.AbstractCPUDevice,
    cache,
    x,
    A,
    b,
)
    space = axes(x)
    mask = Spaces.get_mask(space)
    if space isa Spaces.FiniteDifferenceSpace
        @assert mask isa DataLayouts.NoMask
        single_field_solve_col!(cache, x, A, b)
    else
        Fields.bycolumn(space) do colidx
            I = Fields.universal_index(colidx)
            if DataLayouts.should_compute(mask, I)
                single_field_solve_col!(
                    cache[colidx],
                    x[colidx],
                    A[colidx],
                    b[colidx],
                )
            end
        end
    end
end

single_field_solve_col!(cache, x, A, b) =
    band_matrix_solve!(
        eltype(A),
        unzip_tuple_field_values(Fields.field_values(cache)),
        Fields.field_values(x),
        unzip_tuple_field_values(Fields.field_values(A.entries)),
        Fields.field_values(b),
    )

unzip_tuple_field_values(data) =
    ntuple(i -> data.:($i), Val(length(propertynames(data))))

function band_matrix_solve!(::Type{<:DiagonalMatrixRow}, _, x, Aⱼs, b)
    (A₀,) = Aⱼs
    n = length(x)
    @inbounds for i in 1:n
        x[i] = inv(A₀[i]) * b[i]
    end
end

#=
The Thomas algorithm, as presented in
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Method,
but with the following variable name changes:
    - a → A₋₁
    - b → A₀
    - c → A₊₁
    - d → b
    - c′ → U₊₁
    - d′ → Ux
Transforms the tri-diagonal matrix into a unit upper bi-diagonal matrix, then
solves the resulting system using back substitution. The order of
multiplications has been modified in order to handle block vectors/matrices.
=#
function band_matrix_solve!(::Type{<:TridiagonalMatrixRow}, cache, x, Aⱼs, b)
    A₋₁, A₀, A₊₁ = Aⱼs
    Ux, U₊₁ = cache
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[1])
        U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[1]
        Uxᵢ₋₁ = inv_D₀ * b[1]
        Ux[1] = Uxᵢ₋₁
        U₊₁[1] = U₊₁ᵢ₋₁

        for i in 2:n
            A₋₁ᵢ = A₋₁[i]
            inv_D₀ = inv(A₀[i] - A₋₁ᵢ * U₊₁ᵢ₋₁)
            Uxᵢ₋₁ = inv_D₀ * (b[i] - A₋₁ᵢ * Uxᵢ₋₁)
            Ux[i] = Uxᵢ₋₁
            if i < n
                U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[i] # U₊₁[n] is outside the matrix.
                U₊₁[i] = U₊₁ᵢ₋₁
            end
        end

        x[n] = Ux[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        i = (n - 1)
        # for i in (n - 1):-1:1
        while i ≥ 1
            x[i] = Ux[i] - U₊₁[i] * x[i + 1]
            i -= 1
        end
    end
end

#=
The PTRANS-I algorithm, as presented in
    https://www.hindawi.com/journals/mpe/2015/232456/alg1,
but with the following variable name changes:
    - e → A₋₂
    - c → A₋₁
    - d → A₀
    - a → A₊₁
    - b → A₊₂
    - y → b
    - α → U₊₁
    - β → U₊₂
    - z → Ux
    - γ → L₋₁
    - μ → D₀
Transforms the penta-diagonal matrix into a unit upper tri-diagonal matrix, then
solves the resulting system using back substitution. The order of
multiplications has been modified in order to handle block vectors/matrices.
=#
function band_matrix_solve!(::Type{<:PentadiagonalMatrixRow}, cache, x, Aⱼs, b)
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs
    Ux, U₊₁, U₊₂ = cache
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[1])
        Ux[1] = inv_D₀ * b[1]
        U₊₁[1] = inv_D₀ * A₊₁[1]
        U₊₂[1] = inv_D₀ * A₊₂[1]

        inv_D₀ = inv(A₀[2] - A₋₁[2] * U₊₁[1])
        Ux[2] = inv_D₀ * (b[2] - A₋₁[2] * Ux[1])
        U₊₁[2] = inv_D₀ * (A₊₁[2] - A₋₁[2] * U₊₂[1])
        U₊₂[2] = inv_D₀ * A₊₂[2]

        for i in 3:n
            L₋₁ = A₋₁[i] - A₋₂[i] * U₊₁[i - 2]
            inv_D₀ = inv(A₀[i] - L₋₁ * U₊₁[i - 1] - A₋₂[i] * U₊₂[i - 2])
            Ux[i] = inv_D₀ * (b[i] - L₋₁ * Ux[i - 1] - A₋₂[i] * Ux[i - 2])
            i < n && (U₊₁[i] = inv_D₀ * (A₊₁[i] - L₋₁ * U₊₂[i - 1]))
            i < n - 1 && (U₊₂[i] = inv_D₀ * A₊₂[i])
        end

        x[n] = Ux[n]
        x[n - 1] = Ux[n - 1] - U₊₁[n - 1] * x[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        # for i in (n - 2):-1:1
        i = (n - 2)
        while i ≥ 1
            x[i] = Ux[i] - U₊₁[i] * x[i + 1] - U₊₂[i] * x[i + 2]
            i -= 1
        end
    end
end

#=
Each method for band_matrix_solve! above has an order of operations that is
correct when x, A, and b are block vectors/matrices (i.e., when multiplication
is not necessarily commutative). So, the following are all valid combinations of
eltype(x), eltype(A), and eltype(b):
- Number, Number, and Number
- SVector{N}, SMatrix{N, N}, and SVector{N}
- Tensor{1} with Components B1, Tensor{2} (or its adjoint) with bases (B2, dual(B1)),
  and Tensor{1} with Components B2
- nested type (Tuple or NamedTuple), scalar type (Number, SMatrix, or
  Tensor{2}/adjoint thereof), nested type (Tuple or NamedTuple)

We might eventually want a single general method for band_matrix_solve!, similar
to the BLAS.gbsv function. For now, though, the methods above should be enough.
=#
