import ..DataLayouts
dual_type(::Type{A}) where {A} = typeof(Geometry.dual(A.instance))

inv_return_type(::Type{X}) where {X} = error(
    "Cannot solve linear system because a diagonal entry in A contains the \
     non-invertible type $X",
)
inv_return_type(::Type{X}) where {X <: Union{Number, SMatrix}} = X
inv_return_type(::Type{X}) where {T, X <: Geometry.Axis2TensorOrAdj{T}} =
    axis_tensor_type(
        T,
        Tuple{dual_type(Geometry.axis2(X)), dual_type(Geometry.axis1(X))},
    )

x_eltype(A::ScalingFieldMatrixEntry, b) = x_eltype(eltype(A), eltype(b))
x_eltype(A::ColumnwiseBandMatrixField, b) =
    x_eltype(eltype(eltype(A)), eltype(b))
x_eltype(::Type{T_A}, ::Type{T_b}) where {T_A, T_b} =
    mul_return_type(inv_return_type(T_A), T_b)

unit_eltype(A::ScalingFieldMatrixEntry) = eltype(A)
unit_eltype(A::ColumnwiseBandMatrixField) =
    unit_eltype(eltype(eltype(A)), local_geometry_type(A))
unit_eltype(::Type{T_A}, ::Type{LG}) where {T_A, LG} =
    mul_return_type(inv_return_type(T_A), T_A)

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
    cache_eltype =
        ud == 0 ? Tuple{} :
        Tuple{x_eltype(A, b), ntuple(_ -> unit_eltype(A), Val(ud))...}
    return similar(b, cache_eltype)
end

function single_field_solve_diag_matrix_row!(
    cache,
    x,
    A::ColumnwiseBandMatrixField,
    b,
)
    # Use fields here, and not field values, so that this operation is
    # mask-aware.
    A₀ = A.entries.:1
    @. x = inv(A₀) * b
end
single_field_solve!(_, x, A::ScalingFieldMatrixEntry, b) =
    x .= (inv(scaling_value(A)),) .* b
function single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b)
    if eltype(A) <: MatrixFields.DiagonalMatrixRow
        single_field_solve_diag_matrix_row!(cache, x, A, b)
    else
        single_field_solve!(ClimaComms.device(axes(A)), cache, x, A, b)
    end
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
        _single_field_solve_col!(device, cache, x, A, b)
    else
        Fields.bycolumn(space) do colidx
            I = Fields.universal_index(colidx)
            if DataLayouts.should_compute(mask, I)
                _single_field_solve_col!(
                    device,
                    cache[colidx],
                    x[colidx],
                    A[colidx],
                    b[colidx],
                )
            end
        end
    end
end

function _single_field_solve_col!(
    ::ClimaComms.AbstractCPUDevice,
    cache,
    x,
    A,
    b,
)
    if A isa Fields.ColumnField
        band_matrix_solve!(
            eltype(A),
            unzip_tuple_field_values(Fields.field_values(cache)),
            Fields.field_values(x),
            unzip_tuple_field_values(Fields.field_values(A.entries)),
            Fields.field_values(b),
            vindex,
        )
    elseif A isa ScalingFieldMatrixEntry
        x .= (inv(scaling_value(A)),) .* b
    else
        error("uncaught case")
    end
end

unzip_tuple_field_values(data) =
    ntuple(i -> data.:($i), Val(length(propertynames(data))))

function band_matrix_solve!(::Type{<:DiagonalMatrixRow}, _, x, Aⱼs, b, vi)
    (A₀,) = Aⱼs
    n = length(x)
    @inbounds for i in 1:n
        x[vi(i)] = inv(A₀[vi(i)]) * b[vi(i)]
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
function band_matrix_solve!(
    ::Type{<:TridiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
    vi,
)
    A₋₁, A₀, A₊₁ = Aⱼs
    Ux, U₊₁ = cache
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[vi(1)])
        U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[vi(1)]
        Uxᵢ₋₁ = inv_D₀ * b[vi(1)]
        Ux[vi(1)] = Uxᵢ₋₁
        U₊₁[vi(1)] = U₊₁ᵢ₋₁

        for i in 2:n
            A₋₁ᵢ = A₋₁[vi(i)]
            inv_D₀ = inv(A₀[vi(i)] - A₋₁ᵢ * U₊₁ᵢ₋₁)
            Uxᵢ₋₁ = inv_D₀ * (b[vi(i)] - A₋₁ᵢ * Uxᵢ₋₁)
            Ux[vi(i)] = Uxᵢ₋₁
            if i < n
                U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[vi(i)] # U₊₁[n] is outside the matrix.
                U₊₁[vi(i)] = U₊₁ᵢ₋₁
            end
        end

        x[vi(n)] = Ux[vi(n)]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        i = (n - 1)
        # for i in (n - 1):-1:1
        while i ≥ 1
            x[vi(i)] = Ux[vi(i)] - U₊₁[vi(i)] * x[vi(i + 1)]
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
function band_matrix_solve!(
    ::Type{<:PentadiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
    vi,
)
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs
    Ux, U₊₁, U₊₂ = cache
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[vi(1)])
        Ux[vi(1)] = inv_D₀ * b[vi(1)]
        U₊₁[vi(1)] = inv_D₀ * A₊₁[vi(1)]
        U₊₂[vi(1)] = inv_D₀ * A₊₂[vi(1)]

        inv_D₀ = inv(A₀[vi(2)] - A₋₁[vi(2)] * U₊₁[vi(1)])
        Ux[vi(2)] = inv_D₀ * (b[vi(2)] - A₋₁[vi(2)] * Ux[vi(1)])
        U₊₁[vi(2)] = inv_D₀ * (A₊₁[vi(2)] - A₋₁[vi(2)] * U₊₂[vi(1)])
        U₊₂[vi(2)] = inv_D₀ * A₊₂[vi(2)]

        for i in 3:n
            L₋₁ = A₋₁[vi(i)] - A₋₂[vi(i)] * U₊₁[vi(i - 2)]
            inv_D₀ = inv(
                A₀[vi(i)] - L₋₁ * U₊₁[vi(i - 1)] - A₋₂[vi(i)] * U₊₂[vi(i - 2)],
            )
            Ux[vi(i)] =
                inv_D₀ *
                (b[vi(i)] - L₋₁ * Ux[vi(i - 1)] - A₋₂[vi(i)] * Ux[vi(i - 2)])
            i < n && (U₊₁[vi(i)] = inv_D₀ * (A₊₁[vi(i)] - L₋₁ * U₊₂[vi(i - 1)]))
            i < n - 1 && (U₊₂[vi(i)] = inv_D₀ * A₊₂[vi(i)])
        end

        x[vi(n)] = Ux[vi(n)]
        x[vi(n - 1)] = Ux[vi(n - 1)] - U₊₁[vi(n - 1)] * x[vi(n)]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        # for i in (n - 2):-1:1
        i = (n - 2)
        while i ≥ 1
            x[vi(i)] =
                Ux[vi(i)] - U₊₁[vi(i)] * x[vi(i + 1)] -
                U₊₂[vi(i)] * x[vi(i + 2)]
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
- AxisVector with axis A1, Axis2TensorOrAdj with axes (A2, dual(A1)), and
  AxisVector with axis A2

We might eventually want a single general method for band_matrix_solve!, similar
to the BLAS.gbsv function. For now, though, the methods above should be enough.
=#
