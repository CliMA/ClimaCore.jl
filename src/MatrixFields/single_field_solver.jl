dual_type(::Type{A}) where {A} = typeof(Geometry.dual(A.instance))

inv_return_type(::Type{X}) where {X} = error(
    "Cannot solve linear system because a diagonal entry in A contains the \
     non-invertible type $X",
)
inv_return_type(::Type{X}) where {X <: Union{Number, SMatrix}} = X
inv_return_type(::Type{X}) where {T, X <: Geometry.Axis2TensorOrAdj{T}} =
    axis_tensor_type(T, Tuple{dual_type(axis2(X)), dual_type(axis1(X))})

x_eltype(A::UniformScaling, b) = x_eltype(eltype(A), eltype(b))
x_eltype(A::ColumnwiseBandMatrixField, b) =
    x_eltype(eltype(eltype(A)), eltype(b))
x_eltype(::Type{T_A}, ::Type{T_b}) where {T_A, T_b} =
    rmul_return_type(inv_return_type(T_A), T_b)

unit_eltype(A::UniformScaling) = unit_eltype(eltype(A))
unit_eltype(A::ColumnwiseBandMatrixField) = unit_eltype(eltype(eltype(A)))
unit_eltype(::Type{T_A}) where {T_A} =
    rmul_return_type(inv_return_type(T_A), T_A)

################################################################################

check_single_field_solver(::UniformScaling, _) = nothing
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

single_field_solver_cache(::UniformScaling, b) = similar(b, Tuple{})
function single_field_solver_cache(A::ColumnwiseBandMatrixField, b)
    ud = outer_diagonals(eltype(A))[2]
    cache_eltype =
        ud == 0 ? Tuple{} :
        Tuple{x_eltype(A, b), ntuple(_ -> unit_eltype(A), Val(ud))...}
    return similar(b, cache_eltype)
end

single_field_solve!(_, x, A::UniformScaling, b) = x .= inv(A.λ) .* b
single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b) =
    single_field_solve!(ClimaComms.device(axes(A)), cache, x, A, b)

single_field_solve!(::ClimaComms.AbstractCPUDevice, cache, x, A, b) =
    _single_field_solve!(ClimaComms.device(axes(A)), cache, x, A, b)

# single_field_solve!(::ClimaComms.CUDADevice, ...) is no longer exercised,
# but it may be helpful for debugging, due to its simplicity. So, let's leave
# it here for now.
function single_field_solve!(::ClimaComms.CUDADevice, cache, x, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    nthreads, nblocks = Topologies._configure_threadblock(Ni * Nj * Nh)
    device = ClimaComms.device(A)
    CUDA.@cuda always_inline = true threads = nthreads blocks = nblocks single_field_solve_kernel!(
        device,
        cache,
        x,
        A,
        b,
    )
end

function single_field_solve_kernel!(device, cache, x, A, b)
    idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        _single_field_solve!(
            device,
            Spaces.column(cache, i, j, h),
            Spaces.column(x, i, j, h),
            Spaces.column(A, i, j, h),
            Spaces.column(b, i, j, h),
        )
    end
    return nothing
end
single_field_solve_kernel!(
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::Fields.ColumnField,
    b::Fields.ColumnField,
) = _single_field_solve!(cache, x, A, b)

# CPU (GPU has already called Spaces.column on arg)
_single_field_solve!(device::ClimaComms.AbstractCPUDevice, cache, x, A, b) =
    Fields.bycolumn(axes(A)) do colidx
        _single_field_solve_col!(
            ClimaComms.device(axes(A)),
            cache[colidx],
            x[colidx],
            A[colidx],
            b[colidx],
        )
    end

function _single_field_solve_col!(
    ::ClimaComms.AbstractCPUDevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A,
    b::Fields.ColumnField,
)
    if A isa Fields.ColumnField
        band_matrix_solve!(
            eltype(A),
            unzip_tuple_field_values(Fields.field_values(cache)),
            Fields.field_values(x),
            unzip_tuple_field_values(Fields.field_values(A.entries)),
            Fields.field_values(b),
        )
    elseif A isa UniformScaling
        x .= inv(A.λ) .* b
    else
        error("uncaught case")
    end
end

# called by TuplesOfNTuples.jl's `inner_dispatch`:
# which requires a particular argument order:
_single_field_solve!(
    cache::Fields.Field,
    x::Fields.Field,
    A::Union{Fields.Field, UniformScaling},
    b::Fields.Field,
    dev::ClimaComms.CUDADevice,
) = _single_field_solve!(dev, cache, x, A, b)

_single_field_solve!(
    cache::Fields.Field,
    x::Fields.Field,
    A::Union{Fields.Field, UniformScaling},
    b::Fields.Field,
    dev::ClimaComms.AbstractCPUDevice,
) = _single_field_solve_col!(dev, cache, x, A, b)

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::Fields.ColumnField,
    b::Fields.ColumnField,
)
    band_matrix_solve!(
        eltype(A),
        unzip_tuple_field_values(Fields.field_values(cache)),
        Fields.field_values(x),
        unzip_tuple_field_values(Fields.field_values(A.entries)),
        Fields.field_values(b),
    )
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::UniformScaling,
    b::Fields.ColumnField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    n = length(x_data)
    @inbounds for i in 1:n
        x_data[i] = inv(A.λ) ⊠ b_data[i]
    end
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.PointDataField,
    x::Fields.PointDataField,
    A::UniformScaling,
    b::Fields.PointDataField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    n = length(x_data)
    @inbounds begin
        x_data[] = inv(A.λ) ⊠ b_data[]
    end
end

unzip_tuple_field_values(data) =
    ntuple(i -> data.:($i), Val(length(propertynames(data))))

function band_matrix_solve!(::Type{<:DiagonalMatrixRow}, _, x, Aⱼs, b)
    (A₀,) = Aⱼs
    n = length(x)
    @inbounds for i in 1:n
        x[i] = inv(A₀[i]) ⊠ b[i]
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
        Ux[1] = inv_D₀ ⊠ b[1]
        U₊₁[1] = inv_D₀ ⊠ A₊₁[1]

        for i in 2:n
            inv_D₀ = inv(A₀[i] ⊟ A₋₁[i] ⊠ U₊₁[i - 1])
            Ux[i] = inv_D₀ ⊠ (b[i] ⊟ A₋₁[i] ⊠ Ux[i - 1])
            i < n && (U₊₁[i] = inv_D₀ ⊠ A₊₁[i]) # U₊₁[n] is outside the matrix.
        end

        x[n] = Ux[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        i = (n - 1)
        # for i in (n - 1):-1:1
        while i ≥ 1
            x[i] = Ux[i] ⊟ U₊₁[i] ⊠ x[i + 1]
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
        Ux[1] = inv_D₀ ⊠ b[1]
        U₊₁[1] = inv_D₀ ⊠ A₊₁[1]
        U₊₂[1] = inv_D₀ ⊠ A₊₂[1]

        inv_D₀ = inv(A₀[2] ⊟ A₋₁[2] ⊠ U₊₁[1])
        Ux[2] = inv_D₀ ⊠ (b[2] ⊟ A₋₁[2] ⊠ Ux[1])
        U₊₁[2] = inv_D₀ ⊠ (A₊₁[2] ⊟ A₋₁[2] ⊠ U₊₂[1])
        U₊₂[2] = inv_D₀ ⊠ A₊₂[2]

        for i in 3:n
            L₋₁ = A₋₁[i] ⊟ A₋₂[i] ⊠ U₊₁[i - 2]
            inv_D₀ = inv(A₀[i] ⊟ L₋₁ ⊠ U₊₁[i - 1] ⊟ A₋₂[i] ⊠ U₊₂[i - 2])
            Ux[i] = inv_D₀ ⊠ (b[i] ⊟ L₋₁ ⊠ Ux[i - 1] ⊟ A₋₂[i] ⊠ Ux[i - 2])
            i < n && (U₊₁[i] = inv_D₀ ⊠ (A₊₁[i] ⊟ L₋₁ ⊠ U₊₂[i - 1]))
            i < n - 1 && (U₊₂[i] = inv_D₀ ⊠ A₊₂[i])
        end

        x[n] = Ux[n]
        x[n - 1] = Ux[n - 1] ⊟ U₊₁[n - 1] ⊠ x[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        # for i in (n - 2):-1:1
        i = (n - 2)
        while i ≥ 1
            x[i] = Ux[i] ⊟ U₊₁[i] ⊠ x[i + 1] ⊟ U₊₂[i] ⊠ x[i + 2]
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
- nested type (Tuple or NamedTuple), scalar type (Number, SMatrix, or
  Axis2TensorOrAdj), nested type (Tuple or NamedTuple)

We might eventually want a single general method for band_matrix_solve!, similar
to the BLAS.gbsv function. For now, though, the methods above should be enough.
=#
