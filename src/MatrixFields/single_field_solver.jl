import ..DataLayouts

# Each value of x is a product of a value of b with the inverse of a value of A.
inv_mul(a, b) = inv(a) * b
x_eltype(A::ScalingFieldMatrixEntry, b) =
    x_type(eltype(A), eltype(Base.broadcastable(b)))
x_eltype(A::ColumnwiseBandMatrixField, b) =
    x_type(eltype(eltype(A)), eltype(Base.broadcastable(b)))
x_type(::Type{T_A}, ::Type{T_b}) where {T_A, T_b} = return_type(inv_mul, Tuple{T_A, T_b})

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
    T_U = x_type(eltype(eltype(A)), eltype(eltype(A)))
    return similar(b, Tuple{x_eltype(A, b), ntuple(Returns(T_U), Val(ud))...})
end

single_field_solve!(x, A, b) = single_field_solve!(nothing, x, A, b)
single_field_solve!(_, x, A::ScalingFieldMatrixEntry, b) =
    x .= (inv(scaling_value(A)),) .* b
function single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b)
    if eltype(A) <: DiagonalMatrixRow
        A₀ = A.entries.:1
        @. x = inv(A₀) * b
    else
        x_bc = Base.broadcastable(x)
        b_bc = Base.broadcastable(b)
        scope = DataLayouts.DataScope(x_bc, A.entries, b_bc)
        if !DataLayouts.needs_loop_setup(scope)
            single_field_solve_slice!(scope, cache, x_bc, A, b_bc)
        else
            cache_arg = isnothing(cache) ? single_field_solver_cache(A, x_bc) : cache
            single_field_solve!(ClimaComms.device(axes(A)), cache_arg, x_bc, A, b_bc)
        end
    end
    return nothing
end

# Band-matrix linear solver for a single column inside a foreach_column slice
# loop. On a single-thread scope (CPU or GPU with shmem = Val(:none)), the
# direct Thomas/PTRANS-I algorithm runs in thread-local storage. On a
# multi-thread sub-block, a tridiagonal system with at most one level per thread
# whose 4 column buffers fit in the sub-block's shared memory is solved across
# threads by parallel cyclic reduction, and systems with more levels, less
# shared memory, or pentadiagonal bands stage register-resident or lazy inputs
# through shared buffers and solve on thread 1.
@inline function single_field_solve_slice!(scope, cache, x, A, b)
    b_applied = Operators.apply_operators(DataLayouts.reassign(b, scope))
    N = DataLayouts.static_num_threads(scope)
    if isnothing(N) || N == 1
        b_col =
            b_applied isa Fields.Field ? b_applied :
            Operators.register_values(
                DataLayouts.reassign(b_applied, DataLayouts.ThisThread()),
            )
        if !isnothing(cache)
            x_col = DataLayouts.reassign(x, DataLayouts.ThisThread())
            cache_col = DataLayouts.reassign(cache, DataLayouts.ThisThread())
            single_field_solve_col!(cache_col, x_col, A, b_col)
        else
            x_data_thread =
                DataLayouts.reassign(Fields.field_values(x), DataLayouts.ThisThread())
            ud = outer_diagonals(eltype(A))[2]
            T_U = x_type(eltype(eltype(A)), eltype(eltype(A)))
            cache_tuple =
                ud == 0 ? () :
                (
                    DataLayouts.register_similar(x_data_thread, eltype(x_data_thread)),
                    ntuple(
                        _ -> DataLayouts.register_similar(x_data_thread, T_U),
                        Val(ud),
                    )...,
                )
            band_matrix_solve!(
                eltype(A),
                cache_tuple,
                x_data_thread,
                unzip_tuple_field_values(
                    DataLayouts.reassign(
                        Fields.field_values(A.entries),
                        DataLayouts.ThisThread(),
                    ),
                ),
                DataLayouts.reassign(
                    Fields.field_values(b_col),
                    DataLayouts.ThisThread(),
                ),
            )
        end
        return nothing
    end
    x_data = DataLayouts.reassign(Fields.field_values(x), scope)
    Nv = DataLayouts.nlevels(x_data)
    if eltype(A) <: TridiagonalMatrixRow && Nv <= N
        T_A = eltype(eltype(A))
        T_x = eltype(x_data)
        B_A = DataLayouts.aligned_buffer_bytes(Nv * sizeof(T_A))
        B_x = DataLayouts.aligned_buffer_bytes(Nv * sizeof(T_x))
        if 3 * B_A + B_x <= DataLayouts.total_buffer_limit(scope)
            tridiag_pcr_subblock!(scope, x_data, A, b_applied)
            return nothing
        end
    end
    band_matrix_solve_subblock!(scope, x_data, A, b_applied)
    return nothing
end

@inline function tridiag_pcr_subblock!(scope, x_data, A, b_applied)
    Nv = DataLayouts.nlevels(x_data)
    A_entries = DataLayouts.reassign(Fields.field_values(A.entries), scope)
    A₋₁, A₀, A₊₁ = unzip_tuple_field_values(A_entries)
    b_field =
        b_applied isa Fields.Field ? DataLayouts.reassign(b_applied, scope) :
        Operators.register_values(DataLayouts.reassign(b_applied, scope))
    b_data = Fields.field_values(b_field)
    T_A = eltype(A₀)
    T_x = eltype(x_data)
    B_A = DataLayouts.aligned_buffer_bytes(Nv * sizeof(T_A))
    s_a = DataLayouts.buffer_similar(x_data, T_A, Val(0))
    s_b = DataLayouts.buffer_similar(x_data, T_A, Val(B_A))
    s_c = DataLayouts.buffer_similar(x_data, T_A, Val(2 * B_A))
    s_d = DataLayouts.buffer_similar(x_data, T_x, Val(3 * B_A))
    i = Int(DataLayouts.thread_rank(scope))
    zero_A = zero(T_A)
    zero_x = zero(T_x)
    @inbounds begin
        local_ai = i <= Nv ? A₋₁[CartesianIndex(i, 1, 1, 1)] : zero_A
        local_bi = i <= Nv ? A₀[CartesianIndex(i, 1, 1, 1)] : one(T_A)
        local_ci = i <= Nv ? A₊₁[CartesianIndex(i, 1, 1, 1)] : zero_A
        local_di = i <= Nv ? b_data[CartesianIndex(i, 1, 1, 1)] : zero_x
        if i <= Nv
            s_a[CartesianIndex(i, 1, 1, 1)] = local_ai
            s_b[CartesianIndex(i, 1, 1, 1)] = local_bi
            s_c[CartesianIndex(i, 1, 1, 1)] = local_ci
            s_d[CartesianIndex(i, 1, 1, 1)] = local_di
        end
    end
    DataLayouts.synchronize(scope)
    stride = 1
    while stride < Nv
        if i <= Nv
            i_minus = max(i - stride, 1)
            i_plus = min(i + stride, Nv)
            @inbounds begin
                k1 =
                    i > stride ?
                    -local_ai * inv(s_b[CartesianIndex(i_minus, 1, 1, 1)]) : zero_A
                k2 =
                    i <= Nv - stride ?
                    -local_ci * inv(s_b[CartesianIndex(i_plus, 1, 1, 1)]) : zero_A
                local_ai = k1 * s_a[CartesianIndex(i_minus, 1, 1, 1)]
                local_bi =
                    local_bi +
                    k1 * s_c[CartesianIndex(i_minus, 1, 1, 1)] +
                    k2 * s_a[CartesianIndex(i_plus, 1, 1, 1)]
                local_ci = k2 * s_c[CartesianIndex(i_plus, 1, 1, 1)]
                local_di =
                    local_di +
                    k1 * s_d[CartesianIndex(i_minus, 1, 1, 1)] +
                    k2 * s_d[CartesianIndex(i_plus, 1, 1, 1)]
            end
        end
        DataLayouts.synchronize(scope)
        if i <= Nv
            @inbounds begin
                s_a[CartesianIndex(i, 1, 1, 1)] = local_ai
                s_b[CartesianIndex(i, 1, 1, 1)] = local_bi
                s_c[CartesianIndex(i, 1, 1, 1)] = local_ci
                s_d[CartesianIndex(i, 1, 1, 1)] = local_di
            end
        end
        DataLayouts.synchronize(scope)
        stride *= 2
    end
    if i <= Nv
        @inbounds x_data[CartesianIndex(i, 1, 1, 1)] = inv(local_bi) * local_di
    end
    !DataLayouts.stored_in_registers(x_data) && DataLayouts.synchronize(scope)
    return nothing
end

@inline function band_matrix_solve_subblock!(scope, x_data, A, b_applied)
    A_entries = DataLayouts.reassign(A.entries, scope)
    A_buf = Operators.materialize_buffer(A_entries, Val(0))
    offset_b = Operators.shared_buffer_bytes(A_buf)
    b_buf = Operators.materialize_buffer(
        DataLayouts.reassign(b_applied, scope),
        offset_b,
    )
    offset_cache = Operators.add_bytes(offset_b, Operators.shared_buffer_bytes(b_buf))
    ud = outer_diagonals(eltype(A))[2]
    T_x = eltype(x_data)
    T_U = x_type(eltype(eltype(A)), eltype(eltype(A)))
    Nv = DataLayouts.nlevels(x_data)
    B_x = DataLayouts.aligned_buffer_bytes(Nv * sizeof(T_x))
    B_U = DataLayouts.aligned_buffer_bytes(Nv * sizeof(T_U))
    cache_tuple =
        ud == 0 ? () :
        (
            DataLayouts.buffer_similar(x_data, T_x, offset_cache),
            ntuple(
                k -> DataLayouts.buffer_similar(
                    x_data,
                    T_U,
                    Operators.add_bytes(offset_cache, Val(B_x + (k - 1) * B_U)),
                ),
                Val(ud),
            )...,
        )
    offset_x =
        ud == 0 ? offset_cache :
        Operators.add_bytes(offset_cache, Val(B_x + ud * B_U))
    x_buf =
        DataLayouts.stored_in_registers(x_data) ?
        DataLayouts.buffer_similar(x_data, T_x, offset_x) : x_data
    DataLayouts.synchronize(scope)
    if DataLayouts.thread_rank(scope) == 1
        band_matrix_solve!(
            eltype(A),
            cache_tuple,
            x_buf,
            unzip_tuple_field_values(Fields.field_values(A_buf)),
            Fields.field_values(b_buf),
        )
    end
    DataLayouts.synchronize(scope)
    if DataLayouts.stored_in_registers(x_data)
        @inbounds for v in 1:Nv
            if DataLayouts.owns_column_level(scope, v)
                x_data[CartesianIndex(v, 1, 1, 1)] = x_buf[CartesianIndex(v, 1, 1, 1)]
            end
        end
    end
    return nothing
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

@inline single_field_solve_col!(cache, x, A, b) =
    band_matrix_solve!(
        eltype(A),
        unzip_tuple_field_values(Fields.field_values(cache)),
        Fields.field_values(x),
        unzip_tuple_field_values(Fields.field_values(A.entries)),
        Fields.field_values(b),
    )

# Generate the tuple directly instead of calling ntuple, since the views made
# by getproperty are only free of allocations when they do not escape through
# the non-inlined ntuple closure.
@generated function unzip_tuple_field_values(data)
    property_exprs =
        (:(getproperty(data, $i)) for i in 1:fieldcount(eltype(data)))
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, property_exprs...))
end

@inline function band_matrix_solve!(::Type{<:DiagonalMatrixRow}, _, x, Aⱼs, b)
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
@inline function band_matrix_solve!(::Type{<:TridiagonalMatrixRow}, cache, x, Aⱼs, b)
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
@inline function band_matrix_solve!(::Type{<:PentadiagonalMatrixRow}, cache, x, Aⱼs, b)
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
