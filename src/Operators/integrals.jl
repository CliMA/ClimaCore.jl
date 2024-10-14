import ..RecursiveApply: rzero, ⊠, ⊞
import RootSolvers
import ClimaComms

"""
    column_integral_definite!(ϕ_top, ᶜ∂ϕ∂z, [ϕ_bot])

Sets `ϕ_top```{}= \\int_{z_{bot}}^{z_{top}}\\,```ᶜ∂ϕ∂z```(z)\\,dz +{}```ϕ_bot`,
where ``z_{bot}`` and ``z_{top}`` are the values of `z` at the bottom and top of
the domain, respectively. The input `ᶜ∂ϕ∂z` should be a cell-center `Field` or
`AbstractBroadcasted`, and the output `ϕ_top` should be a horizontal `Field`.
The default value of `ϕ_bot` is 0.
"""
function column_integral_definite!(ϕ_top, ᶜ∂ϕ∂z, ϕ_bot = rzero(eltype(ϕ_top)))
    ᶜΔϕ = Base.Broadcast.broadcasted(⊠, ᶜ∂ϕ∂z, Fields.Δz_field(axes(ᶜ∂ϕ∂z)))
    column_reduce!(⊞, ϕ_top, ᶜΔϕ; init = ϕ_bot)
end

"""
    column_integral_indefinite!(ᶠϕ, ᶜ∂ϕ∂z, [ϕ_bot])

Sets `ᶠϕ```(z) = \\int_{z_{bot}}^z\\,```ᶜ∂ϕ∂z```(z')\\,dz' +{}```ϕ_bot`, where
``z_{bot}`` is the value of `z` at the bottom of the domain. The input `ᶜ∂ϕ∂z`
should be a cell-center `Field` or `AbstractBroadcasted`, and the output `ᶠϕ`
should be a cell-face `Field`. The default value of `ϕ_bot` is 0.

    column_integral_indefinite!(∂ϕ∂z, ᶠϕ, [ϕ_bot], [rtol])

Sets
`ᶠϕ```(z) = \\int_{z_{bot}}^z\\,```∂ϕ∂z```(```ᶠϕ```(z'), z')\\,dz' +{}```ϕ_bot`,
where `∂ϕ∂z` can be any scalar-valued two-argument function. The output `ᶠϕ`
satisfies `ᶜgradᵥ.(ᶠϕ) ≈ ∂ϕ∂z.(ᶜint.(ᶠϕ), ᶜz)`, where `ᶜgradᵥ = GradientF2C()`,
`ᶜint = InterpolateF2C()`, and `ᶜz = Fields.coordinate_field(ᶜint.(ᶠϕ)).z`, and
where the approximation is accurate to a relative tolerance of `rtol`. The
default value of `ϕ_bot` is 0, and the default value of `rtol` is 0.001.
"""
function column_integral_indefinite!(ᶠϕ, ᶜ∂ϕ∂z, ϕ_bot = rzero(eltype(ᶠϕ)))
    ᶜΔϕ = Base.Broadcast.broadcasted(⊠, ᶜ∂ϕ∂z, Fields.Δz_field(axes(ᶜ∂ϕ∂z)))
    column_accumulate!(⊞, ᶠϕ, ᶜΔϕ; init = ϕ_bot)
end
function column_integral_indefinite!(
    ∂ϕ∂z::F,
    ᶠϕ,
    ϕ_bot = eltype(ᶠϕ)(0),
    rtol = eltype(ᶠϕ)(0.001),
) where {F <: Function}
    device = ClimaComms.device(ᶠϕ)
    face_space = axes(ᶠϕ)
    center_space = if face_space isa Spaces.FaceFiniteDifferenceSpace
        Spaces.CenterFiniteDifferenceSpace(face_space)
    elseif face_space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)
    else
        error("output of column_integral_indefinite! must be on cell faces")
    end
    ᶜz = Fields.coordinate_field(center_space).z
    ᶜΔz = Fields.Δz_field(center_space)
    ᶜz_and_Δz = Base.Broadcast.broadcasted(tuple, ᶜz, ᶜΔz)
    column_accumulate!(ᶠϕ, ᶜz_and_Δz; init = ϕ_bot) do ϕ_prev, (z, Δz)
        residual(ϕ_new) = (ϕ_new - ϕ_prev) / Δz - ∂ϕ∂z((ϕ_prev + ϕ_new) / 2, z)
        (; converged, root) = RootSolvers.find_zero(
            residual,
            RootSolvers.NewtonsMethodAD(ϕ_prev),
            RootSolvers.CompactSolution(),
            RootSolvers.RelativeSolutionTolerance(rtol),
        )
        ClimaComms.@assert device converged "∂ϕ∂z could not be integrated over \
                                             z = $z with rtol set to $rtol"
        return root
    end
end

################################################################################

const PointwiseOrColumnwiseBroadcasted = Union{
    Base.Broadcast.Broadcasted{
        <:Union{Fields.FieldStyle, Operators.AbstractStencilStyle},
    },
    Operators.StencilBroadcasted,
}

Base.@propagate_inbounds function get_level_value(
    space::Spaces.FiniteDifferenceSpace,
    field_or_bc,
    level,
)
    is_periodic = Topologies.isperiodic(Spaces.vertical_topology(space))
    (_, lw, rw, _) = window_bounds(space, field_or_bc)
    window = if !is_periodic && level < lw
        LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    elseif !is_periodic && level > rw
        RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    else
        Interior()
    end
    return getidx(space, field_or_bc, window, level, (1, 1, 1))
end

"""
    UnspecifiedInit()

Analogue of `Base._InitialValue` for `column_reduce!` and `column_accumulate!`.
"""
struct UnspecifiedInit end

"""
    column_reduce!(f, output, input; [init], [transform])

Applies `reduce` to `input` along the vertical direction, storing the result in
`output`. The `input` can be either a `Field` or an `AbstractBroadcasted` that
performs pointwise or columnwise operations on `Field`s. Each reduced value is
computed by iteratively applying `f` to the values in `input`, starting from the
bottom of each column and moving upward, and the result of the final iteration
is passed to the `transform` function before being stored in `output`. If `init`
is specified, it is used as the initial value of the iteration; otherwise, the
value at the bottom of each column in `input` is used as the initial value.
    
With `first_level` and `last_level` denoting the indices of the boundary levels
of `input`, the reduction in each column can be summarized as follows:
  - If `init` is unspecified,
    ```
    reduced_value = input[first_level]
    for level in (first_level + 1):last_level
        reduced_value = f(reduced_value, input[level])
    end
    output[] = transform(reduced_value)
    ```
  - If `init` is specified,
    ```
    reduced_value = init
    for level in first_level:last_level
        reduced_value = f(reduced_value, input[level])
    end
    output[] = transform(reduced_value)
    ```
"""
function column_reduce!(
    f::F,
    output::Fields.Field,
    input::Union{Fields.Field, PointwiseOrColumnwiseBroadcasted};
    init = UnspecifiedInit(),
    transform::T = identity,
) where {F, T}
    device = ClimaComms.device(output)
    space = axes(input)
    column_reduce_device!(device, f, transform, output, input, init, space)
end

column_reduce_device!(
    ::ClimaComms.AbstractCPUDevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
) where {F, T} =
    space isa Spaces.FiniteDifferenceSpace ?
    single_column_reduce!(f, transform, output, input, init, space) :
    Fields.bycolumn(space) do colidx
        single_column_reduce!(
            f,
            transform,
            output[colidx],
            input[colidx],
            init,
            space[colidx],
        )
    end

# On GPUs, input and output go through strip_space to become _input and _output.
function single_column_reduce!(
    f::F,
    transform::T,
    _output,
    _input,
    init,
    space,
) where {F, T}
    first_level = left_idx(space)
    last_level = right_idx(space)
    @inbounds if init == UnspecifiedInit()
        reduced_value = get_level_value(space, _input, first_level)
        next_level = first_level + 1
    else
        reduced_value = init
        next_level = first_level
    end
    @inbounds for level in next_level:last_level
        reduced_value = f(reduced_value, get_level_value(space, _input, level))
    end
    Fields.field_values(_output)[] = transform(reduced_value)
    return nothing
end

"""
    column_accumulate!(f, output, input; [init], [transform])

Applies `accumulate` to `input` along the vertical direction, storing the result
in `output`. The `input` can be either a `Field` or an `AbstractBroadcasted`
that performs pointwise or columnwise operations on `Field`s. Each accumulated
value is computed by iteratively applying `f` to the values in `input`, starting
from the bottom of each column and moving upward, and the result of each
iteration is passed to the `transform` function before being stored in `output`.
The `init` value is is optional for center-to-center, face-to-face, and
face-to-center accumulation, but it is required for center-to-face accumulation.
    
With `first_level` and `last_level` denoting the indices of the boundary levels
of `input`, the accumulation in each column can be summarized as follows:
  - For center-to-center and face-to-face accumulation with `init` unspecified,
    ```
    accumulated_value = input[first_level]
    output[first_level] = transform(accumulated_value)
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level] = transform(accumulated_value)
    end
    ```
  - For center-to-center and face-to-face accumulation with `init` specified,
    ```
    accumulated_value = init
    for level in first_level:last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level] = transform(accumulated_value)
    end
    ```
  - For face-to-center accumulation with `init` unspecified,
    ```
    accumulated_value = input[first_level]
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level - half] = transform(accumulated_value)
    end
    ```
  - For face-to-center accumulation with `init` specified,
    ```
    accumulated_value = f(init, input[first_level])
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level - half] = transform(accumulated_value)
    end
    ```
  - For center-to-face accumulation,
    ```
    accumulated_value = init
    output[first_level - half] = transform(accumulated_value)
    for level in first_level:last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level + half] = transform(accumulated_value)
    end
    ```
"""
function column_accumulate!(
    f::F,
    output::Fields.Field,
    input::Union{Fields.Field, PointwiseOrColumnwiseBroadcasted};
    init = UnspecifiedInit(),
    transform::T = identity,
) where {F, T}
    device = ClimaComms.device(output)
    space = axes(input)
    init == UnspecifiedInit() &&
        Spaces.staggering(space) == Spaces.CellCenter() &&
        Spaces.staggering(axes(output)) == Spaces.CellFace() &&
        error("init must be specified for center-to-face accumulation")
    column_accumulate_device!(device, f, transform, output, input, init, space)
end

column_accumulate_device!(
    ::ClimaComms.AbstractCPUDevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
) where {F, T} =
    space isa Spaces.FiniteDifferenceSpace ?
    single_column_accumulate!(f, transform, output, input, init, space) :
    Fields.bycolumn(space) do colidx
        single_column_accumulate!(
            f,
            transform,
            output[colidx],
            input[colidx],
            init,
            space[colidx],
        )
    end

# On GPUs, input and output go through strip_space to become _input and _output.
function single_column_accumulate!(
    f::F,
    transform::T,
    _output,
    _input,
    init,
    space,
) where {F, T}
    device = ClimaComms.device(space)
    first_level = left_idx(space)
    last_level = right_idx(space)
    output = unstrip_space(_output, space)
    is_c2c_or_f2f = Spaces.staggering(space) == Spaces.staggering(axes(output))
    is_c2f = !is_c2c_or_f2f && Spaces.staggering(space) == Spaces.CellCenter()
    is_f2c = !is_c2c_or_f2f && !is_c2f
    @inbounds if init == UnspecifiedInit()
        @assert !is_c2f
        accumulated_value = get_level_value(space, _input, first_level)
        next_level = first_level + 1
        init_output_level = is_c2c_or_f2f ? first_level : nothing
    else
        accumulated_value =
            is_f2c ? f(init, get_level_value(space, _input, first_level)) : init
        next_level = is_f2c ? first_level + 1 : first_level
        init_output_level = is_c2f ? first_level - half : nothing
    end
    @inbounds if !isnothing(init_output_level)
        Fields.level(output, init_output_level)[] = transform(accumulated_value)
    end
    @inbounds for level in next_level:last_level
        accumulated_value =
            f(accumulated_value, get_level_value(space, _input, level))
        output_level =
            is_c2c_or_f2f ? level : (is_c2f ? level + half : level - half)
        Fields.level(output, output_level)[] = transform(accumulated_value)
    end
end
