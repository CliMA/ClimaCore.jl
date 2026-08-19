# Note: This list must be kept up-to-date with finitedifference.jl.
const OneArgFDOperatorWithCenterInput = Union{
    Operators.InterpolateC2F,
    Operators.LeftBiasedC2F,
    Operators.RightBiasedC2F,
    Operators.GradientC2F,
    Operators.DivergenceC2F,
    Operators.CurlC2F,
}
const OneArgFDOperatorWithFaceInput = Union{
    Operators.InterpolateF2C,
    Operators.LeftBiasedF2C,
    Operators.RightBiasedF2C,
    Operators.SetBoundaryOperator,
    Operators.GradientF2C,
    Operators.DivergenceF2C,
}
const TwoArgFDOperatorWithCenterInput = Union{
    Operators.WeightedInterpolateC2F,
}
const TwoArgFDOperatorWithFaceInput = Union{
    Operators.WeightedInterpolateF2C,
}

const OneArgFDOperator =
    Union{OneArgFDOperatorWithCenterInput, OneArgFDOperatorWithFaceInput}
const TwoArgFDOperator =
    Union{TwoArgFDOperatorWithCenterInput, TwoArgFDOperatorWithFaceInput}

# The advection operators are two-argument operators over a center-valued
# advected field, but unlike the operators above they are only rewritten as
# operator-matrix multiplies when their interior stencil and boundary
# reconstructions are all linear in the advected argument
# (`Operators.has_linear_stencil`); the rest are evaluated pointwise.
const FDOperatorWithCenterInput = Union{
    OneArgFDOperatorWithCenterInput,
    TwoArgFDOperatorWithCenterInput,
    Operators.AdvectionOperator,
}
const FDOperatorWithFaceInput =
    Union{OneArgFDOperatorWithFaceInput, TwoArgFDOperatorWithFaceInput}

operator_input_space(
    ::FDOperatorWithCenterInput,
    space::Spaces.FiniteDifferenceSpace,
) = Spaces.CenterFiniteDifferenceSpace(space)
operator_input_space(
    ::FDOperatorWithCenterInput,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(space)
operator_input_space(
    ::FDOperatorWithFaceInput,
    space::Spaces.FiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(space)
operator_input_space(
    ::FDOperatorWithFaceInput,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

operator_input_space(
    ::FDOperatorWithCenterInput,
    space::Spaces.MultiColumnFiniteDifferenceSpace,
) = Spaces.CenterMultiColumnFiniteDifferenceSpace(space)
operator_input_space(
    ::FDOperatorWithFaceInput,
    space::Spaces.MultiColumnFiniteDifferenceSpace,
) = Spaces.FaceMultiColumnFiniteDifferenceSpace(space)

has_affine_bc(op) = unrolled_any(
    bc ->
        bc isa Union{
            Operators.SetValue,
            Operators.SetGradient,
            Operators.SetDivergence,
            Operators.SetCurl,
        } && ((typeof(bc.val) <: Fields.Field) || bc.val != rzero(typeof(bc.val))),
    op.bcs,
)

################################################################################

struct FDOperatorMatrix{O <: Operators.FiniteDifferenceOperator} <:
       Operators.FiniteDifferenceOperator
    op::O
end
function FDOperatorMatrix(op::O) where {O}
    has_affine_bc(op) &&
        @warn "$(O.name.name) applies an affine transformation because of the \
               boundary conditions it has been assigned; in order to be \
               represented by an operator matrix, it must be converted into a \
               linear operator, so its boundary conditions will be zeroed out"
    return FDOperatorMatrix{O}(op)
end

Operators.strip_space(op::FDOperatorMatrix, parent_space) =
    FDOperatorMatrix(Operators.strip_space(op.op, parent_space))

struct LazyOneArgFDOperatorMatrix{O <: OneArgFDOperator} <: AbstractLazyOperator
    op::O
end

Adapt.adapt_structure(to, op::FDOperatorMatrix) =
    FDOperatorMatrix(Adapt.adapt_structure(to, op.op))

# Since the operator matrix of a one-argument operator does not have any
# arguments, we need to use a lazy operator to add an argument.
replace_lazy_operator(space, lazy_op::LazyOneArgFDOperatorMatrix) =
    Base.Broadcast.broadcasted(
        FDOperatorMatrix(lazy_op.op),
        Fields.local_geometry_field(operator_input_space(lazy_op.op, space)),
    )

# Since the operator matrix of a two-argument operator already has one argument,
# we can modify Base.broadcasted to add a second argument.
Base.Broadcast.broadcasted(
    op_matrix::FDOperatorMatrix{
        <:Union{TwoArgFDOperator, Operators.AdvectionOperator},
    },
    arg,
) = Base.Broadcast.broadcasted(
    op_matrix,
    arg,
    Fields.local_geometry_field(operator_input_space(op_matrix.op, axes(arg))),
)

# A boundary condition that fixes a value (SetValue, SetGradient, SetDivergence,
# or SetCurl) contributes an affine (constant) term that a linear operator matrix
# cannot produce on its own. When a broadcast is rewritten as a matrix multiply,
# that term is reinjected with a SetBoundaryOperator, and `modifies_output` /
# `modifies_input` decide where: `modifies_output` conditions are applied to the
# result (after the multiply), while `modifies_input` conditions are applied to
# the argument (before the multiply). A condition that is linear (e.g. Extrapolate
# on the interpolation operators) is encoded directly in the matrix and is neither.
#
# For nearly every operator such a boundary condition prescribes the operator's
# output at the boundary, so it modifies the output. Examples:
#  - InterpolateC2F  with SetValue(x₀):    I(x)[½] = x₀
#  - LeftBiasedF2C   with SetValue(x₀):    L(x)[1] = x₀
#  - GradientC2F     with SetGradient(v₀): G(x)[½] = v₀
#
# GradientF2C and DivergenceF2C are the exception, and the only operators for
# which `modifies_input` is true. They map faces to centers, so the domain
# boundary (always a face) is a point of their input, not their output. A
# SetValue there prescribes the argument's boundary-face value, which the
# derivative stencil then differences against the adjacent interior face:
#  - GradientF2C   with SetValue(x₀): G(x)[1]³ = x[1+½] - x₀
#  - DivergenceF2C with SetValue(v₀): D(v)[1]  = (Jv³[1+½] - Jv³₀) / J[1]
# Because x₀ enters through the input, the operator matrix keeps its ordinary
# interior stencil and x₀ is written into the argument's boundary face rather than
# added to the result; hence `modifies_input` is true and `modifies_output` false.
modifies_output(
    op,
    boundary_condition::Union{
        Operators.SetGradient,
        Operators.SetDivergence,
        Operators.SetCurl,
    },
) = true
modifies_output(
    op::Union{Operators.GradientF2C, Operators.DivergenceF2C},
    boundary_condition::Operators.SetValue,
) = false
modifies_output(op, boundary_condition::Operators.SetValue) = true
modifies_output(op, boundary_condition) = false

modifies_input(
    op::Union{Operators.GradientF2C, Operators.DivergenceF2C},
    boundary_condition::Operators.SetValue,
) = true
modifies_input(op, boundary_condition) = false

# An operator's boundary conditions are split into three groups. Those that are
# linear can be encoded directly in the operator matrix; the rest modify the
# operator's input or output, and must instead be reapplied to the argument
# (before the matrix multiply) or to the result (after it) using a
# SetBoundaryOperator. Each boundary condition belongs to exactly one group.
filter_bcs(f::F, bcs::NamedTuple) where {F} =
    let kept = unrolled_filter(name -> f(bcs[name]), keys(bcs))
        NamedTuple{kept}(unrolled_map(name -> bcs[name], kept))
    end
matrix_bcs(op) =
    filter_bcs(bc -> !modifies_input(op, bc) && !modifies_output(op, bc), op.bcs)
input_bcs(op) = filter_bcs(Base.Fix1(modifies_input, op), op.bcs)
output_bcs(op) = filter_bcs(Base.Fix1(modifies_output, op), op.bcs)

# Returns `op` carrying only the boundary conditions that can be encoded in its
# operator matrix. Operators without boundary conditions are returned unchanged,
# avoiding an unnecessary rebuild.
op_with_matrix_bcs(op) =
    isempty(op.bcs) ? op : Base.typename(typeof(op)).wrapper(matrix_bcs(op))

# Constructs the `op_matrix * arg` StencilBroadcasted that applies an operator
# matrix to `arg`.
multiply_matrix_broadcasted(::Type{Style}, op_matrix, arg, axes, work) where {Style} =
    let args = (op_matrix, arg)
        Operators.StencilBroadcasted{
            Style,
            MultiplyColumnwiseBandMatrixField,
            typeof(args),
            typeof(axes),
            typeof(work),
        }(
            MultiplyColumnwiseBandMatrixField(),
            args,
            axes,
            work,
        )
    end

# Wraps `arg` in a StencilBroadcasted that applies the SetBoundaryOperator `op`.
apply_boundary_operator(::Type{Style}, op, arg, axes, work) where {Style} =
    let args = (arg,)
        Operators.StencilBroadcasted{
            Style,
            typeof(op),
            typeof(args),
            typeof(axes),
            typeof(work),
        }(
            op,
            args,
            axes,
            work,
        )
    end

# A gradient operator matrix has vector entries and a divergence operator matrix has
# covector entries, so for the plain `*` of the matrix multiply to produce a result of
# the right rank, a gradient needs an adjoint on its argument and a divergence needs
# one on its result.
adjoint_matrix_arg(op, arg) = arg
adjoint_matrix_arg(::Operators.GradientOperator, arg) =
    Base.Broadcast.broadcasted(adjoint, arg)
adjoint_matrix_result(op, result) = result
adjoint_matrix_result(::Operators.DivergenceOperator, result) =
    Base.Broadcast.broadcasted(adjoint, result)

# Builds an ordinary StencilBroadcasted, without rewriting `op` into a matrix multiply.
unconverted_stencil_broadcasted(
    ::Type{Style},
    op,
    args::Args,
    axes,
    work::Work,
) where {Style, Args, Work} = Operators.StencilBroadcasted{
    Style,
    typeof(op),
    Args,
    typeof(axes),
    Work,
}(
    op,
    args,
    axes,
    work,
)

# A SetBoundaryOperator has no operator matrix: it is what the conversions below use to
# reapply the boundary conditions they strip out, so it is built verbatim.
Operators.StencilBroadcasted{Style}(
    op::Operators.SetBoundaryOperator,
    args::Args,
    axes::Spaces.AbstractSpace,
    work::Work = nothing,
) where {Style, Args, Work} =
    unconverted_stencil_broadcasted(Style, op, args, axes, work)

# Converts a broadcast over a one-argument operator, `op(arg)`, into the
# equivalent operator matrix expression, `op_matrix() * arg`. Boundary conditions
# that modify the operator's input or output are stripped from the matrix and
# reapplied to `arg` or to the result with a SetBoundaryOperator. Gradient and
# Divergence operators require an additional adjoint on the input and output,
# respectively.
function Operators.StencilBroadcasted{Style}(
    op::OneArgFDOperator,
    args::Args,
    axes::Spaces.AbstractSpace,
    work::Work = nothing,
) where {Style, Args, Work}
    op_matrix = Base.Broadcast.broadcasted(
        FDOperatorMatrix(op_with_matrix_bcs(op)),
        Fields.local_geometry_field(operator_input_space(op, axes)),
    )

    bcs_in = input_bcs(op)
    arg =
        isempty(bcs_in) ? args[1] :
        apply_boundary_operator(
            Style,
            Operators.SetBoundaryOperator(bcs_in),
            args[1],
            Base.axes(args[1]),
            nothing,
        )
    arg = adjoint_matrix_arg(op, arg)

    result = multiply_matrix_broadcasted(Style, op_matrix, arg, axes, work)
    result = adjoint_matrix_result(op, result)

    bcs_out = output_bcs(op)
    return isempty(bcs_out) ? result :
           apply_boundary_operator(
        Style,
        Operators.SetBoundaryOperator(bcs_out),
        result,
        axes,
        work,
    )
end


# Converts a broadcast over a two-argument operator, `op(weight, arg)`, into the
# equivalent operator matrix expression, `op_matrix(weight) * arg`. As for one-argument
# operators, boundary conditions that modify the output are stripped from the matrix and
# reapplied to the result with a SetBoundaryOperator. In practice only
# WeightedInterpolateC2F has such conditions (SetValue); every other two-argument
# operator's conditions are linear, so `output_bcs` is empty for them and both
# `op_with_matrix_bcs` and this function leave them untouched.
Operators.StencilBroadcasted{Style}(
    op::TwoArgFDOperator,
    args::Args,
    axes::Spaces.AbstractSpace,
    work::Work = nothing,
) where {Style, Args, Work} =
    two_arg_matrix_broadcasted(Style, op, args, axes, work)

# An advection operator is only equivalent to a matrix multiply when its
# interior stencil and its boundary reconstructions are all linear in the
# advected argument; everything else (a flux-limited operator, or a
# user-supplied ghost-point reconstruction) is left as an ordinary stencil and
# evaluated pointwise. `has_linear_stencil` only depends on the types of the
# operator and its boundary conditions, so this branch folds at compile time.
Operators.StencilBroadcasted{Style}(
    op::Operators.AdvectionOperator,
    args::Args,
    axes::Spaces.AbstractSpace,
    work::Work = nothing,
) where {Style, Args, Work} =
    Operators.has_linear_stencil(op) ?
    two_arg_matrix_broadcasted(Style, op, args, axes, work) :
    unconverted_stencil_broadcasted(Style, op, args, axes, work)

function two_arg_matrix_broadcasted(
    ::Type{Style},
    op,
    args,
    axes,
    work,
) where {Style}
    op_matrix =
        Base.Broadcast.broadcasted(FDOperatorMatrix(op_with_matrix_bcs(op)), args[1])

    result = multiply_matrix_broadcasted(Style, op_matrix, args[2], axes, work)

    bcs_out = output_bcs(op)
    return isempty(bcs_out) ? result :
           apply_boundary_operator(
        Style,
        Operators.SetBoundaryOperator(bcs_out),
        result,
        axes,
        work,
    )
end


"""
    operator_matrix(op)

Constructs a new operator (or operator-like object) that generates the matrix
applied by `op` to its final argument. If `op_matrix = operator_matrix(op)`,
we can use the following identities:

  - When `op` takes one argument, `@. op(arg) == @. op_matrix() * arg`.
  - When `op` takes multiple arguments,
    `@. op(args..., arg) == @. op_matrix(args...) * arg`.

When `op` takes more than one argument, `operator_matrix(op)` constructs a
`FiniteDifferenceOperator` that generates the operator matrix. When `op` only
takes one argument, it instead constructs an `AbstractLazyOperator`, which is
internally converted into a `FiniteDifferenceOperator` when used in a broadcast
expression. Implementing `op_matrix` as a lazy operator allows us to add an
argument to the expression `op_matrix.()`, and we then use this argument to
infer the space and element type of the operator matrix.

As an example, the `InterpolateF2C()` operator on a space with ``n`` cell
centers applies an ``n \\times (n + 1)`` bidiagonal matrix:

```math
\\textrm{interp}(arg) = \\begin{bmatrix}
    0.5 &     0.5 &       0 & \\cdots &       0 &       0 &       0 \\\\
      0 &     0.5 &     0.5 & \\cdots &       0 &       0 &       0 \\\\
      0 &       0 &     0.5 & \\cdots &       0 &       0 &       0 \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\
      0 &       0 &       0 & \\cdots &     0.5 &     0.5 &       0 \\\\
      0 &       0 &       0 & \\cdots &       0 &     0.5 &     0.5
\\end{bmatrix} * arg
```

The `GradientF2C()` operator applies a similar matrix, but with different
entries:

```math
\\textrm{grad}(arg) = \\begin{bmatrix}
-\\textbf{e}^3 &  \\textbf{e}^3 &              0 & \\cdots &              0 &              0 &             0 \\\\
             0 & -\\textbf{e}^3 &  \\textbf{e}^3 & \\cdots &              0 &              0 &             0 \\\\
             0 &              0 & -\\textbf{e}^3 & \\cdots &              0 &              0 &             0 \\\\
       \\vdots &        \\vdots &        \\vdots & \\ddots &        \\vdots &        \\vdots &       \\vdots \\\\
             0 &              0 &              0 & \\cdots & -\\textbf{e}^3 &  \\textbf{e}^3 &             0 \\\\
             0 &              0 &              0 & \\cdots &              0 & -\\textbf{e}^3 & \\textbf{e}^3
\\end{bmatrix} * arg
```

The unit vector ``\\textbf{e}^3``, which can also be thought of as the
differential along the third coordinate axis (``\\textrm{d}\\xi^3``), is
implemented as a `Geometry.Covariant3Vector(1)`.

Not all operators have well-defined operator matrices. For example, the operator
`GradientC2F(; bottom = SetGradient(grad_b), top = SetGradient(grad_t))` applies
an affine transformation:

```math
\\textrm{grad}(arg) = \\begin{bmatrix}
grad_b \\\\ 0 \\\\ 0 \\\\ \\vdots \\\\ 0 \\\\ 0 \\\\ grad_t
\\end{bmatrix} + \\begin{bmatrix}
             0 &              0 &              0 & \\cdots &              0 &             0 \\\\
-\\textbf{e}^3 &  \\textbf{e}^3 &              0 & \\cdots &              0 &             0 \\\\
             0 & -\\textbf{e}^3 &  \\textbf{e}^3 & \\cdots &              0 &             0 \\\\
       \\vdots &        \\vdots &        \\vdots & \\ddots &        \\vdots &       \\vdots \\\\
             0 &              0 &              0 & \\cdots &  \\textbf{e}^3 &             0 \\\\
             0 &              0 &              0 & \\cdots & -\\textbf{e}^3 & \\textbf{e}^3 \\\\
             0 &              0 &              0 & \\cdots &              0 &             0
\\end{bmatrix} * arg
```

However, this simplifies to a linear transformation when ``grad_b`` and
``grad_t`` are both 0:

```math
\\textrm{grad}(arg) = \\begin{bmatrix}
             0 &              0 &              0 & \\cdots &              0 &             0 \\\\
-\\textbf{e}^3 &  \\textbf{e}^3 &              0 & \\cdots &              0 &             0 \\\\
             0 & -\\textbf{e}^3 &  \\textbf{e}^3 & \\cdots &              0 &             0 \\\\
       \\vdots &        \\vdots &        \\vdots & \\ddots &        \\vdots &       \\vdots \\\\
             0 &              0 &              0 & \\cdots &  \\textbf{e}^3 &             0 \\\\
             0 &              0 &              0 & \\cdots & -\\textbf{e}^3 & \\textbf{e}^3 \\\\
             0 &              0 &              0 & \\cdots &              0 &             0
\\end{bmatrix} * arg
```

In general, when `op` has nonzero boundary conditions that make it apply an
affine transformation, `operator_matrix(op)` will print out a warning and zero
out the boundary conditions before computing the operator matrix.

In addition to affine transformations, there are also some operators that apply
nonlinear transformations to their arguments; that is, transformations which
cannot be accurately approximated without using more terms of the form

```math
\\textrm{op}(\\textbf{0}) +
\\textrm{op}'(\\textbf{0}) * arg +
\\textrm{op}''(\\textbf{0}) * arg * arg +
\\ldots.
```

When `op` is such an operator, `operator_matrix(op)` will throw an error. In the
future, we may want to modify `operator_matrix(op)` so that it will instead
return ``\\textrm{op}'(\\textbf{0})``, where ``\\textbf{0} ={} ```zero.(arg)`.
"""
operator_matrix(op::OneArgFDOperator) = LazyOneArgFDOperatorMatrix(op)
operator_matrix(op::TwoArgFDOperator) = FDOperatorMatrix(op)
operator_matrix(op::Operators.AdvectionOperator) =
    Operators.has_linear_stencil(op) ? FDOperatorMatrix(op) :
    error(
        "$(typeof(op).name.name) applies a nonlinear transformation to its \
         argument (in its interior stencil or through a boundary condition), \
         so it cannot be represented by a matrix",
    )
operator_matrix(::O) where {O <: Operators.AbstractOperator} =
    error("operator_matrix has not been defined for $(O.name.name)")

################################################################################

Operators.get_boundary(
    op_matrix::FDOperatorMatrix,
    lbw::Operators.LeftBoundaryWindow{name},
) where {name} = Operators.get_boundary(op_matrix.op, lbw)
Operators.get_boundary(
    op_matrix::FDOperatorMatrix,
    rbw::Operators.RightBoundaryWindow{name},
) where {name} = Operators.get_boundary(op_matrix.op, rbw)

Operators.stencil_interior_width(op_matrix::FDOperatorMatrix, args...) =
    Operators.stencil_interior_width(op_matrix.op, args...)

Operators.left_interior_idx(
    space::Spaces.AbstractSpace,
    op_matrix::FDOperatorMatrix,
    bc::Operators.AbstractBoundaryCondition,
    args...,
) = Operators.left_interior_idx(space, op_matrix.op, bc, args...)

Operators.right_interior_idx(
    space::Spaces.AbstractSpace,
    op_matrix::FDOperatorMatrix,
    bc::Operators.AbstractBoundaryCondition,
    args...,
) = Operators.right_interior_idx(space, op_matrix.op, bc, args...)

Operators.return_space(op_matrix::FDOperatorMatrix, spaces...) =
    Operators.return_space(op_matrix.op, spaces...)

function Operators.return_eltype(op_matrix::FDOperatorMatrix, args...)
    args′ = args[1:(end - 1)]
    if typeof(args[end]) <: Spaces.AbstractSpace
        FT = Spaces.undertype(args[end])
    else
        FT = Geometry.undertype(eltype(args[end]))
    end
    return op_matrix_row_type(op_matrix.op, FT, args′...)
end

Base.@propagate_inbounds function Operators.stencil_interior(
    op_matrix::FDOperatorMatrix,
    space,
    idx,
    hidx,
    args...,
)
    args′ = args[1:(end - 1)]
    row = op_matrix_interior_row(op_matrix.op, space, idx, hidx, args′...)
    return convert(Operators.return_eltype(op_matrix, args...), row)
end

Base.@propagate_inbounds function Operators.stencil_left_boundary(
    op_matrix::FDOperatorMatrix,
    bc::Operators.AbstractBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
)
    args′ = args[1:(end - 1)]
    row = op_matrix_first_row(op_matrix.op, bc, space, idx, hidx, args′...)
    return convert(Operators.return_eltype(op_matrix, args...), row)
end

Base.@propagate_inbounds function Operators.stencil_right_boundary(
    op_matrix::FDOperatorMatrix,
    bc::Operators.AbstractBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
)
    args′ = args[1:(end - 1)]
    row = op_matrix_last_row(op_matrix.op, bc, space, idx, hidx, args′...)
    return convert(Operators.return_eltype(op_matrix, args...), row)
end

# Simplified methods for when the operator matrix only depends on FT.
op_matrix_row_type(op, ::Type{FT}, args...) where {FT} =
    typeof(op_matrix_interior_row(op, FT))
op_matrix_interior_row(op, space, idx, hidx, args...) =
    op_matrix_interior_row(op, Spaces.undertype(space))
op_matrix_first_row(op, bc, space, idx, hidx, args...) =
    op_matrix_first_row(op, bc, Spaces.undertype(space))
op_matrix_last_row(op, bc, space, idx, hidx, args...) =
    op_matrix_last_row(op, bc, Spaces.undertype(space))

# Fallback methods for unspecified boundary conditions (need to use zero here
# instead of NaN to avoid polluting nearby interior rows with NaNs).
#
# The row must not be the interior row. Only operators whose input is centers reach
# these methods -- every face-input operator has
# `boundary_width(op, ::NullBoundaryCondition) == 0`, so no boundary row is requested
# for it -- and a center-input operator's interior row at the boundary face reaches a
# center outside the domain. The multiply clips those band entries, so the result is
# unaffected, but merely building the row can read out of range:
# `DivergenceOperator`'s row evaluates `LocalGeometry(space, idx - half, hidx)`, which
# is center 0 at the bottom face, and that is a `BoundsError` under
# `--check-bounds=yes`.
Operators.stencil_left_boundary(
    op_matrix::FDOperatorMatrix,
    ::Operators.NullBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
) = rzero(Operators.return_eltype(op_matrix, args...))
Operators.stencil_right_boundary(
    op_matrix::FDOperatorMatrix,
    ::Operators.NullBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
) = rzero(Operators.return_eltype(op_matrix, args...))

# Boundary rows for value-fixing boundary conditions that are still attached to
# the operator matrix. This only happens through the explicit `operator_matrix(op)`
# API (`@. op_matrix() * arg`), which keeps `op`'s boundary conditions inside the
# FDOperatorMatrix. The automatic `@. op(arg)` conversion instead strips these
# conditions from the matrix and reapplies them with a SetBoundaryOperator (see
# `modifies_input` / `modifies_output`), so in that case the matrix carries no
# boundary condition and the NullBoundaryCondition methods above are used instead.
#
# An operator matrix can only capture the linear part of the operator; the
# constant contributed by the boundary value is zeroed out (see `has_affine_bc`).
# For every operator except GradientF2C/DivergenceF2C, a value-fixing condition
# prescribes the output at the boundary as a pure constant, so its linear part is zero
# and the boundary row is all zeros (`rzero` of the row type, which keeps the row's
# bandwidth and zeroes its entries; the multiply clips the out-of-range band entries at
# the column ends, so the row need not be narrowed).
const ValueFixingBoundaryCondition = Union{
    Operators.SetValue,
    Operators.SetGradient,
    Operators.SetDivergence,
    Operators.SetCurl,
}
# The operators whose SetValue fixes an input rather than an output, and so keep a
# genuine boundary stencil in the matrix (see `modifies_input`).
const InputFixingFDOperator =
    Union{Operators.GradientF2C, Operators.DivergenceF2C}

Base.@propagate_inbounds Operators.stencil_left_boundary(
    op_matrix::FDOperatorMatrix,
    ::ValueFixingBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
) = rzero(Operators.return_eltype(op_matrix, args...))
# Mirror of stencil_left_boundary above, for the right boundary.
Base.@propagate_inbounds Operators.stencil_right_boundary(
    op_matrix::FDOperatorMatrix,
    ::ValueFixingBoundaryCondition,
    space,
    idx,
    hidx,
    args...,
) = rzero(Operators.return_eltype(op_matrix, args...))

# GradientF2C/DivergenceF2C with a SetValue are the exception (as with
# `modifies_input`): the condition fixes an input value, and the near-boundary output
# still depends linearly on the adjacent interior input, so the row is the genuine
# boundary stencil (with the fixed input's coefficient dropped) rather than zero. The
# last of `args` is the local geometry field that `op_matrix` was given, which the
# underlying operator's row functions do not take.
Base.@propagate_inbounds function Operators.stencil_left_boundary(
    op_matrix::FDOperatorMatrix{<:InputFixingFDOperator},
    bc::Operators.SetValue,
    space,
    idx,
    hidx,
    args...,
)
    row = op_matrix_first_row(
        op_matrix.op,
        bc,
        space,
        idx,
        hidx,
        args[1:(end - 1)]...,
    )
    return convert(Operators.return_eltype(op_matrix, args...), row)
end
# Mirror of stencil_left_boundary above, for the right boundary.
Base.@propagate_inbounds function Operators.stencil_right_boundary(
    op_matrix::FDOperatorMatrix{<:InputFixingFDOperator},
    bc::Operators.SetValue,
    space,
    idx,
    hidx,
    args...,
)
    row = op_matrix_last_row(
        op_matrix.op,
        bc,
        space,
        idx,
        hidx,
        args[1:(end - 1)]...,
    )
    return convert(Operators.return_eltype(op_matrix, args...), row)
end

################################################################################

# Additional aliases for CenterToFace or FaceToCenter matrix rows
const LowerDiagonalMatrixRow = BandMatrixRow{-1 + half, 1}    # -0.5
const UpperDiagonalMatrixRow = BandMatrixRow{half, 1}         #  0.5
const LowerTridiagonalMatrixRow = BandMatrixRow{-2 + half, 3} # -1.5, -0.5, 0.5
const UpperTridiagonalMatrixRow = BandMatrixRow{-1 + half, 3} # -0.5,  0.5, 1.5

const C3{T} = Geometry.Covariant3Vector{T}
const CT3{T} = Geometry.Contravariant3Vector{T}
# Covector (row-vector) type for C3: result of adjoint(C3{T}(x))
const C3Cov{T} = Geometry.Tensor{
    2, T,
    Tuple{Geometry.ScalarComponents, Geometry.Components{Geometry.Covariant, (3,)}},
    Adjoint{T, SVector{1, T}},
}
const CT12_CT12{T} = Geometry.Tensor{
    2,
    T,
    Tuple{Geometry.Contravariant12Axis, Geometry.Contravariant12Axis},
    SMatrix{2, 2, T, 4},
}

# Levi-Civita symbol in 2D
const εⁱʲ = Geometry.Tensor(
    SMatrix{2, 2}(0, 1, -1, 0),
    (Geometry.Contravariant12Axis(), Geometry.Contravariant12Axis()),
)

Base.@propagate_inbounds ct3_data(velocity, space, idx, hidx) =
    Geometry.contravariant3(
        Operators.getidx(space, velocity, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )

################################################################################

# Boundary rows for the Extrapolate boundary condition, which is only accepted
# by the interpolation operators: the output at the boundary face is the
# input's value at the closest interior point, so the boundary row is an
# identity entry pointing at that point. Such a row fits inside the interior
# row's band, so the operator's row type is unchanged.
const CopyInputExtrapolateOp =
    Union{Operators.InterpolateC2F, Operators.WeightedInterpolateC2F}

op_matrix_first_row(
    ::CopyInputExtrapolateOp,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(true)
op_matrix_last_row(
    ::CopyInputExtrapolateOp,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(true)

################################################################################

op_matrix_interior_row(
    ::Union{Operators.InterpolateC2F, Operators.InterpolateF2C},
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(FT(1), FT(1)) / 2

op_matrix_interior_row(
    ::Union{Operators.LeftBiasedC2F, Operators.LeftBiasedF2C},
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(true)

op_matrix_interior_row(
    ::Union{Operators.RightBiasedC2F, Operators.RightBiasedF2C},
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(true)

op_matrix_row_type(
    ::Operators.WeightedInterpolationOperator,
    ::Type{FT},
    weight,
) where {FT} = BidiagonalMatrixRow{eltype(weight)}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.WeightedInterpolationOperator,
    space,
    idx,
    hidx,
    weight,
)
    w⁻ = Operators.getidx(space, weight, idx - half, hidx)
    w⁺ = Operators.getidx(space, weight, idx + half, hidx)
    denominator = w⁻ + w⁺
    return BidiagonalMatrixRow(w⁻ / denominator, w⁺ / denominator)
end

op_matrix_row_type(
    ::Operators.UpwindBiasedProductC2F,
    ::Type{FT},
    _,
) where {FT} = BidiagonalMatrixRow{CT3{FT}}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.UpwindBiasedProductC2F,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    return BidiagonalMatrixRow(v³ + av³, v³ - av³) / 2
end

op_matrix_row_type(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Type{FT},
    _,
) where {FT} = QuaddiagonalMatrixRow{CT3{FT}}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    return QuaddiagonalMatrixRow(-v³ - av³, 7v³ + 3av³, 7v³ - 3av³, -v³ + av³) /
           12
end

# Boundary rows for the matrix-representable advection operators. Boundary
# faces are computed with the interior stencil, padding the ghost points it
# reaches with one-sided reconstructions from the closest interior points:
#
#  - FirstOrderOneSided (the default): every ghost point takes the value of
#    the closest interior point, matching the index clamping that the
#    pointwise-evaluated advection operators apply.
#  - ThirdOrderOneSided (Upwind3rdOrderBiasedProductC2F only): at the face one
#    in from the boundary, the single ghost point is linearly extrapolated
#    from the two closest interior points (2 * closest - second-closest); at
#    the boundary face itself, both ghost points take the value of the closest
#    interior point.
#
# The matrix multiply clips out-of-range band entries instead of clamping
# their indices, so each face whose interior row reaches a ghost point gets a
# boundary row that folds the ghost coefficients into the closest interior
# columns, leaving zeros in the out-of-range slots (which the multiply then
# clips). The folded rows fit the interior row type, so no widening is needed.
# The rows are reached through the generic FDOperatorMatrix
# stencil_left_boundary / stencil_right_boundary methods; advection operators
# never carry NullBoundaryCondition (FirstOrderOneSided is added by default),
# so no zero-row fallback applies here.

# At the boundary face, the padded upwind and downwind values coincide, so the
# folded row is `v³` on the closest interior center regardless of upwind
# direction: ((v³ + |v³|) + (v³ - |v³|)) / 2 = v³.
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    return BidiagonalMatrixRow(zero(v³), v³)
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    return BidiagonalMatrixRow(v³, zero(v³))
end

# The interior stencil reaches one center beyond its face on each side, so two
# faces on each side need folded rows. At the boundary face both lower (resp.
# upper) ghost coefficients fold into the closest center; at the face one in
# from the boundary only the outermost one does. With FirstOrderOneSided (or
# no boundary condition), the ghost coefficient folds entirely into the
# closest center: e.g. at the bottom, the interior row
# (-v³ - |v³|, 7v³ + 3|v³|, 7v³ - 3|v³|, -v³ + |v³|) / 12 becomes
# (0, 6v³ + 2|v³|, 7v³ - 3|v³|, -v³ + |v³|) / 12 at the face one in from the
# boundary, and (0, 0, 13v³ - |v³|, -v³ + |v³|) / 12 at the boundary face.
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Operators.FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    z = zero(v³)
    return idx == Operators.left_face_boundary_idx(space) ?
           QuaddiagonalMatrixRow(z, z, 13v³ - av³, -v³ + av³) / 12 :
           QuaddiagonalMatrixRow(z, 6v³ + 2av³, 7v³ - 3av³, -v³ + av³) / 12
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Operators.FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    z = zero(v³)
    return idx == Operators.right_face_boundary_idx(space) ?
           QuaddiagonalMatrixRow(-v³ - av³, 13v³ + av³, z, z) / 12 :
           QuaddiagonalMatrixRow(-v³ - av³, 7v³ + 3av³, 6v³ - 2av³, z) / 12
end

# With ThirdOrderOneSided, the boundary face is treated as with
# FirstOrderOneSided, but at the face one in from the boundary the ghost point
# is linearly extrapolated from the two closest interior points, so its
# coefficient c folds as 2c into the closest column and -c into the
# second-closest: e.g. at the bottom, the interior row becomes
# (0, 5v³ + |v³|, 8v³ - 2|v³|, -v³ + |v³|) / 12.
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Operators.ThirdOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    z = zero(v³)
    return idx == Operators.left_face_boundary_idx(space) ?
           QuaddiagonalMatrixRow(z, z, 13v³ - av³, -v³ + av³) / 12 :
           QuaddiagonalMatrixRow(z, 5v³ + av³, 8v³ - 2av³, -v³ + av³) / 12
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Operators.ThirdOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    z = zero(v³)
    return idx == Operators.right_face_boundary_idx(space) ?
           QuaddiagonalMatrixRow(-v³ - av³, 13v³ + av³, z, z) / 12 :
           QuaddiagonalMatrixRow(-v³ - av³, 8v³ + 2av³, 5v³ - av³, z) / 12
end

op_matrix_interior_row(::Operators.SetBoundaryOperator, ::Type{FT}) where {FT} =
    DiagonalMatrixRow(true)

op_matrix_row_type(::Operators.GradientOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow{C3{FT}}
op_matrix_interior_row(::Operators.GradientOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow(-C3(FT(1)), C3(FT(1)))
op_matrix_first_row(
    ::Operators.GradientF2C,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(C3(FT(0)), C3(FT(1)))
op_matrix_last_row(
    ::Operators.GradientF2C,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(-C3(FT(1)), C3(FT(0)))

op_matrix_row_type(::Operators.DivergenceOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow{C3Cov{FT}}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.DivergenceOperator,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    J⁻ = Geometry.LocalGeometry(space, idx - half, hidx).J
    J⁺ = Geometry.LocalGeometry(space, idx + half, hidx).J
    return BidiagonalMatrixRow(-C3(J⁻)', C3(J⁺)') * invJ
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.DivergenceF2C,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    FT = Spaces.undertype(space)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    J⁺ = Geometry.LocalGeometry(space, idx + half, hidx).J
    return BidiagonalMatrixRow(C3(FT(0))', C3(J⁺)') * invJ
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.DivergenceF2C,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    FT = Spaces.undertype(space)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    J⁻ = Geometry.LocalGeometry(space, idx - half, hidx).J
    return BidiagonalMatrixRow(-C3(J⁻)', C3(FT(0))') * invJ
end

op_matrix_row_type(
    ::Operators.CurlFiniteDifferenceOperator,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow{CT12_CT12{FT}}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.CurlFiniteDifferenceOperator,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    return BidiagonalMatrixRow(-εⁱʲ, εⁱʲ) * invJ
end
