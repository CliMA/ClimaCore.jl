# Note: This list must be kept up-to-date with finitedifference.jl.
const OneArgFDOperatorWithCenterInput = Union{
    Operators.InterpolateC2F,
    Operators.BottomBiasedC2F,
    Operators.TopBiasedC2F,
    Operators.GradientC2F,
    Operators.DivergenceC2F,
    Operators.CurlC2F,
}
const OneArgFDOperatorWithFaceInput = Union{
    Operators.InterpolateF2C,
    Operators.BottomBiasedF2C,
    Operators.TopBiasedF2C,
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

# A SetBoundaryOperator is space-preserving, so its operator matrix must be
# built on the argument's own space.
operator_input_space(
    ::Operators.SetBoundaryOperator,
    space::Spaces.FiniteDifferenceSpace,
) = space
operator_input_space(
    ::Operators.SetBoundaryOperator,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
) = space
operator_input_space(
    ::Operators.SetBoundaryOperator,
    space::Spaces.MultiColumnFiniteDifferenceSpace,
) = space

# Whether a boundary value is nonzero. The value can be composite (e.g. a
# `NamedTuple`, with or without `AutoBroadcaster` wrappers), so its zero is
# built through `AutoBroadcaster`s, and the wrappers are dropped from both sides
# because `!=` on `AutoBroadcaster`s is elementwise.
is_nonzero_value(val) =
    drop_auto_broadcasters(val) != drop_auto_broadcasters(zero(add_auto_broadcasters(val)))

has_affine_bc(op) = unrolled_any(
    bc ->
        bc isa Union{
            Operators.SetValue,
            Operators.SetGradient,
            Operators.SetDivergence,
            Operators.SetCurl,
        } && (typeof(bc.val) <: Fields.MaybeLazyField || is_nonzero_value(bc.val)),
    op.bcs,
)

uses_extrapolate(op) =
    unrolled_any(Base.Fix2(isa, Operators.Extrapolate), op.bcs)

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
# the argument (before the multiply). A condition that is linear (e.g. Extrapolate)
# is encoded directly in the matrix and is neither. (For DivergenceF2C this widens
# every matrix row by one diagonal on each side; see `extrapolate_row_type`.)
#
# For nearly every operator such a boundary condition prescribes the operator's
# output at the boundary, so it modifies the output. Examples:
#  - InterpolateC2F  with SetValue(x₀):    I(x)[½] = x₀
#  - BottomBiasedF2C with SetValue(x₀):    B(x)[1] = x₀
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

# Constructs the `op_matrix .* arg` broadcast expression that applies an operator
# matrix to `arg`.
multiply_matrix_broadcasted(::Type{Style}, op_matrix, arg, axes) where {Style} =
    Base.Broadcast.Broadcasted{Style}(
        MultiplyColumnwiseBandMatrixField(),
        (op_matrix, projected_operand(op_matrix, arg)),
        axes,
    )

# Wraps `arg` in a broadcast expression that applies a SetBoundaryOperator.
apply_boundary_operator(::Type{Style}, op, arg, axes) where {Style} =
    Base.Broadcast.Broadcasted{Style}(op, (arg,), axes)

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

# Builds an ordinary broadcast, without rewriting `op` into a matrix multiply.
unconverted_stencil_broadcasted(::Type{Style}, op, args, axes) where {Style} =
    Base.Broadcast.Broadcasted{Style}(op, args, axes)

# A SetBoundaryOperator has no operator matrix: it is what the conversions below use to
# reapply the boundary conditions they strip out, so it is built verbatim.
Base.Broadcast.Broadcasted(
    ::Style,
    op::Operators.SetBoundaryOperator,
    args::Tuple,
    axes::Spaces.AbstractSpace,
) where {Style <: Operators.StencilStyle} =
    unconverted_stencil_broadcasted(Style, op, args, axes)

# Converts a broadcast over a one-argument operator, `op(arg)`, into the
# equivalent operator matrix expression, `op_matrix() * arg`. Boundary conditions
# that modify the operator's input or output are stripped from the matrix and
# reapplied to `arg` or to the result with a SetBoundaryOperator. Gradient and
# Divergence operators require an additional adjoint on the input and output,
# respectively.
function Base.Broadcast.Broadcasted(
    ::Style,
    op::OneArgFDOperator,
    args::Tuple,
    axes::Spaces.AbstractSpace,
) where {Style <: Operators.StencilStyle}
    op_matrix = Base.Broadcast.instantiate(
        Base.Broadcast.broadcasted(
            FDOperatorMatrix(op_with_matrix_bcs(op)),
            Fields.local_geometry_field(operator_input_space(op, axes)),
        ),
    )

    bcs_in = input_bcs(op)
    arg =
        isempty(bcs_in) ? args[1] :
        apply_boundary_operator(
            Style,
            Operators.SetBoundaryOperator(bcs_in),
            args[1],
            Base.axes(args[1]),
        )
    arg = adjoint_matrix_arg(op, arg)

    result = multiply_matrix_broadcasted(Style, op_matrix, arg, axes)
    result = adjoint_matrix_result(op, result)

    bcs_out = output_bcs(op)
    return isempty(bcs_out) ? result :
           apply_boundary_operator(
        Style,
        Operators.SetBoundaryOperator(bcs_out),
        result,
        axes,
    )
end


# Converts a broadcast over a two-argument operator, `op(weight, arg)`, into the
# equivalent operator matrix expression, `op_matrix(weight) * arg`. As for one-argument
# operators, boundary conditions that modify the output are stripped from the matrix and
# reapplied to the result with a SetBoundaryOperator. In practice only
# WeightedInterpolateC2F has such conditions (SetValue); every other two-argument
# operator's conditions are linear, so `output_bcs` is empty for them and both
# `op_with_matrix_bcs` and this function leave them untouched.
Base.Broadcast.Broadcasted(
    ::Style,
    op::TwoArgFDOperator,
    args::Tuple,
    axes::Spaces.AbstractSpace,
) where {Style <: Operators.StencilStyle} =
    two_arg_matrix_broadcasted(Style, op, args, axes)

# An advection operator is only equivalent to a matrix multiply when its
# interior stencil and its boundary reconstructions are all linear in the
# advected argument; everything else (i.e. a flux-limited operator) is left as
# an ordinary stencil and evaluated pointwise. `has_linear_stencil` only
# depends on the types of the operator and its boundary conditions, so this
# branch folds at compile time.
Base.Broadcast.Broadcasted(
    ::Style,
    op::Operators.AdvectionOperator,
    args::Tuple,
    axes::Spaces.AbstractSpace,
) where {Style <: Operators.StencilStyle} =
    Operators.has_linear_stencil(op) ?
    two_arg_matrix_broadcasted(Style, op, args, axes) :
    unconverted_stencil_broadcasted(Style, op, args, axes)

function two_arg_matrix_broadcasted(::Type{Style}, op, args, axes) where {Style}
    op_matrix = Base.Broadcast.instantiate(
        Base.Broadcast.broadcasted(FDOperatorMatrix(op_with_matrix_bcs(op)), args[1]),
    )

    result = multiply_matrix_broadcasted(Style, op_matrix, args[2], axes)

    bcs_out = output_bcs(op)
    return isempty(bcs_out) ? result :
           apply_boundary_operator(
        Style,
        Operators.SetBoundaryOperator(bcs_out),
        result,
        axes,
    )
end


"""
    operator_matrix(op)

Construct a new operator (or operator-like object) that generates the matrix
applied by `op` to its final argument. If `op_matrix = operator_matrix(op)`, the
following identities hold:

  - When `op` takes one argument, `@. op(arg) == @. op_matrix() * arg`.
  - When `op` takes multiple arguments,
    `@. op(args..., arg) == @. op_matrix(args...) * arg`.

These identities do not hold as stated for gradient and divergence operators.
A gradient operator matrix has vector-valued entries and a divergence operator
matrix has covector-valued entries, so when ClimaCore itself rewrites a
gradient or divergence broadcast into a matrix multiply, it compensates with
an `adjoint`: on the argument for gradients and on the result for divergences.
The explicit `@. op_matrix() * arg` form applies no such compensation, so for
a divergence operator it evaluates to `adjoint.(@. op(arg))` rather than
`@. op(arg)`. When the divergence's result is a scalar (e.g. the divergence of
a vector field), the adjoint is a no-op and the identity holds exactly; when
the argument is a higher-rank tensor field, the result holds the same
components in transposed (row) form, and materializing it into a destination
field with the operator's own element type throws a `DimensionMismatch`.

When `op` takes more than one argument, `operator_matrix(op)` constructs a
`FiniteDifferenceOperator` that generates the operator matrix. When `op` only
takes one argument, it instead constructs an `AbstractLazyOperator`, which is
internally converted into a `FiniteDifferenceOperator` when used in a broadcast
expression. Implementing `op_matrix` as a lazy operator adds an argument to the
expression `op_matrix.()`, from which the space and element type of the operator
matrix are inferred.

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
affine transformation, `operator_matrix(op)` prints a warning and zeros out the
boundary conditions before computing the operator matrix.

In addition to affine transformations, there are also some operators that apply
nonlinear transformations to their arguments; that is, transformations which
cannot be accurately approximated without using more terms of the form

```math
\\textrm{op}(\\textbf{0}) +
\\textrm{op}'(\\textbf{0}) * arg +
\\textrm{op}''(\\textbf{0}) * arg * arg +
\\ldots.
```

When `op` is such an operator, `operator_matrix(op)` throws an error.
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

Operators.fuses_into_stencils(::FDOperatorMatrix) = true

# The rows of a weighted interpolation read the weights at neighboring points.
# Every row reads its arguments at the same offsets, at indices clamped to the
# column, before any boundary condition is applied (see stencil_value).
Operators.reads_neighbors(
    ::FDOperatorMatrix{<:Operators.WeightedInterpolationOperator},
    _,
    _,
) = (Val(true), Val(false))
Operators.reads_in_lockstep(::FDOperatorMatrix) = true

Operators.return_space(op_matrix::FDOperatorMatrix, args...) =
    Operators.return_space(op_matrix.op, args...)

function Operators.return_eltype(op_matrix::FDOperatorMatrix, args...)
    if last(args) isa Spaces.AbstractSpace
        FT = Spaces.undertype(last(args))
    else
        FT = Geometry.undertype(eltype(last(args)))
    end
    return op_matrix_row_type(op_matrix.op, FT, Base.front(args)...)
end

# Simplified methods for when the operator matrix only depends on FT. The rows
# of such a matrix read no values, so they are recomputed at every point that
# reads them rather than cached (see Operators.recomputable).
op_matrix_row_type(op, ::Type{FT}, args...) where {FT} =
    typeof(op_matrix_interior_row(op, FT))
op_matrix_interior_row(op, space, idx, args...) =
    op_matrix_interior_row(op, Spaces.undertype(space))
Operators.recomputable_operator(op_matrix::FDOperatorMatrix) =
    op_matrix.op isa ConstantRowOperator

# Every row of an operator matrix is computed by the same code. The operator's
# interior row is evaluated at every index, reading local geometry and arguments
# at indices clamped to the column (see `lower_index`), so that all the
# points of a column read the arguments in lockstep (see
# `Operators.reads_in_lockstep`). The boundary conditions then transform the
# row's static band near the boundaries, without reading any arguments, by
# selecting (with `ifelse`) between entries based on the distances from idx to
# the ends of the column:
#  - With `Extrapolate`, the entries that lie outside of the column (the
#    coefficients of the ghost points of the row's stencil) are folded into the
#    in-range entries and zeroed (see `fold_ghosts`).
#  - In the boundary window of a condition (the indices where a pointwise
#    stencil uses its boundary stencil; see `Operators.left_interior_idx`),
#    which contains every row that reaches its ghost points, other conditions
#    replace the row (see `boundary_row`). On a column too short to separate the
#    two windows, the left (bottom) one takes precedence, as in
#    `Operators.stencil_value`.
#  - DivergenceF2C with `Extrapolate` replicates the row adjacent to the
#    boundary (see `op_matrix_row`).
# The rows that no transform affects are returned before the transforms are
# computed. The ghost points and windows only depend on the types of the
# operator and its boundary conditions (see `Operators.max_ghost_counts` and
# `Operators.right_window_width`), so this check is skipped at compile time when
# no row is affected. The rows of a periodic column are all interior rows. The
# last of `args` is the local geometry field of the operator's input space,
# which the row functions do not take.
Base.@propagate_inbounds function Operators.stencil_value(
    op_matrix::FDOperatorMatrix,
    space,
    idx,
    args...,
)
    op = op_matrix.op
    Row = Operators.return_eltype(op_matrix, args...)
    Topologies.isperiodic(space) && return convert(
        Row,
        op_matrix_interior_row(op, space, idx, Base.front(args)...),
    )
    row = op_matrix_row(Row, op, space, idx, Base.front(args)...)
    left_bc = Operators.get_boundary(op, Operators.left_boundary_window(space))
    right_bc = Operators.get_boundary(op, Operators.right_boundary_window(space))
    (ld, ud) = outer_diagonals(Row)
    (nghost_left, nghost_right) = Operators.ghost_counts(space, idx, ld, ud)
    in_left = Operators.in_window(
        idx - Operators.left_idx(space),
        Operators.left_window_width(space, op, left_bc, args...),
    )
    in_right =
        !in_left & Operators.in_window(
            Operators.right_idx(space) - idx,
            Operators.right_window_width(space, op, right_bc, args...),
        )
    (nghost_left > 0) | (nghost_right > 0) | in_left | in_right || return row

    (max_left, max_right) = Operators.max_ghost_counts(space, idx, ld, ud)
    # Every row of a column has at least one entry in range, and a row with at
    # most two entries can only reach ghost points on one side, so when both of
    # its sides are folded, both folds can be computed from the interior row.
    nin = max(ud - ld + 1 - nghost_left - nghost_right, 1)
    left_row = fold_ghosts(row, op, left_bc, nghost_left, max_left, nin, Val(true))
    row =
        ud - ld + 1 <= 2 && folds_ghosts(op, left_bc) && folds_ghosts(op, right_bc) ?
        select_row(
            nghost_left > 0,
            left_row,
            fold_ghosts(row, op, right_bc, nghost_right, max_right, nin, Val(false)),
        ) :
        fold_ghosts(left_row, op, right_bc, nghost_right, max_right, nin, Val(false))
    FT = Spaces.undertype(space)
    row = select_row(
        in_left,
        boundary_row(row, op, left_bc, FT, nghost_left, max_left, Val(true)),
        row,
    )
    return select_row(
        in_right,
        boundary_row(row, op, right_bc, FT, nghost_right, max_right, Val(false)),
        row,
    )
end

@inline select_row(condition, row1, row2) =
    map((entry1, entry2) -> ifelse(condition, entry1, entry2), row1, row2)

# The row of `op` at idx, converted to the row type of its operator matrix. This
# is the interior row, except for DivergenceF2C with Extrapolate, which
# replicates the interior output adjacent to each such boundary
# (D(v)[1] = D(v)[2]): its row at a boundary center is the interior row at the
# adjacent center, with band offsets shifted by one to make them relative to
# idx. The shifted rows fit in the row type, which `extrapolate_row_type` widens
# by one diagonal on each side (the interior row is read at every index first).
Base.@propagate_inbounds op_matrix_row(
    ::Type{Row},
    op,
    space,
    idx,
    args...,
) where {Row} = convert(Row, op_matrix_interior_row(op, space, idx, args...))
Base.@propagate_inbounds function op_matrix_row(
    ::Type{Row},
    op::Operators.DivergenceF2C,
    space,
    idx,
) where {Row}
    uses_extrapolate(op) ||
        return convert(Row, op_matrix_interior_row(op, space, idx))
    row_idx = replicated_row_index(op, space, idx)
    row = op_matrix_interior_row(op, space, row_idx)
    row_idx == idx && return convert(Row, row)
    return select_row(
        row_idx - idx == 1,
        convert(Row, shift_row_band(row, Val(1))),
        convert(Row, shift_row_band(row, Val(-1))),
    )
end

# The index of the interior row of DivergenceF2C that its row at idx replicates:
# the adjacent center at a boundary center with Extrapolate, and idx elsewhere.
@inline function replicated_row_index(op::Operators.DivergenceF2C, space, idx)
    left_bc = Operators.get_boundary(op, Operators.left_boundary_window(space))
    right_bc = Operators.get_boundary(op, Operators.right_boundary_window(space))
    first_idx = Operators.left_idx(space) + (left_bc isa Operators.Extrapolate)
    last_idx = Operators.right_idx(space) - (right_bc isa Operators.Extrapolate)
    return Operators.column_index(space, clamp(idx, first_idx, last_idx))
end

# The row index and first factor from which a matrix-vector product computes its
# row at idx. When every condition of a DivergenceF2C matrix is an Extrapolate,
# each row of the product is the product's interior row at the replicated index,
# so it is computed there with the matrix of DivergenceF2C without conditions,
# whose rows have the two entries of the interior stencil rather than the four
# of the widened row type: the two extra entries are zeros, but a product with
# the widened rows reads the vector at both of them in every row (on CPUs, this
# evaluates a fused flux expression twice as many times). A zero entry only
# changes a row of the product when the vector has a value that is not finite
# there. Other conditions modify the rows near the boundaries (see
# boundary_row), so their matrices are multiplied as they are. A periodic column
# has no boundaries, so all of its rows are interior rows (see stencil_value).
@inline replicated_product_row(space, idx, matrix1) = (idx, matrix1)
@inline function replicated_product_row(
    space,
    idx,
    matrix1::Base.Broadcast.Broadcasted{
        Operators.StencilStyle,
        <:Any,
        <:FDOperatorMatrix{<:Operators.DivergenceF2C},
    },
)
    op = matrix1.f.op
    uses_extrapolate(op) &&
    unrolled_all(Base.Fix2(isa, Operators.Extrapolate), op.bcs) ||
        return (idx, matrix1)
    interior_op = Operators.DivergenceF2C()
    interior_matrix = Base.Broadcast.Broadcasted(
        matrix1.style,
        FDOperatorMatrix{typeof(interior_op)}(interior_op),
        matrix1.args,
        matrix1.axes,
    )
    row_idx =
        Topologies.isperiodic(space) ? idx : replicated_row_index(op, space, idx)
    return (row_idx, interior_matrix)
end

# Reinterprets a row computed at `idx - shift` as a row at `idx` by shifting
# its band offsets. The offsets are type-level constants, so the shift happens
# at compile time.
shift_row_band(row::BandMatrixRow{ld}, ::Val{shift}) where {ld, shift} =
    BandMatrixRow{ld + shift}(row.entries...)

widen_row_type(::Type{BandMatrixRow{ld, bw, T}}) where {ld, bw, T} =
    BandMatrixRow{ld - 1, bw + 2, T}
extrapolate_row_type(op, ::Type{Row}) where {Row <: BandMatrixRow} =
    uses_extrapolate(op) ? widen_row_type(Row) : Row

# An Extrapolate condition folds the coefficients of the ghost points of a row
# (its `nghost` entries that lie outside of the column, counted from the left
# boundary when `from_left` and from the right one otherwise) into its in-range
# entries: as for the ghost values of the pointwise advection stencils (see
# `Operators.advection_ghost_values`), every ghost point on one side takes the
# value extrapolated from the in-range points, so the row is multiplied (on the
# right) by the matrix `E` that expresses each stencil point in terms of the
# in-range points -- the identity for the in-range points, and the extrapolation
# weights for the ghost points. Since all ghost rows of `E` are equal, the
# product reduces to adding `sum(ghost coefficients) * weight[k]` to the k-th
# in-range entry, ordered from the boundary outwards. For example, at the
# bottom, the interior row of Upwind3rdOrderBiasedProductC2F,
# (-v³ - |v³|, 7v³ + 3|v³|, 7v³ - 3|v³|, -v³ + |v³|) / 12, becomes
# (0, 6v³ + 2|v³|, 7v³ - 3|v³|, -v³ + |v³|) / 12 at the face one in from the
# boundary with Extrapolate{0}, and (0, 5v³ + |v³|, 8v³ - 2|v³|, -v³ + |v³|) / 12
# with Extrapolate{1}; at a boundary face of InterpolateC2F, every order copies
# the closest input. The weights come from `extrapolate_weights` with the number
# of in-range entries, `nin`, which on 1- and 2-center columns also excludes the
# ghost points beyond the opposite boundary, so that the second fold (of the
# right boundary) extrapolates from the same points as the first.
#
# The rows of the other conditions only reach ghost points in their boundary
# windows, where `boundary_row` replaces them (and the multiply skips entries
# outside of the column), so they are left unchanged. DivergenceF2C's
# Extrapolate replicates an output rather than extrapolating its input, and its
# rows only reach ghost points on degenerate columns.
folds_ghosts(op, bc) = false
folds_ghosts(op, ::Operators.Extrapolate) = true
folds_ghosts(::Operators.DivergenceF2C, ::Operators.Extrapolate) = false
@inline fold_ghosts(row, op, bc, nghost, max_ghost, nin, from_left) =
    folds_ghosts(op, bc) ?
    clip_row(
        row,
        Operators.extrapolate_weights(bc, nin, eltype(eltype(row))),
        nghost,
        max_ghost,
        from_left,
    ) : row

# Zeroes the first `nghost` entries of a row from one of its ends, after folding
# them into the other entries with the extrapolation `weights` (unless they are
# `nothing`), where `nghost` is at most `max_ghost` (a compile-time constant).
# The entry at position `pos` from that end is only folded when
# 1 <= nghost < pos, so its folded value is selected from those static cases,
# which keeps the folds of rows with constant entries constant. Entries are
# selected with `ifelse` rather than multiplied by masks, so that a row with no
# ghost points is returned unchanged (adding a zero would turn a -0.0 entry into
# 0.0).
@inline function clip_row(
    row::BandMatrixRow{ld, bw},
    weights,
    nghost,
    max_ghost,
    ::Val{from_left},
) where {ld, bw, from_left}
    entries = row.entries
    z = zero(eltype(row))
    from_boundary(j) = from_left ? j : bw + 1 - j
    ghost_sum(n) =
        unrolled_sum(ntuple(k -> ifelse(from_boundary(k) <= n, entries[k], z), Val(bw)))
    clipped_entries = ntuple(Val(bw)) do j
        pos = from_boundary(j)
        folded = unrolled_reduce(
            ntuple(identity, Val(bw - 1));
            init = entries[j],
        ) do entry, n
            isnothing(weights) || !(n <= max_ghost && n < pos <= n + 3) ? entry :
            ifelse(
                nghost == n,
                entries[j] + ghost_sum(n) * weights[pos - n],
                entry,
            )
        end
        pos <= max_ghost ? ifelse(pos <= nghost, z, folded) : folded
    end
    return BandMatrixRow{ld}(clipped_entries...)
end

# The row of an operator matrix in the boundary window of a condition, given the
# clipped row. A missing condition gives `NaN`s, matching the `NaN` that the
# pointwise stencils produce (see `Operators.NullBoundaryCondition`); a multiply
# by that row makes only its own output `NaN`. A condition that fixes the
# operator's output at the boundary gives zeros: an operator matrix only captures
# the linear part of the operator, and the fixed value is a constant (see
# `has_affine_bc`). Such conditions only remain in the matrix through the
# explicit `operator_matrix(op)` API (`@. op_matrix() * arg`); the automatic
# conversion of `@. op(arg)` strips them and reapplies them with a
# SetBoundaryOperator instead (see `modifies_output`), leaving a missing
# condition in the matrix. GradientF2C and DivergenceF2C with SetValue are the
# exception (as with `modifies_input`): the condition fixes an input value, and
# the output still depends linearly on the adjacent interior input, so only the
# coefficient of the fixed boundary face is zeroed, like that of a ghost point.
# Every other condition leaves the row unchanged.
const ValueFixingBoundaryCondition = Union{
    Operators.SetValue,
    Operators.SetGradient,
    Operators.SetDivergence,
    Operators.SetCurl,
}
const InputFixingFDOperator =
    Union{Operators.GradientF2C, Operators.DivergenceF2C}

@inline boundary_row(row, op, bc, _, nghost, max_ghost, from_left) = row
@inline boundary_row(
    row,
    op,
    ::Operators.NullBoundaryCondition,
    ::Type{FT},
    nghost,
    max_ghost,
    from_left,
) where {FT} = convert(typeof(row), zero(row) * FT(NaN))
@inline boundary_row(
    row,
    op,
    ::ValueFixingBoundaryCondition,
    _,
    nghost,
    max_ghost,
    from_left,
) = zero(row)
@inline boundary_row(
    row,
    ::InputFixingFDOperator,
    ::Operators.SetValue,
    _,
    nghost,
    max_ghost,
    from_left,
) = clip_row(row, nothing, nghost + 1, max(max_ghost, 0) + 1, from_left)

################################################################################

# Additional aliases for CenterToFace or FaceToCenter matrix rows
"""
    LowerDiagonalMatrixRow{T}
    LowerDiagonalMatrixRow(entry)

Alias for `BandMatrixRow{-1 + half, 1, T}`: a row of a [`BandMatrixRow`](@ref) matrix
field with a single entry on the diagonal `-1/2`. Together with
`UpperDiagonalMatrixRow`, it is used for the one-sided rows of center-to-face and
face-to-center operator matrices, e.g. the rows generated by
[`operator_matrix`](@ref) for the biased interpolation operators.
"""
const LowerDiagonalMatrixRow = BandMatrixRow{-1 + half, 1}    # -0.5

"""
    UpperDiagonalMatrixRow{T}
    UpperDiagonalMatrixRow(entry)

Alias for `BandMatrixRow{half, 1, T}`: a row of a [`BandMatrixRow`](@ref) matrix
field with a single entry on the diagonal `+1/2`. Together with
`LowerDiagonalMatrixRow`, it is used for the one-sided rows of center-to-face and
face-to-center operator matrices, e.g. the rows generated by
[`operator_matrix`](@ref) for the biased interpolation operators.
"""
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

# Indices of the points below and above idx (at idx ∓ 1/2) that a row reads,
# clamped to the column: the faces adjacent to a center are always in the
# column, but the center below the bottom face and the center above the top
# face are not.
@inline lower_index(space, idx) =
    idx isa PlusHalf && !Topologies.isperiodic(space) ?
    max(idx - half, Operators.first_index(space, idx - half)) : idx - half
@inline upper_index(space, idx) =
    idx isa PlusHalf && !Topologies.isperiodic(space) ?
    min(idx + half, Operators.last_index(space, idx + half)) : idx + half

Base.@propagate_inbounds ct3_data(velocity, space, idx) =
    Geometry.contravariant3(
        Operators.column_value(velocity, space, idx),
        Geometry.LocalGeometry(space, idx),
    )

################################################################################

# Operators with the rows below that only depend on FT.
const ConstantRowOperator = Union{
    Operators.InterpolateC2F,
    Operators.InterpolateF2C,
    Operators.BottomBiasedC2F,
    Operators.BottomBiasedF2C,
    Operators.TopBiasedC2F,
    Operators.TopBiasedF2C,
    Operators.SetBoundaryOperator,
    Operators.GradientOperator,
}

op_matrix_interior_row(
    ::Union{Operators.InterpolateC2F, Operators.InterpolateF2C},
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(FT(1), FT(1)) / 2

op_matrix_interior_row(
    ::Union{Operators.BottomBiasedC2F, Operators.BottomBiasedF2C},
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(true)

op_matrix_interior_row(
    ::Union{Operators.TopBiasedC2F, Operators.TopBiasedF2C},
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
    weight,
)
    idx⁻ = lower_index(space, idx)
    idx⁺ = upper_index(space, idx)
    w⁻ = Operators.column_value(weight, space, idx⁻)
    w⁺ = Operators.column_value(weight, space, idx⁺)
    # At a boundary face, both points of the stencil are clamped to the same
    # point, so their weights are equal whatever their value, and the boundary
    # row only depends on the boundary conditions. The weights are read at every
    # face (see `Operators.stencil_value`), but only divided where they differ.
    idx⁻ == idx⁺ && return BidiagonalMatrixRow(one(w⁻), one(w⁺)) / 2
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
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx))
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
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx))
    av³ = CT3(abs(v³.u³))
    return QuaddiagonalMatrixRow(-v³ - av³, 7v³ + 3av³, 7v³ - 3av³, -v³ + av³) /
           12
end

op_matrix_interior_row(::Operators.SetBoundaryOperator, ::Type{FT}) where {FT} =
    DiagonalMatrixRow(true)

op_matrix_row_type(::Operators.GradientOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow{C3{FT}}
op_matrix_interior_row(::Operators.GradientOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow(-C3(FT(1)), C3(FT(1)))

op_matrix_row_type(op::Operators.DivergenceOperator, ::Type{FT}) where {FT} =
    extrapolate_row_type(op, BidiagonalMatrixRow{C3Cov{FT}})
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.DivergenceOperator,
    space,
    idx,
)
    invJ = Geometry.LocalGeometry(space, idx).invJ
    J⁻ = Geometry.LocalGeometry(space, lower_index(space, idx)).J
    J⁺ = Geometry.LocalGeometry(space, upper_index(space, idx)).J
    return BidiagonalMatrixRow(-C3(J⁻)', C3(J⁺)') * invJ
end

op_matrix_row_type(
    ::Operators.CurlFiniteDifferenceOperator,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow{CT12_CT12{FT}}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.CurlFiniteDifferenceOperator,
    space,
    idx,
)
    invJ = Geometry.LocalGeometry(space, idx).invJ
    return BidiagonalMatrixRow(-εⁱʲ, εⁱʲ) * invJ
end
