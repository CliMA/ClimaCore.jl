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
    Operators.UpwindBiasedProductC2F,
    Operators.Upwind3rdOrderBiasedProductC2F,
    Operators.AdvectionC2C,
    Operators.FluxCorrectionC2C,
}
const TwoArgFDOperatorWithFaceInput = Union{
    Operators.WeightedInterpolateF2C,
    Operators.AdvectionF2F,
    Operators.FluxCorrectionF2F,
}
const ErroneousFDOperator = Union{
    Operators.LeftBiased3rdOrderC2F,
    Operators.LeftBiased3rdOrderF2C,
    Operators.RightBiased3rdOrderC2F,
    Operators.RightBiased3rdOrderF2C,
}
const NonlinearFDOperator = Union{Operators.FCTBorisBook, Operators.FCTZalesak}

const OneArgFDOperator =
    Union{OneArgFDOperatorWithCenterInput, OneArgFDOperatorWithFaceInput}
const TwoArgFDOperator =
    Union{TwoArgFDOperatorWithCenterInput, TwoArgFDOperatorWithFaceInput}

const FDOperatorWithCenterInput =
    Union{OneArgFDOperatorWithCenterInput, TwoArgFDOperatorWithCenterInput}
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

has_affine_bc(op) = any(
    bc ->
        bc isa Union{
            Operators.SetValue,
            Operators.SetGradient,
            Operators.SetDivergence,
            Operators.SetCurl,
        } && !iszero(bc.val),
    op.bcs,
)

uses_extrapolate(op) = unrolled_any(Base.Fix2(isa, Operators.Extrapolate), op.bcs)

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
    op_matrix::FDOperatorMatrix{<:TwoArgFDOperator},
    arg,
) = Base.Broadcast.broadcasted(
    op_matrix,
    arg,
    Fields.local_geometry_field(operator_input_space(op_matrix.op, axes(arg))),
)

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
operator_matrix(::O) where {O <: ErroneousFDOperator} = error(
    "$(O.name.name) always throws an AssertionError when it is used, so its \
     operator matrix has not been implemented",
)
operator_matrix(::O) where {O <: NonlinearFDOperator} = error(
    "$(O.name.name) applies a nonlinear transformation to its argument, so it \
     cannot be represented by a matrix",
)
operator_matrix(::O) where {O <: Operators.AbstractOperator} =
    error("operator_matrix has not been defined for $(O.name.name)")

################################################################################

Operators.has_boundary(
    op_matrix::FDOperatorMatrix,
    lbw::Operators.LeftBoundaryWindow{name},
) where {name} = Operators.has_boundary(op_matrix.op, lbw)
Operators.has_boundary(
    op_matrix::FDOperatorMatrix,
    rbw::Operators.RightBoundaryWindow{name},
) where {name} = Operators.has_boundary(op_matrix.op, rbw)

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
    FT = Geometry.undertype(eltype(args[end]))
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

################################################################################

# Additional aliases for CenterToFace or FaceToCenter matrix rows
const LowerEmptyMatrixRow = BandMatrixRow{-1 + half, 0}
const UpperEmptyMatrixRow = BandMatrixRow{half, 0}
const LowerDiagonalMatrixRow = BandMatrixRow{-1 + half, 1}    # -0.5
const UpperDiagonalMatrixRow = BandMatrixRow{half, 1}         #  0.5
const LowerBidiagonalMatrixRow = BandMatrixRow{-2 + half, 2}  # -1.5, -0.5
const UpperBidiagonalMatrixRow = BandMatrixRow{half, 2}       #  0.5,  1.5
const LowerTridiagonalMatrixRow = BandMatrixRow{-2 + half, 3} # -1.5, -0.5, 0.5
const UpperTridiagonalMatrixRow = BandMatrixRow{-1 + half, 3} # -0.5,  0.5, 1.5

# Additional aliases for Square matrix rows
const LowerBidiagonalSquareMatrixRow = BandMatrixRow{-1, 2} # -1, 0
const UpperBidiagonalSquareMatrixRow = BandMatrixRow{0, 2}  #  0, 1

const C3{T} = Geometry.Covariant3Vector{T}
const CT3{T} = Geometry.Contravariant3Vector{T}
const CT12_CT12{T} = Geometry.Axis2Tensor{
    T,
    Tuple{Geometry.Contravariant12Axis, Geometry.Contravariant12Axis},
    SMatrix{2, 2, T, 4},
}

# Levi-Civita symbol in 2D
const εⁱʲ = Geometry.AxisTensor(
    (Geometry.Contravariant12Axis(), Geometry.Contravariant12Axis()),
    SMatrix{2, 2}(0, 1, -1, 0),
)

Base.@propagate_inbounds ct3_data(velocity, space, idx, hidx) =
    Geometry.contravariant3(
        Operators.getidx(space, velocity, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )

################################################################################

op_matrix_interior_row(
    ::Union{Operators.InterpolateC2F, Operators.InterpolateF2C},
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(FT(1), FT(1)) / 2
op_matrix_first_row(
    ::Operators.InterpolateC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(0))
op_matrix_last_row(
    ::Operators.InterpolateC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(0))
op_matrix_first_row(
    ::Operators.InterpolateC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(1))
op_matrix_last_row(
    ::Operators.InterpolateC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(1))
op_matrix_first_row(
    ::Operators.InterpolateC2F,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(1))
op_matrix_last_row(
    ::Operators.InterpolateC2F,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(1))

op_matrix_interior_row(
    ::Union{Operators.LeftBiasedC2F, Operators.LeftBiasedF2C},
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(1))
op_matrix_first_row(
    ::Operators.LeftBiasedC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = LowerEmptyMatrixRow()
op_matrix_first_row(
    ::Operators.LeftBiasedF2C,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(0))

op_matrix_interior_row(
    ::Union{Operators.RightBiasedC2F, Operators.RightBiasedF2C},
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(1))
op_matrix_last_row(
    ::Operators.RightBiasedC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = UpperEmptyMatrixRow()
op_matrix_last_row(
    ::Operators.RightBiasedF2C,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(0))

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
op_matrix_first_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(0))
op_matrix_last_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(0))
op_matrix_first_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(1))
op_matrix_last_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(1))
op_matrix_first_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(FT(1))
op_matrix_last_row(
    ::Operators.WeightedInterpolateC2F,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(FT(1))

op_matrix_row_type(
    op::Operators.UpwindBiasedProductC2F,
    ::Type{FT},
    _,
) where {FT} =
    uses_extrapolate(op) ? QuaddiagonalMatrixRow{CT3{FT}} :
    BidiagonalMatrixRow{CT3{FT}}
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
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    return UpperDiagonalMatrixRow(v³ - av³) / 2
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    av³ = CT3(abs(v³.u³))
    return LowerDiagonalMatrixRow(v³ + av³) / 2
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx + 1, hidx))
    av³ = CT3(abs(v³.u³))
    return UpperBidiagonalMatrixRow(v³ + av³, v³ - av³) / 2
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.UpwindBiasedProductC2F,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx - 1, hidx))
    av³ = CT3(abs(v³.u³))
    return LowerBidiagonalMatrixRow(v³ + av³, v³ - av³) / 2
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
    return UpperTridiagonalMatrixRow(8v³ + 4av³, 5v³ - 5av³, -v³ + av³) / 12
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
    return LowerTridiagonalMatrixRow(-v³ - av³, 5v³ + 5av³, 8v³ - 4av³) / 12
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.Upwind3rdOrderBiasedProductC2F,
    ::Operators.ThirdOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
)
    v³ = CT3(ct3_data(velocity, space, idx, hidx))
    return UpperTridiagonalMatrixRow(4v³, 10v³, -2v³) / 12
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
    return LowerTridiagonalMatrixRow(-2v³, 10v³, 4v³) / 12
end

op_matrix_row_type(::Operators.AdvectionOperator, ::Type{FT}, _) where {FT} =
    TridiagonalMatrixRow{FT}
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.AdvectionC2C,
    space,
    idx,
    hidx,
    velocity,
)
    v³⁻_data = ct3_data(velocity, space, idx - half, hidx)
    v³⁺_data = ct3_data(velocity, space, idx + half, hidx)
    return TridiagonalMatrixRow(-v³⁻_data, v³⁻_data - v³⁺_data, v³⁺_data) / 2
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.AdvectionC2C,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
    velocity,
)
    v³⁻_data = ct3_data(velocity, space, idx - half, hidx)
    v³⁺_data = ct3_data(velocity, space, idx + half, hidx)
    return UpperBidiagonalSquareMatrixRow(2v³⁻_data - v³⁺_data, v³⁺_data) / 2
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.AdvectionC2C,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
    velocity,
)
    v³⁻_data = ct3_data(velocity, space, idx - half, hidx)
    v³⁺_data = ct3_data(velocity, space, idx + half, hidx)
    return LowerBidiagonalSquareMatrixRow(-v³⁻_data, v³⁻_data - 2v³⁺_data) / 2
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.AdvectionC2C,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    v³⁺_data = ct3_data(velocity, space, idx + half, hidx)
    return UpperBidiagonalSquareMatrixRow(-v³⁺_data, v³⁺_data)
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.AdvectionC2C,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    v³⁻_data = ct3_data(velocity, space, idx - half, hidx)
    return LowerBidiagonalSquareMatrixRow(-v³⁻_data, v³⁻_data)
end
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Operators.AdvectionF2F,
    space,
    idx,
    hidx,
    velocity,
)
    FT = Spaces.undertype(space)
    v³_data = ct3_data(velocity, space, idx, hidx)
    return TridiagonalMatrixRow(-v³_data, FT(0), v³_data) / 2
end
Base.@propagate_inbounds function op_matrix_interior_row(
    ::Union{Operators.FluxCorrectionC2C, Operators.FluxCorrectionF2F},
    space,
    idx,
    hidx,
    velocity,
)
    av³⁻_data = abs(ct3_data(velocity, space, idx - half, hidx))
    av³⁺_data = abs(ct3_data(velocity, space, idx + half, hidx))
    return TridiagonalMatrixRow(av³⁻_data, -av³⁻_data - av³⁺_data, av³⁺_data)
end
Base.@propagate_inbounds function op_matrix_first_row(
    ::Union{Operators.FluxCorrectionC2C, Operators.FluxCorrectionF2F},
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    av³⁺_data = abs(ct3_data(velocity, space, idx + half, hidx))
    return UpperBidiagonalSquareMatrixRow(-av³⁺_data, av³⁺_data)
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Union{Operators.FluxCorrectionC2C, Operators.FluxCorrectionF2F},
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
    velocity,
)
    av³⁻_data = abs(ct3_data(velocity, space, idx - half, hidx))
    return LowerBidiagonalSquareMatrixRow(av³⁻_data, -av³⁻_data)
end

op_matrix_interior_row(::Operators.SetBoundaryOperator, ::Type{FT}) where {FT} =
    DiagonalMatrixRow(FT(1))
op_matrix_first_row(
    ::Operators.SetBoundaryOperator,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = DiagonalMatrixRow(FT(0))
op_matrix_last_row(
    ::Operators.SetBoundaryOperator,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = DiagonalMatrixRow(FT(0))

op_matrix_row_type(op::Operators.GradientOperator, ::Type{FT}) where {FT} =
    uses_extrapolate(op) ? QuaddiagonalMatrixRow{C3{FT}} :
    BidiagonalMatrixRow{C3{FT}}
op_matrix_interior_row(::Operators.GradientOperator, ::Type{FT}) where {FT} =
    BidiagonalMatrixRow(-C3(FT(1)), C3(FT(1)))
op_matrix_first_row(
    ::Operators.GradientC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(C3(FT(2)))
op_matrix_last_row(
    ::Operators.GradientC2F,
    ::Operators.SetValue,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(-C3(FT(2)))
op_matrix_first_row(
    ::Operators.GradientC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(C3(FT(0)))
op_matrix_last_row(
    ::Operators.GradientC2F,
    ::Operators.SetGradient,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(C3(FT(0)))
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
op_matrix_first_row(
    ::Operators.GradientF2C,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = UpperTridiagonalMatrixRow(C3(FT(0)), -C3(FT(1)), C3(FT(1)))
op_matrix_last_row(
    ::Operators.GradientF2C,
    ::Operators.Extrapolate,
    ::Type{FT},
) where {FT} = LowerTridiagonalMatrixRow(-C3(FT(1)), C3(FT(1)), C3(FT(0)))

op_matrix_row_type(op::Operators.DivergenceOperator, ::Type{FT}) where {FT} =
    uses_extrapolate(op) ? QuaddiagonalMatrixRow{Adjoint{FT, C3{FT}}} :
    BidiagonalMatrixRow{Adjoint{FT, C3{FT}}}
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
    ::Operators.DivergenceC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    J⁺ = Geometry.LocalGeometry(space, idx + half, hidx).J
    return UpperDiagonalMatrixRow(C3(J⁺)') * 2invJ
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.DivergenceC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    J⁻ = Geometry.LocalGeometry(space, idx - half, hidx).J
    return LowerDiagonalMatrixRow(-C3(J⁻)') * 2invJ
end
op_matrix_first_row(
    ::Operators.DivergenceC2F,
    ::Operators.SetDivergence,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(C3(FT(0))')
op_matrix_last_row(
    ::Operators.DivergenceC2F,
    ::Operators.SetDivergence,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(C3(FT(0))')
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
op_matrix_first_row(
    ::Operators.DivergenceF2C,
    ::Operators.SetDivergence,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(C3(FT(0))', C3(FT(0))')
op_matrix_last_row(
    ::Operators.DivergenceF2C,
    ::Operators.SetDivergence,
    ::Type{FT},
) where {FT} = BidiagonalMatrixRow(C3(FT(0))', C3(FT(0))')
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.DivergenceF2C,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
)
    FT = Spaces.undertype(space)
    invJ = Geometry.LocalGeometry(space, idx + 1, hidx).invJ
    J⁻ = Geometry.LocalGeometry(space, idx + 1 - half, hidx).J
    J⁺ = Geometry.LocalGeometry(space, idx + 1 + half, hidx).J
    return UpperTridiagonalMatrixRow(C3(FT(0))', -C3(J⁻)', C3(J⁺)') * invJ
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.DivergenceF2C,
    ::Operators.Extrapolate,
    space,
    idx,
    hidx,
)
    FT = Spaces.undertype(space)
    invJ = Geometry.LocalGeometry(space, idx - 1, hidx).invJ
    J⁻ = Geometry.LocalGeometry(space, idx - 1 - half, hidx).J
    J⁺ = Geometry.LocalGeometry(space, idx - 1 + half, hidx).J
    return LowerTridiagonalMatrixRow(-C3(J⁻)', C3(J⁺)', C3(FT(0))') * invJ
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
Base.@propagate_inbounds function op_matrix_first_row(
    ::Operators.CurlC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    return UpperDiagonalMatrixRow(εⁱʲ) * 2invJ
end
Base.@propagate_inbounds function op_matrix_last_row(
    ::Operators.CurlC2F,
    ::Operators.SetValue,
    space,
    idx,
    hidx,
)
    invJ = Geometry.LocalGeometry(space, idx, hidx).invJ
    return LowerDiagonalMatrixRow(-εⁱʲ) * 2invJ
end
op_matrix_first_row(
    ::Operators.CurlC2F,
    ::Operators.SetCurl,
    ::Type{FT},
) where {FT} = UpperDiagonalMatrixRow(zero(CT12_CT12{FT}))
op_matrix_last_row(
    ::Operators.CurlC2F,
    ::Operators.SetCurl,
    ::Type{FT},
) where {FT} = LowerDiagonalMatrixRow(zero(CT12_CT12{FT}))
