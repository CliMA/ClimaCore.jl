# Build an implicit vertical solver

An implicit vertical step needs the Jacobian of the implicit tendency, one
banded matrix per column, and a solver for it. `ClimaCore.MatrixFields`
represents such matrices as fields of matrix rows, builds them from the
finite-difference operators, and solves the resulting block systems.

## Prerequisites

A tendency split into explicit and implicit parts for ClimaTimeSteppers
([Time-step with ClimaTimeSteppers](timestepping.md)).

## Steps

 1. Declare the structure of the Jacobian as a `FieldMatrix`: one entry per
    pair of state components, each a field of `BandMatrixRow`s of the right
    bandwidth (`DiagonalMatrixRow`, `BidiagonalMatrixRow`,
    `TridiagonalMatrixRow`, `QuaddiagonalMatrixRow`, `PentadiagonalMatrixRow`).
    Components are addressed by `@name`:

    ```julia
    import ClimaCore.MatrixFields
    import ClimaCore.MatrixFields: @name, ⋅
    jacobian = MatrixFields.FieldMatrix(
        (@name(c.ρ), @name(f.u₃)) => similar(Y.c.ρ, MatrixFields.BidiagonalMatrixRow{FT}),
        (@name(f.u₃), @name(c.ρ)) => similar(Y.f.u₃, MatrixFields.BidiagonalMatrixRow{FT}),
        (@name(f.u₃), @name(f.u₃)) =>
            similar(Y.f.u₃, MatrixFields.TridiagonalMatrixRow{FT}),
    )
    ```

    An entry that is a multiple of the identity is written as
    `(@name(c.ρ), @name(c.ρ)) => FT(-1) * LinearAlgebra.I`, with no field
    storage.

 2. Fill the entries from operator matrices. `MatrixFields.operator_matrix(op)`
    is the banded matrix of a finite-difference operator, and products of
    operator matrices (`⋅`) and of matrices with fields are banded matrices
    too, so the Jacobian of a composed stencil is written as the same
    composition:

    ```julia
    divᵥ_matrix = MatrixFields.operator_matrix(divᵥ)
    gradᵥ_matrix = MatrixFields.operator_matrix(gradᵥ)
    function Wfact(W, Y, p, dtγ, t)
        @. W.matrix[@name(c.ρ), @name(c.ρ)] =
            dtγ * κ * divᵥ_matrix() ⋅ gradᵥ_matrix() - (LinearAlgebra.I,)
        return nothing
    end
    ```

 3. Wrap the Jacobian in a solver and hand it to the time stepper.
    `FieldMatrixWithSolver(jacobian, Y)` chooses a block solver from the
    matrix structure; `MatrixFields.FieldMatrixSolverAlgorithm`s
    (`BlockDiagonalSolve`, `BlockLowerTriangularSolve`, `SchurComplementSolve`,
    `ApproximateBlockArrowheadIterativeSolve`, …) can be chosen explicitly for
    coupled systems:

    ```julia
    T_imp = CTS.ODEFunction(
        T_imp!;
        jac_prototype = MatrixFields.FieldMatrixWithSolver(jacobian, Y),
        Wfact,
    )
    ```

[Three dimensions on the cubed sphere](../tutorials/extruded_sphere.md) runs
this pattern for vertical diffusion. The [MatrixFields](../reference/matrix_fields.md)
reference page documents the row types, the indexing rules for `FieldMatrix`,
and the solver algorithms, and ClimaAtmos's
[implicit solver page](https://clima.github.io/ClimaAtmos.jl/stable/implicit_solver/)
shows the full atmospheric Jacobian assembled this way.
