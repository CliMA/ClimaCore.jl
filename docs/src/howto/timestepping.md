# Time-step with ClimaTimeSteppers

ClimaCore fields and field vectors are the state vectors of
[ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/stable/),
which provides explicit, implicit-explicit (IMEX), and Rosenbrock Runge–Kutta
methods. This page shows how a ClimaCore tendency plugs in; the algorithms and
their properties are ClimaTimeSteppers's documentation.

## Prerequisites

`import ClimaTimeSteppers as CTS` in a project that has both packages.

## Steps

 1. Put the state in a `Field` (for one variable or a `NamedTuple` of them on
    one space) or a `Fields.FieldVector` (for variables on different spaces,
    e.g. centers and faces):

    ```julia
    Y = Fields.FieldVector(; c = center_state, f = face_state)
    ```

 2. Write the tendency in place, with the signature `(∂ₜY, Y, p, t)`, where
    `p` holds parameters and caches. Complete spectral-element results by DSS
    or with a `tendency_completion` inside the tendency, or pass a `dss!`
    callback to the stepper (step 4).

    ```julia
    function T_exp!(∂ₜY, Y, p, t)
        @. ∂ₜY.c = -wdiv(flux(Y.c, p))
        Operators.complete_tendency!(p.completion, ∂ₜY.c, Y.c, p)
        return nothing
    end
    ```

 3. For an explicit method, wrap the tendency in a `ClimaODEFunction` and
    solve:

    ```julia
    prob = CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp!), Y, (t0, t_end), p)
    sol = CTS.solve(prob, CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()); dt, saveat)
    ```

 4. For an IMEX method, split the tendency into `T_exp!` and `T_imp!`, and give
    the implicit part a Jacobian. The Jacobian of a column-local implicit
    tendency is a banded matrix per column; `MatrixFields` stores it as a field
    of matrix rows and `FieldMatrixWithSolver` solves it
    ([Build an implicit vertical solver](matrix_fields.md)):

    ```julia
    T_imp = CTS.ODEFunction(
        T_imp!;
        jac_prototype = MatrixFields.FieldMatrixWithSolver(jacobian, Y),
        Wfact,   # Wfact(W, Y, p, dtγ, t) fills W = dtγ ∂T_imp/∂Y − I
    )
    dss!(Y, p, t) = Spaces.weighted_dss!(Y.c)
    prob = CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp!, T_imp!, dss!), Y, (t0, t_end), p)
    integrator =
        CTS.init(prob, CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 1)); dt)
    CTS.solve!(integrator)
    ```

    `ARS343` is the third-order additive scheme of [Ascher97a](@cite) that the
    atmosphere uses; with only the acoustic and gravity-wave terms implicit
    one Newton iteration per stage suffices [Gardner18a](@cite). The `dss!`
    callback runs after each stage so that the state is continuous when the
    next stage evaluates it.

 5. Save output through `saveat`, or step manually with `CTS.step!(integrator)`
    and inspect `integrator.u` between steps.

## Where time stepping is demonstrated

  - [Solve a column PDE](../tutorials/column_heat.md): explicit, one field.
  - [Shallow water on a plane](../tutorials/shallow_water_plane.md): explicit,
    `NamedTuple` state, DSS inside the tendency.
  - [Three dimensions on the cubed sphere](../tutorials/extruded_sphere.md):
    implicit vertical diffusion with a `MatrixFields` Jacobian and a `dss!`
    callback.
  - [`examples/hybrid/driver.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/driver.jl) with [`examples/hybrid/ode_config.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/ode_config.jl): the IMEX
    configuration of the staggered nonhydrostatic model, including the
    Jacobian in [`examples/hybrid/implicit_equation_jacobian.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/implicit_equation_jacobian.jl).
