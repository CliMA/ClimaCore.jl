using MPI

include("inertial_gravity_wave_utils.jl")

function run_solver(;
    velem,
    helem,
    npoly,
    tspan,
    𝔼_var,
    𝕄_var,
    J_𝕄ρ_overwrite,
    ode_algorithm,
    is_imex,
    max_iter,
    kwargs...,
)
    prob = inertial_gravity_wave_prob(;
        velem,
        helem,
        npoly,
        is_large_domain = true,
        tspan,
        𝔼_var,
        𝕄_var,
        J_𝕄ρ_overwrite,
        ode_algorithm,
        is_imex,
    )
    local code
    naccept = nf = nf2 = nsolve = nnonliniter = -1
    err_ρ = err_ρθ = err_ρuₕ = NaN
    try
        alg_args = (;)
        if ode_algorithm <: Union{
            OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
            OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm
        }
            alg_args = (; alg_args..., linsolve = linsolve!)
            if ode_algorithm <: Union{
                OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
                OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm
            }
                alg_args = (; alg_args, nlsolve = NLNewton(; max_iter))
            end
        end
        sol = solve(
            prob,
            ode_algorithm(; alg_args...);
            adaptive = false,
            save_on = false,
            verbose = false,
            kwargs...,
        )
        @unpack naccept, nf, nf2, nsolve, nnonliniter = sol.destats
        code = sol.retcode
        if code == :Success
            err = (sol.u[2] .- sol.u[1]) ./ abs.(sol.u[1])
            err_ρ = sum(err.Yc.ρ)
            err_ρuₕ = sum(err.Yc.ρuₕ)[1]
            try
                ρθ1 = get_ρθ(sol.u[1], sol.prob.p)
                ρθ2 = get_ρθ(sol.u[2], sol.prob.p)
                err_ρθ = sum((ρθ2 .- ρθ1) ./ abs.(ρθ1))
            catch
                code = :FalseSuccess
            end
        end
    catch
        code = :Error
    end
    return (code, naccept, nf, nf2, nsolve, nnonliniter, err_ρ, err_ρθ, err_ρuₕ)
end

function get_outputs(inputs, comm, tag, num_workers)
    num_inputs = length(inputs)
    outputs = Array{Any}(undef, num_inputs)
    current_worker_assignments = zeros(Int, num_workers)
    num_inputs_sent = 0
    num_outputs_recieved = 0

    initial_time = time()
    println("Progress:   0%; Time Elapsed =     0 seconds")
    last_printed_progress = 0

    while num_outputs_recieved < num_inputs
        for worker_index in 1:num_workers
            # If an output has been receieved from this worker, record it.
            if MPI.Iprobe(worker_index, tag, comm)[1]
                outputs[current_worker_assignments[worker_index]] =
                    MPI.recv(worker_index, tag, comm)[1]
                num_outputs_recieved += 1
                current_worker_assignments[worker_index] = 0
            end
            # If this worker is free, send it the next input, if there is one.
            if (
                current_worker_assignments[worker_index] == 0 &&
                num_inputs_sent < num_inputs
            )
                num_inputs_sent += 1
                MPI.send(inputs[num_inputs_sent], worker_index, tag, comm)
                current_worker_assignments[worker_index] = num_inputs_sent
            end
        end
        
        progress = 100 * num_outputs_recieved ÷ num_inputs
        if progress > last_printed_progress
            p_str = lpad(progress, 3)
            t_str = lpad(round(Int, time() - initial_time), 5)
            println("Progress: $p_str%; Time Elapsed = $t_str seconds")
            last_printed_progress = progress
        end

        if time() - initial_time > 60. * 60. * 4.
            println("This is taking too long. I give up!")
            for output_index in 1:num_inputs
                if !isassigned(outputs, output_index)
                    outputs[output_index] =
                        (:Timeout, -1, -1, -1, -1, -1, NaN, NaN, NaN)
                end
            end
            break
        end
    end

    return outputs
end

function compare_solvers()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_workers = MPI.Comm_size(comm) - 1
    root_index = 0
    tag = 0

    if rank == root_index
        problems = [(; velem) for velem in 10:10:30]
        explicit_formulations = [
            (; 𝔼_var = :ρθ, 𝕄_var = :ρw, J_𝕄ρ_overwrite = :none),
            (; 𝔼_var = :ρθ, 𝕄_var = :w, J_𝕄ρ_overwrite = :none),
            (; 𝔼_var = :ρe_tot, 𝕄_var = :ρw, J_𝕄ρ_overwrite = :none),
            (; 𝔼_var = :ρe_tot, 𝕄_var = :w, J_𝕄ρ_overwrite = :none),
        ]
        explicit_solver_infos = [
            (; ode_algorithm, dt) for
            dt in 4.:1.:10.,
            ode_algorithm in (
                # Explicit Runge-Kutta Methods
                Euler, Midpoint, Heun, Ralston, RK4, BS3, OwrenZen3, OwrenZen4,
                OwrenZen5, DP5, Tsit5, Anas5, FRK65, PFRK87, RKO65, TanYam7,
                DP8, TsitPap8, Feagin10, Feagin12, Feagin14, BS5, Vern6, Vern7,
                Vern8, Vern9,
                # Explicit Strong-Stability Preserving Runge-Kutta Methods for
                # Hyperbolic PDEs (Conservation Laws)
                SSPRK22, SSPRK33, SSPRK53, SSPRK63, SSPRK73, SSPRK83, SSPRK432,
                SSPRK43, SSPRK932, SSPRK54, SSPRK104, SSPRKMSVS32, SSPRKMSVS43,
            )
        ]
        # implicit_formulations = [
        #     (; 𝔼_var = :ρθ, 𝕄_var = :ρw, J_𝕄ρ_overwrite = :none),
        #     (; 𝔼_var = :ρθ, 𝕄_var = :w, J_𝕄ρ_overwrite = :none),
        #     (; 𝔼_var = :ρθ, 𝕄_var = :w, J_𝕄ρ_overwrite = :grav),
        #     (; 𝔼_var = :ρe_tot, 𝕄_var = :ρw, J_𝕄ρ_overwrite = :none),
        #     (; 𝔼_var = :ρe_tot, 𝕄_var = :ρw, J_𝕄ρ_overwrite = :grav),
        #     (; 𝔼_var = :ρe_tot, 𝕄_var = :w, J_𝕄ρ_overwrite = :none),
        #     (; 𝔼_var = :ρe_tot, 𝕄_var = :w, J_𝕄ρ_overwrite = :grav),
        #     (; 𝔼_var = :ρe_tot, 𝕄_var = :w, J_𝕄ρ_overwrite = :pres),
        # ]
        # rosenbrock_solver_infos = [
        #     (; ode_algorithm, dt) for
        #     dt in 8.:2.:30.,
        #     ode_algorithm in (
        #         # Rosenbrock Methods
        #         ROS3P, Rodas3, RosShamp4, Veldd4, Velds4, GRK4T, GRK4A,
        #         Ros4LStab, Rodas4, Rodas42, Rodas4P, Rodas4P2, Rodas5,
        #         # Rosenbrock-W Methods
        #         Rosenbrock23, Rosenbrock32, RosenbrockW6S4OS, ROS34PW1a,
        #         ROS34PW1b, ROS34PW2, ROS34PW3,
        #     )
        # ]
        variable_inputs = [
            (; problem..., formulation..., solver_info...) for
            solver_info in explicit_solver_infos, # rosenbrock_solver_infos,
            formulation in explicit_formulations, # implicit_formulations,
            problem in problems
        ]
        constant_input = (;
            helem = 75,
            npoly = 4,
            tspan = (0., 40000.),
            is_imex = false,
            reltol = 1e-3,
            abstol = 1e-6,
            max_iter = 100,
        )
        inputs = [
            (; variable_input..., constant_input...) for
            variable_input in variable_inputs
        ]
        outputs = get_outputs(inputs, comm, tag, num_workers)
        
        for worker_index in 1:num_workers
            MPI.send(nothing, worker_index, tag, comm)
        end
        
        open("comparison_results_rosenbrock_40000s_version2.csv", "w") do io
            variable_keys = keys(variable_inputs[1])
            println(io,
                join(variable_keys, ", "), ", ",
                "Return Code, ",
                "# of Timesteps, ",
                "# of Implicit Evals, ",
                "# of Non-Implicit Evals, ",
                "# of Linear Solves, ",
                "# of Nonlinear Iterations, ",
                "Relative error in ρ, ",
                "Relative error in ρθ, ",
                "Relative error in ρuₕ, ",
            )
            for (input, output) in zip(inputs, outputs)
                variable_input = [input[key] for key in variable_keys]
                println(io, join(variable_input, ", "), ", ", join(output, ", "))
            end
        end

        if any(output -> output[1] == :Timeout, outputs)
            error("Timeout")
        end
    else
        while true
            input = MPI.recv(root_index, tag, comm)[1]
            if isnothing(input)
                break
            end
            MPI.send(run_solver(; input...), root_index, tag, comm)
        end
    end

    MPI.Finalize()
end

compare_solvers()