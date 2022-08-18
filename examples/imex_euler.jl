Base.@kwdef struct IMEXEulerAlgorithm{L} <:
        ClimaTimeSteppers.DistributedODEAlgorithm
    linsolve::L
    num_newton_iters::Int = 2
    recompute_jac::Bool = true
end

struct IMEXEulerCache{U, W, L}
    u_exp::U
    b::U
    Δu::U
    W::W
    linsolve!::L
end

function ClimaTimeSteppers.cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXEulerAlgorithm;
    kwargs...
)
    u = prob.u0
    W = prob.f.f1.jac_prototype
    linsolve! = alg.linsolve(Val{:init}, W, u; kwargs...)
    return IMEXEulerCache(similar(u), similar(u), similar(u), W, linsolve!)
end

#=
Let u_n denote the value of u at time t_n. Let dt = t_{n+1} - t_n.
IMEX Euler algorithm:
    u_{n+1} = u_n + dt * (f_exp(u_n, t_n) + f_imp(u_{n+1}, t_{n+1})) ==>
    F(u_{n+1}) = u_exp, where F(u) = u - dt * f_imp(u, t_{n+1}) and u_exp = u_n + dt * f_exp(u_n, t_n)
Let u_{n+1,k} denote the value of u_{n+1} on the k-th Newton iteration, with u_{n+1,0} = u_exp.
Taylor series expansion of F around u_{n+1,k}, assuming no t-dependence:
    F(u_{n+1,k+1}) = F(u_{n+1,k}) + ∂F/∂u(u_{n+1,k}) * (u_{n+1,k+1} - u_{n+1,k}) =
        u_{n+1,k} - dt * f_imp(u_{n+1,k}, t_{n+1}) - W * (u_{n+1,k+1} - u_{n+1,k}),
    where W = -I + dt * J(u_{n+1,k}, t_{n+1})
Setting F(u_{n+1,k+1}) = u_exp:
    u_{n+1,k} - dt * f_imp(u_{n+1,k}, t_{n+1}) - W * (u_{n+1,k+1} - u_{n+1,k}) = u_exp ==>
    u_{n+1,k+1} = u_{n+1,k} + W \ b, where b = u_{n+1,k} - dt * f_imp(u_{n+1,k}, t_{n+1}) - u_exp
=#
function ClimaTimeSteppers.step_u!(integrator, cache::IMEXEulerCache)
    (; u, p, t, dt) = integrator
    (; f1, f2) = integrator.prob.f
    (; num_newton_iters, recompute_jac) = integrator.alg
    (; u_exp, b, Δu, W, linsolve!) = cache

    @info "t = $t" # ClimaTimeSteppers doesn't support progress bars

    @assert f1 isa OrdinaryDiffEq.ODEFunction
    @assert f2 isa OrdinaryDiffEq.ODEFunction ||
        f2 isa ClimaTimeSteppers.ForwardEulerODEFunction

    if f2 isa OrdinaryDiffEq.ODEFunction
        f2(u_exp, u, p, t)
        @. u_exp = u + dt * u_exp # u_exp = u_n + dt * f_exp(u_n, t_n)
    else # f2 is a ForwardEulerODEFunction, which means it may apply limiters
        @. u_exp = u
        f2(u_exp, u, p, t, dt)
    end
    @. u = u_exp # u_{n+1,0} = u_exp
    for iter in 1:num_newton_iters
        if iter == 1 || recompute_jac
            f1.Wfact(W, u, p, dt, t + dt) # W = -I + dt * J(u_{n+1,k}, t_{n+1})
        end
        f1(b, u, p, t + dt)
        @. b = u - dt * b - u_exp # b = u_{n+1,k} - dt * f_imp(u_{n+1,k}, t_{n+1}) - u_exp
        linsolve!(Δu, W, b)
        @. u += Δu # u_{n+1,k+1} = u_{n+1,k} + W \ b
    end
end

function ClimaTimeSteppers.step_u!(int, cache::ClimaTimeSteppers.ARSCache{Nstages}) where {Nstages}

    f = int.prob.f
    f1! = f.f1
    f2! = f.f2
    Wfact! = f1!.Wfact

    u = int.u
    p = int.p
    t = int.t
    dt = int.dt

    tab = cache.tableau
    W = cache.W
    U = cache.U
    Uhat = cache.Uhat
    idu = cache.idu
    linsolve! = cache.linsolve!

    # Update W
    # Wfact!(W, u, p, dt*tab.γ, t)


    # implicit eqn:
    #   ux = u + dt * f(ux, p, t)
    # Newton iteration:
    #   ux <- ux + (I - dt J) \ (u + dt f(ux, p, t) - ux)
    # initial iteration
    #   ux <- u + dt (I - dt J) \ f(u, p, t)

    function implicit_step!(ux, u, p, t, dt)
        ux .= u
        for _ in 1:1
            Wfact!(W, ux, p, dt, t)
            f1!(idu, ux, p, t)
            @. idu = ux - dt * idu - u
            linsolve!(idu, W, idu)
            @. ux += idu
        end
    end

    #### stage 1
    # explicit
    Uhat[1] .= u # utilde[i],  Q0[1] == 1
    f2!(Uhat[1], u, p, t+dt*tab.chat[1], dt*tab.ahat[2,1])

    # implicit
    implicit_step!(U[1], Uhat[1], p, t+dt*tab.c[1], dt*tab.a[1,1])
    if Nstages == 1
        u .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
        f2!(u, U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])
        return
    end

    #### stage 2
    Uhat[2] .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
    f2!(Uhat[2], U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])

    implicit_step!(U[2], Uhat[2], p, t+dt*tab.c[2], dt*tab.a[2,2])

    if Nstages == 2
        u .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
        f2!(u, U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
        return
    end

    #### stage 3
    Uhat[3] .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
    f2!(Uhat[3], U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
    # @show Uhat[3] t+dt*tab.chat[3]

    implicit_step!(U[3], Uhat[3], p, t+dt*tab.c[3], dt*tab.a[3,3])
    # @show U[3] t+dt*tab.c[3]

    ### final update
    u .= tab.Q0[4] .* u .+
    tab.Qhat[4,1] .* Uhat[1] .+ tab.Q[4,1] .* U[1] .+
    tab.Qhat[4,2] .* Uhat[2] .+ tab.Q[4,2] .* U[2] .+
    tab.Qhat[4,3] .* Uhat[3] .+ tab.Q[4,3] .* U[3]

    # @show u
    f2!(u, U[3], p, t+dt*tab.chat[4], dt*tab.ahat[5,4])
    # @show u t+dt*tab.chat[4]
    return
end

#=
s-stage DIRK for ∂u/∂t = f(u):
u_{n+1} = u_n + k * ∑_{i=1}^s b_i * K_i, where, ∀ i ∈ 1:s,
    K_i = f(t_n + c_i * k, U_i) and U_i = u_n + k * ∑_{j=1}^i a_{i,j} * K_j
Note that, if b_i = a_{s,i} ∀ i ∈ 1:s, then u_{n+1} = U_s

Rewriting this more succinctly:
u_{n+1} = u_n + k * ∑_{i=1}^s b_i * f(t_n + c_i * k, U_i), where, ∀ i ∈ 1:s,
    U_i = u_n + k * ∑_{j=1}^i a_{i,j} * f(t_n + c_j * k, U_j)

Examples:
    Backward Euler:
        s = 1, a = [1], b = [1], c = [1] ==>
        u_{n+1} = U_1, where U_1 = u_n + k * f(t_n + k, U_1)

s,σ-stage IMEX DIRK for ∂u/∂t = f_exp(u) + f_imp(u), where σ = s + 1:
u_{n+1} = u_n + k * ∑_{i=1}^s b_i * K_i + k * ∑_{i=1}^{s+1} b̂_i * K̂_i, where
    K_i = f_imp(t_n + c_i * k, U_i) ∀ i ∈ 1:s,
    U_i = u_n + k * ∑_{j=1}^i (a_{i,j} * K_j + â_{i+1,j} * K̂_j) ∀ i ∈ 1:s,
    K̂_1 = f_exp(t_n, u_n), and K̂_i = f_exp(t_n + ĉ_i * k, U_{i-1}) ∀ i ∈ 2:s+1
Note that, if b̂_{s+1} = 0, b̂_i = â_{s+1,i} ∀ i ∈ 1:s+1, and b_i = a_{s,i} ∀ i ∈ 1:s, then u_{n+1} = U_s

Rewriting this more succinctly:
Let U_0 = u_n ==>
u_{n+1} = u_n + k * ∑_{i=1}^s b_i * f_imp(t_n + c_i * k, U_i) + k * ∑_{i=1}^{s+1} b̂_i * f_exp(t_n + ĉ_i * k, U_{i-1}), where
    U_i = u_n + k * ∑_{j=1}^i (a_{i,j} * f_imp(t_n + c_j * k, U_j) + â_{i+1,j} * f_exp(t_n + ĉ_j * k, U_{j-1})) ∀ i ∈ 1:s
    
An s-stage (DIRK) IMEX ARK method for ∂u/∂t = f_exp(u, t) + f_imp(u, t) is given by
    u_next := u + Δt * ∑_{χ∈(exp,imp)} ∑_{i=1}^s b_χ[i] * f_χ(U[i], t + Δt * c_χ[i]), where
    U[i] := u + Δt * ∑_{j=1}^{i-1} a_exp[i,j] * f_exp(U[j], t + Δt * c_exp[j]) +
            Δt * ∑_{j=1}^i a_imp[i,j] * f_imp(U[j], t + Δt * c_imp[j]) ∀ i ∈ 1:s
Here, u_next denotes the value of u(t) at t_next = t + Δt.
The values a_χ[i,j] are called the "internal coefficients", the b_χ[i] are called
the "weights", and the c_χ[i] are called the "abcissae".
The abscissae are often defined as c_χ[i] := ∑_{j=1}^s a_χ[i,j] for both the explicit
and implicit methods to be "internally consistent", with c_exp[i] = c_imp[i] for the
overall IMEX method to be "internally consistent", but this is not required.
To simplify our notation, let
    a_χ[s+1,i] := b_χ[i] ∀ i ∈ 1:s,
    F_χ[i] := f_χ(U[i], t + Δt * c_χ[i]) ∀ i ∈ 1:s, and
    Δu_χ[i,j] := Δt * a_χ[i,j] * F_χ[j] ∀ i ∈ 1:s+1, j ∈ 1:s
This allows us to rewrite our earlier definitions as
    u_next = u + ∑_{χ∈(exp,imp)} ∑_{i=1}^s Δu_χ[s+1,i], where
    U[i] = u + ∑_{j=1}^{i-1} Δu_exp[i,j] + ∑_{j=1}^i Δu_imp[i,j] ∀ i ∈ 1:s

We will now rewrite the algorithm so that we can express each value of F_χ in
terms of the first increment Δu_χ it is used to generate.
First, ∀ j ∈ 1:s, let
    I_χ[j] := min(i ∈ 1:s+1 ∣ a_χ[i,j] != 0)
Note that I_χ[j] is undefined if the j-th column of a_χ only contains zeros.
Also, note that I_imp[j] >= j and I_exp[j] > j ∀ j ∈ 1:s.
In addition, ∀ i ∈ 1:s+1, let
    S_χ[i] := [j ∈ 1:s ∣ I_χ[j] == i],
    Sᶜ_χ[i] := [j ∈ 1:s ∣ I_χ[j] < i], and
    N_χ[i] := length(S_χ[i])
We can then define, ∀ i ∈ 1:s+1,
    ũ[i] := u + ∑_{χ∈(exp,imp)} ∑_{j ∈ Sᶜ_χ[i]} Δu_χ[i,j] and
    Û_χ[i,k] := Û_χ[i,k-1] + Δu_χ[i,S_χ[i][k]] ∀ k ∈ 1:N_χ[i], where
        Û_exp[i,0] := ũ[i] and Û_imp[i,0] := Û_exp[i,N_exp[i]]
We then find that
    u_next = Û_imp[s+1,N_imp[s+1]] and U[i] = Û_imp[i,N_imp[i]] ∀ i ∈ 1:s
Next, ∀ j ∈ Sᶜ_χ[s+1] (or, more generally, ∀ j ∈ Sᶜ_χ[i] ∀ i ∈ 1:s+1, since
Sᶜ_χ[i-1] ⊂ Sᶜ_χ[i]), let
    K_χ[j] := k ∈ N_χ[I_χ[j]] | S_χ[I_χ[j]][k] == j
We then have that, ∀ j ∈ Sᶜ_χ[s+1],
    Û_χ[I_χ[j],K_χ[j]] = Û_χ[I_χ[j],K_χ[j]-1] + Δu_χ[I_χ[j],j]
Since a_χ[I_χ[j],j] != 0, this means that, ∀ j ∈ Sᶜ_χ[s+1],
    F_χ[j] = (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]) / (Δt * a_χ[I_χ[j],j])

Now, suppose that we want to modify this algorithm so that we can apply a
filter/limiter during the addition of the increments Δu_χ[i,S_χ[i][k]].
Specifically, instead of specifying f_χ(u, t), we want to specify g_χ(û, u, t, Δt)
and redefine, ∀ i ∈ 1:s+1 and ∀ k ∈ 1:N_χ[i],
    Û_χ[i,k] := g_χ(Û_χ[i,k-1], U[j], t + Δt * c_χ[j], Δt * a_χ[i,j], where j = S_χ[i][k]
Note that specifying g_χ(û, u, t, Δt) := û + Δt * f_χ(u, t) is equivalent to not
using any filters/limiters.
We can use our earlier expression to redefine F_χ[j] as, ∀ j ∈ Sᶜ_χ[s+1],
    F_χ[j] := (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]) / (Δt * a_χ[I_χ[j],j])
We then have that, ∀ i ∈ 1:s+1 and ∀ j ∈ 1:Sᶜ_χ[s+1],
    Δu_χ[i,j] = ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]), where
        ā_χ[i,j] = a_χ[i,j]/a_χ[I_χ[j],j]
We can then use these values of Δu_χ[i,j] to determine each value of ũ[i].

This procedure of computing the values of F_χ[j] and using those values to
compute ũ[i] is rather inefficient, and it would be better to directly use the
values of Û_χ to compute ũ[i].
From the previous section, we know that, ∀ i ∈ 1:s+1,
    ũ[i] = u +
           ∑_{χ∈(exp,imp)} ∑_{j ∈ Sᶜ_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
Now, ∀ i ∈ 1:s+1, let
    Sᶜ1_χ[i] := [j ∈ Sᶜ_χ[i] | K_χ[j] == 1] and
    Sᶜ2_χ[i] := [j ∈ Sᶜ_χ[i] | K_χ[j] > 1]
Since Û_exp[i,0] = ũ[i] and Û_imp[i,0] = Û_exp[i,N_exp[i]], we then have that
    ũ[i] = u +
           ∑_{j ∈ Sᶜ1_exp[i]} ā_exp[i,j] * (Û_exp[I_exp[j],1] - ũ[I_exp[j]]) +
           ∑_{j ∈ Sᶜ1_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - Û_exp[I_imp[j],N_exp[I_imp[j]]]) +
           ∑_{χ∈(exp,imp)} ∑_{j ∈ Sᶜ2_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
Next, ∀ i ∈ 1:s+1, let
    Sᶜ11_imp[i] := [j ∈ Sᶜ1_imp[i] | N_exp[I_imp[j]] == 0] and
    Sᶜ12_imp[i] := [j ∈ Sᶜ1_imp[i] | N_exp[I_imp[j]] > 0]
Since Û_exp[i,0] = ũ[i], this means that
    ũ[i] = u +
           ∑_{j ∈ Sᶜ1_exp[i]} ā_exp[i,j] * (Û_exp[I_exp[j],1] - ũ[I_exp[j]]) +
           ∑_{j ∈ Sᶜ11_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - ũ[I_imp[j]]) +
           ∑_{j ∈ Sᶜ12_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - Û_exp[I_imp[j],N_exp[I_imp[j]]]) +
           ∑_{χ∈(exp,imp)} ∑_{j ∈ Sᶜ2_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
We will now show that, ∀ i ∈ 1:s+1, there are some Q₀ and Q_χ such that
    ũ[i] = Q₀[i] * u + ∑_{χ∈(exp,imp)} ∑_{l=1}^{i-1} ∑_{k=1}^{N_χ[l]} Q_χ[i, l, k] * Û_χ[l, k]
First, we check the base case: ũ[1] = u, so that
    ũ[1] = Q₀[1] * u, where Q₀[1] = 1
Next, we apply the inductive step...
Is this too messy to do in the general case?

Given ũ[i], we can compute Û[i] and then U[i] by adding the increments
Δu_exp[i,i-1] and Δu_imp[i,i].
If we apply a filter/limiter to the result of an increment (or to the result of
a partial increment), the increment will be modified from Δu_χ to Δu_χ′; e.g.,
    Û[i] → L(ũ[i], ũ[i] + Δu_exp[i,i-1]) ⟹
    Û[i] → ũ[i] + Δu_exp′[i,i-1], where
    Δu_exp′[i,i-1] := L(ũ[i], ũ[i] + Δu_exp[i,i-1]) - ũ[i]
We can use the modified increment to define the modified tendency f_χ′:
    Δu_exp′[i,i-1] = Δt * a_exp[i,i-1] * f_exp′(U[i-1], t + Δt * c_exp[i-1]), where
    f_exp′(U[i-1], t + Δt * c_exp[i-1]) :=
        = (L(ũ[i], ũ[i] + Δu_exp[i,i-1]) - ũ[i]) / (Δt * a_exp[i,i-1])
This procedure lets us define every modified tendency f_χ′, which in turn gives us
every remaining increment Δu_χ′.
However, it is somewhat inefficient to compute f_χ′ from the output of the
filter/limiter, and it would require fewer operations to directly use the output
of the filter/limiter to compute ũ[i].
Therefore, our goal is to express ũ[i] as a linear combination of u and the
values of U[i] and Û[i].

From the definition of Δu_χ[i,j], we have that, ∀ i ∈ 1:s+1
    a_imp[j,j] * Δu_imp[i,j] =
        = Δt * a_imp[i,j] * a_imp[j,j] * f_imp(U[j], t + Δt * c_imp[j]) =
        = a_imp[i,j] * Δu_imp[j,j] ∀ j ∈ 1:s and
    a_exp[j,j-1] * Δu_exp[i,j-1] =
        = Δt * a_exp[i,j-1] * a_exp[j,j-1] * f_exp(U[j-1], t + Δt * c_exp[j-1]) =
        = a_exp[i,j-1] * Δu_exp[j,j-1] ∀ j ∈ 2:s+1
So, ∀ i ∈ 1:s+1,
    a_imp[j,j] != 0 ⟹ Δu_imp[i,j] = a_imp[i,j]/a_imp[j,j] * Δu_imp[j,j] ∀ j ∈ 1:s and
    a_exp[j,j-1] != 0 ⟹ Δu_exp[i,j-1] = a_exp[i,j-1]/a_exp[j,j-1] * Δu_exp[j,j-1] ∀ j ∈ 2:s+1
Moreover, ∀ i ∈ 1:s+1
    a_imp[i,j] == 0 ⟹ Δu_imp[i,j] = 0 = 0 * Δu_imp[j,j] ∀ j ∈ 1:s and
    a_exp[i,j-1] == 0 ⟹ Δu_exp[i,j-1] = 0 = 0 * Δu_exp[j,j-1] ∀ j ∈ 2:s+1
From our last expressions for Û[i] and U[i], we find that
    Δu_imp[j,j] = U[j] - Û[j] ∀ j ∈ 1:s and Δu_exp[j,j-1] = Û[j] - ũ[j] ∀ j ∈ 2:s+1
Taken together, the last three sets of expressions tell us that, ∀ i ∈ 1:s+1,
    a_imp[i,j] == 0 || a_imp[j,j] != 0 ⟹ Δu_imp[i,j] = ā_imp[i,j] * (U[j] - Û[j]), where
    ā_imp[i,j] := (a_imp[i,j] == 0) ? 0 : a_imp[i,j]/a_imp[j,j] ∀ j ∈ 1:s, and
    a_exp[i,j-1] == 0 || a_exp[j,j-1] != 0 ⟹ Δu_exp[i,j-1] = ā_exp[i,j-1] * (Û[j] - ũ[j]), where
    ā_exp[i,j-1] := (a_exp[i,j-1] == 0) ? 0 : a_exp[i,j-1]/a_exp[j,j-1] ∀ j ∈ 2:s+1

We will now show that, if there are no values of i and j for which
a_imp[j,j] == 0 && a_imp[i,j] != 0 or a_exp[j+1,j] == 0 && a_exp[i,j] != 0, then
we can express ũ[i] in terms of u, U[i], and Û[i] as
    ũ[i] = Q₀[i] * u + ∑_{j=1}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j]) ∀ i ∈ 2:s+1
Since there are no such values of i and j, we can use our formulas for Δu_χ[i,j]
to express ũ[i] as
    ũ[i] = u + ∑_{j=1}^{i-1} ā_imp[i,j] * (U[j] - Û[j]) +
           ∑_{j=2}^{i-1} ā_exp[i,j-1] * (Û[j] - ũ[j]) =
         = u - ā_imp[i,1] * Û[1] + ∑_{j=1}^{i-1} ā_imp[i,j] * U[j] +
           ∑_{j=2}^{i-1} (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j] -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] * ũ[j]
First, we check the base case:
    ũ[2] = u + ā_imp[2,1] * U[1] - ā_imp[2,1] * Û[1]
This shows that
    ũ[2] = Q₀[2] * u + Q[2,1] * U[1] + Q̂[2,1] * Û[1], where
    Q₀[2] := 1, Q[2,1] := ā_imp[2,1], and Q̂[2,1] := -ā_imp[2,1]
Next, we apply the inductive step: suppose that
    ũ[j] = Q₀[j] * u + ∑_{k=1}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) ∀ j ∈ 2:i-1
Substituting this into the last expression for ũ[i] gives us
    ũ[i] = u - ā_imp[i,1] * Û[1] + ∑_{j=1}^{i-1} ā_imp[i,j] * U[j] +
           ∑_{j=2}^{i-1} (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j] -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] *
               (Q₀[j] * u + ∑_{k=1}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u - ā_imp[i,1] * Û[1] + 
           ∑_{j=1}^{i-1} ā_imp[i,j] * U[j] +
           ∑_{j=2}^{i-1} (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j] -
           ∑_{j=2}^{i-1} ∑_{k=1}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
For any values of s[j,k], we have that
    ∑_{j=2}^{i-1} ∑_{k=1}^{j-1} s[j,k] = ∑_{k=2}^{i-1} ∑_{j=1}^{k-1} s[k,j] =
        = ∑_{j=1}^{i-2} ∑_{k=j+1}^{i-1} s[k,j] = ∑_{j=1}^{i-1} ∑_{k=j+1}^{i-1} s[k,j]
Using this to rewrite the last sum in the previous expression gives us
    ũ[i] = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u - ā_imp[i,1] * Û[1] + 
           ∑_{j=1}^{i-1} ā_imp[i,j] * U[j] +
           ∑_{j=2}^{i-1} (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j] -
           ∑_{j=1}^{i-1} ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           (-ā_imp[i,1] - ∑_{k=2}^{i-1} ā_exp[i,k-1] * Q̂[k,1]) * Û[1] + 
           ∑_{j=1}^{i-1} (ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q[k,j]) * U[j] +
           ∑_{j=2}^{i-1} (
               ā_exp[i,j-1] - ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j]
           ) * Û[j]
This shows that, ∀ i ∈ 2:s+1,
    ũ[i] = Q₀[i] * u + ∑_{j=1}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j]), where
    Q₀[i] := 1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j],
    Q[i,j] := ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q[k,j] ∀ j ∈ 1:i-1,
    Q̂[i,1] := -ā_imp[i,1] - ∑_{k=2}^{i-1} ā_exp[i,k-1] * Q̂[k,1], and
    Q̂[i,j] := ā_exp[i,j-1] - ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j] ∀ j ∈ 2:i-1

If there are values of i and j which break the aforementioned condition, this
procedure must be modified.
For now, we will only consider the case where a_imp[1,1] == 0 && a_imp[i,1] != 0
for some i ∈ 2:s+1, which corresponds to an EDIRK implicit method.
Let i_imp1 be the smallest value of i for which a_imp[i,1] != 0, which is also
the smallest value of i for which Δu_imp[i,1] != 0, so that
    ũ[i] = u + ∑_{j=2}^{i-1} Δu_imp[i,j] + ∑_{j=2}^{i-1} Δu_exp[i,j-1] ∀ i ∈ 2:i_imp1-1 and
    ũ[i] = u + ∑_{j=1}^{i-1} Δu_imp[i,j] + ∑_{j=2}^{i-1} Δu_exp[i,j-1] ∀ i ∈ i_imp1:s+1
Let
    u_imp1 := u + ∑_{j=2}^{i_imp1-1} Δu_imp[i_imp1,j] + ∑_{j=2}^{i_imp1-1} Δu_exp[i_imp1,j-1]
We then have that
    ũ[i_imp1] = u_imp1 + Δu_imp[i_imp1,1]
Since a_imp[i_imp1,1] != 0, this means that
    F_imp[1] = (ũ[i_imp1] - u_imp1) / (Δt * a_imp[i_imp1,1])
This tells us that
    Δt * a_imp[i,1] * F_imp[1] = ȧ_imp[i,1] * (ũ[i_imp1] - u_imp1), where
    ȧ_imp[i,1] := a_imp[i,1]/a_imp[i_imp1,1]
We will now show that, if there are no other values of i and j for which
a_imp[j,j] == 0 && a_imp[i,j] != 0 or a_exp[j+1,j] == 0 && a_exp[i,j] != 0, then
we can express ũ[i] in terms of u, ũ[i_imp1], U[i], and Û[i] as
    ũ[i] = Q₀[i] * u + ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j])
    ∀ i ∈ 2:i_imp1-1 and
    ũ[i] = Q₀[i] * u + Q_imp1[i] * ũ[i_imp1] +
           ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j])
    ∀ i ∈ i_imp1+1:s+1
Since there are no such values of i and j, we can use our formulas for
Δt * a_χ[i,j] * F_χ[j] to express u_imp1 as
    u_imp1 = u + Δt * ∑_{j=2}^{i_imp1-1} a_imp[i_imp1,j] * F_imp[j] +
             Δt * ∑_{j=1}^{i_imp1-2} a_exp[i_imp1,j] * F_exp[j]



    u_imp1 = u + ∑_{j=2}^{i_imp1-1} ā_imp[i_imp1,j] * (U[j] - Û[j]) +
             ∑_{j=1}^{i_imp1-2} ā_exp[i_imp1,j] * (Û[j+1] - ũ[j+1]) =
           = u + ∑_{j=2}^{i_imp1-1} (
                 ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
             ) -
             ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * ũ[j]
We can also use these formulas to express ũ[i] as
    ũ[i] = u + ∑_{j=2}^{i-1} ā_imp[i,j] * (U[j] - Û[j]) +
           ∑_{j=1}^{i-2} ā_exp[i,j] * (Û[j+1] - ũ[j+1]) =
         = u +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] * ũ[j]
    ∀ i ∈ 2:i_imp1-1 and
    ũ[i] = u + ȧ_imp[i,1] * (ũ[i_imp1] - u_imp1) +
           ∑_{j=2}^{i-1} ā_imp[i,j] * (U[j] - Û[j]) +
           ∑_{j=1}^{i-2} ā_exp[i,j] * (Û[j+1] - ũ[j+1]) =
         = u + ȧ_imp[i,1] * ũ[i_imp1] - ȧ_imp[i,1] * (
               u + ∑_{j=2}^{i_imp1-1} (
                   ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
               ) -
               ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * ũ[j]
           ) +
           ∑_{j=2}^{i-1} ā_imp[i,j] * (U[j] - Û[j]) +
           ∑_{j=1}^{i-2} ā_exp[i,j] * (Û[j+1] - ũ[j+1]) =
         = (1 - ȧ_imp[i,1]) * u + ȧ_imp[i,1] * ũ[i_imp1] -
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} (
               ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
           ) +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) +
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * ũ[j] -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] * ũ[j]
    ∀ i ∈ i_imp1+1:s+1
The base case is the same as before, so we will skip to the inductive step:
    ũ[i] = u +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] *
               (Q₀[j] * u + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
    ∀ i ∈ 2:i_imp1-1 and
    ũ[i] = (1 - ȧ_imp[i,1]) * u + ȧ_imp[i,1] * ũ[i_imp1] -
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} (
               ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
           ) +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) +
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] *
               (Q₀[j] * u + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])) -
           ∑_{j=2}^{i_imp1-1} ā_exp[i,j-1] *
               (Q₀[j] * u + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])) -
           ā_exp[i,i_imp1-1] * ũ[i_imp1] -
           ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * (
               Q₀[j] * u + Q_imp1[j] * ũ[i_imp1] +
               ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
           ) =
         = (
               1 - ȧ_imp[i,1] +
               ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * Q₀[j] -
               ∑_{j=2}^{i_imp1-1} ā_exp[i,j-1] * Q₀[j] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q₀[j]
           ) * u +
           (
               ȧ_imp[i,1] - ā_exp[i,i_imp1-1] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q_imp1[j]
           ) * ũ[i_imp1] -
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} (
               ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
           ) +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) +
           ∑_{j=2}^{i_imp1-1} ∑_{k=2}^{j-1} (ȧ_imp[i,1] * ā_exp[i_imp1,j-1] - ā_exp[i,j-1]) *
               (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) -
           ∑_{j=i_imp1+1}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
    ∀ i ∈ i_imp1+1:s+1
We can rewrite some of the sums above as
    ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) =
        = ∑_{k=2}^{i-1} ∑_{j=2}^{k-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{k=3}^{i-1} ∑_{j=2}^{k-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i-2} ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i-1} ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]),
    ∑_{j=i_imp1+1}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) =
        = ∑_{k=i_imp1+1}^{i-1} ∑_{j=2}^{k-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i-2} ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i-1} ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]), and
    ∑_{j=2}^{i_imp1-1} ∑_{k=2}^{j-1} (ȧ_imp[i,1] * ā_exp[i_imp1,j-1] - ā_exp[i,j-1]) *
        (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) =
        = ∑_{k=2}^{i_imp1-1} ∑_{j=2}^{k-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) *
              (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{k=3}^{i_imp1-1} ∑_{j=2}^{k-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) *
              (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i_imp1-2} ∑_{k=j+1}^{i_imp1-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) *
              (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
        = ∑_{j=2}^{i_imp1-1} ∑_{k=j+1}^{i_imp1-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) *
              (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
We then have that
    ũ[i] = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           ∑_{j=2}^{i-1} (
               (ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q[k,j]) * U[j] +
               (ā_exp[i,j-1] - ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j]) * Û[j]
           )
    ∀ i ∈ 2:i_imp1-1 and
    ũ[i] = (
               1 - ȧ_imp[i,1] +
               ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * Q₀[j] -
               ∑_{j=2}^{i_imp1-1} ā_exp[i,j-1] * Q₀[j] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q₀[j]
           ) * u +
           (
               ȧ_imp[i,1] - ā_exp[i,i_imp1-1] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q_imp1[j]
           ) * ũ[i_imp1] -
           ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} (
               ā_imp[i_imp1,j] * U[j] + (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) * Û[j]
           ) +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) +
           ∑_{j=2}^{i_imp1-1} ∑_{k=j+1}^{i_imp1-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) *
              (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) -
           ∑_{j=2}^{i-1} ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = (
               1 - ȧ_imp[i,1] +
               ȧ_imp[i,1] * ∑_{j=2}^{i_imp1-1} ā_exp[i_imp1,j-1] * Q₀[j] -
               ∑_{j=2}^{i_imp1-1} ā_exp[i,j-1] * Q₀[j] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q₀[j]
           ) * u +
           (
               ȧ_imp[i,1] - ā_exp[i,i_imp1-1] -
               ∑_{j=i_imp1+1}^{i-1} ā_exp[i,j-1] * Q_imp1[j]
           ) * ũ[i_imp1] -
           ∑_{j=2}^{i_imp1-1} (
               (
                   ā_imp[i,j] - ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * Q[k,j] +
                   ȧ_imp[i,1] * ā_imp[i_imp1,j] +
                   ∑_{k=j+1}^{i_imp1-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) * Q[k,j]
               ) * U[j] +
               (
                   ā_exp[i,j-1] - ā_imp[i,j] -
                   ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j] +
                   ȧ_imp[i,1] * (ā_exp[i_imp1,j-1] - ā_imp[i_imp1,j]) +
                   ∑_{k=j+1}^{i_imp1-1} (ȧ_imp[i,1] * ā_exp[i_imp1,k-1] - ā_exp[i,k-1]) * Q̂[k,j]
               ) * Û[j]
           ) +
           ∑_{j=i_imp1}^{i-1} (
               (ā_imp[i,j] - ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * Q[k,j]) * U[j] +
               (
                   ā_exp[i,j-1] - ā_imp[i,j] -
                   ∑_{k=max(j,i_imp1)+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j]
               ) * Û[j]
           )
    ∀ i ∈ i_imp1+1:s+1

If there are values of i and j which break the aforementioned condition, this
procedure must be modified by allowing for certain values of F_χ[i] to be stored.
In fact, for all j such that a_imp[j,j] == 0 && a_imp[i,j] != 0 for some value of i,
we have to store F_imp[j].
Similarly, for all j such that a_exp[j+1,j] == 0 && a_exp[i,j] != 0 for some value of i,
we have to store F_exp[j].
For now, we will only consider the case where a_imp[1,1] == 0 && a_imp[i,1] != 0
for some value of i, which corresponds to an EDIRK implicit method.
We will show that, if there are no other values of i and j which break the condition,
then we can express ũ[i] in terms of F_imp[1], u, U[i], and Û[i] as
    ũ[i] = Q₀[i] * u + Δt * Q₁_imp[i] * F_imp[1] +
           ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j]) ∀ i ∈ 2:s+1
Since there are no other such values of i and j, we have that, ∀ i ∈ 2:s+1,
    ũ[i] = u + Δt * a_imp[i,1] * F_imp[1] + ∑_{j=2}^{i-1} ā_imp[i,j] * (U[j] - Û[j]) +
           ∑_{j=1}^{i-2} ā_exp[i,j] * (Û[j+1] - ũ[j+1]) =
         = u + Δt * a_imp[i,1] * F_imp[1] +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] * ũ[j]
First, we check the base case:
    ũ[2] = u + Δt * a_imp[2,1] * F_imp[1] =
         = Q₀[2] * u + Δt * Q₁_imp[2] * F_imp[1], where
    Q₀[2] := 1 and Q₁_imp[2] := a_imp[2,1]
Next, we apply the inductive step: suppose that
    ũ[i] = Q₀[i] * u + Δt * a_imp[i,1] * F_imp[1] +
           ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j]) ∀ j ∈ 2:i-1
Substituting this into the last expression for ũ[i] gives us
    ũ[i] = u + Δt * a_imp[i,1] * F_imp[1] +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ā_exp[i,j-1] *
               (Q₀[j] * u + Δt * Q₁_imp[j] * F_imp[1] + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           Δt * (a_imp[i,1] - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₁_imp[j]) * F_imp[1] +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
We can rewrite the last part of this expression as
    ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} ā_exp[i,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) =
         = ∑_{k=2}^{i-1} ∑_{j=2}^{k-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = ∑_{j=2}^{i-2} ∑_{k=j}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = ∑_{j=2}^{i-1} ∑_{k=j}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
We then have that
    ũ[i] = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           Δt * (a_imp[i,1] - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₁_imp[j]) * F_imp[1] +
           ∑_{j=2}^{i-1} (ā_imp[i,j] * U[j] + (ā_exp[i,j-1] - ā_imp[i,j]) * Û[j]) -
           ∑_{j=2}^{i-1} ∑_{k=j}^{i-1} ā_exp[i,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j]) =
         = (1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j]) * u +
           Δt * (a_imp[i,1] - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₁_imp[j]) * F_imp[1] +
           ∑_{j=2}^{i-1} (
               (ā_imp[i,j] - ∑_{k=j}^{i-1} ā_exp[i,k-1] * Q[k,j]) * U[j] +
               (ā_exp[i,j-1] - ā_imp[i,j] - ∑_{k=j}^{i-1} ā_exp[i,k-1] * Q̂[k,j]) * Û[j]
           ) =
This shows that, ∀ i ∈ 2:s+1,
    ũ[i] = Q₀[i] * u + Δt * a_imp[i,1] * F_imp[1] +
           ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j]), where
    Q₀[i] := 1 - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₀[j],
    Q₁_imp[i] := a_imp[i,1] - ∑_{j=2}^{i-1} ā_exp[i,j-1] * Q₁_imp[j]
    Q[i,j] := ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q[k,j] ∀ j ∈ 2:i-1, and
    Q̂[i,j] := ā_exp[i,j-1] - ā_imp[i,j] - ∑_{k=j+1}^{i-1} ā_exp[i,k-1] * Q̂[k,j] ∀ j ∈ 2:i-1

If the implicit method is EDIRK, then
    We will express ũ[i] in terms of u, U, Û, and F_imp[1] as
        ũ[i] = Q₀[i] * u + Q_edirk[i] * F_imp[1] + ∑_{j=2}^{i-1} (Q[i,j] * U[j] + Q̂[i,j] * Û[j])
    First, we check the base case:
        ũ[2] = u + Δt * a_imp[2,1] * F_imp[1] = Q₀[2] * u + Q_edirk[2] * F_imp[1], where
        Q₀[2] := 1 and Q_edirk[2] := Δt * a_imp[2,1]
    Substituting the expressions for F_imp[i] and F_exp[i] into the definition of ũ[i] gives us
        ũ[i] = u + Δt * ∑_{j=1}^{i-1} a_imp[i,j] * F_imp[j] + Δt * ∑_{j=1}^{i-2} a_exp[i,j] * F_exp[j]
             = u + Δt * a_imp[i,1] * F_imp[1] +
               Δt * ∑_{j=2}^{i-1} a_imp[i,j] * (U[j] - Û[j]) / (Δt * a_imp[j,j]) +
               Δt * ∑_{j=1}^{i-2} a_exp[i,j] * (Û[j+1] - ũ[j+1]) / (Δt * a_exp[j+1,j])
             = u + Δt * a_imp[i,1] * F_imp[1] +
               ∑_{j=2}^{i-1} (
                   a_imp[i,j]/a_imp[j,j] * U[j] +
                   (a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j]) * Û[j]
               ) -
               ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * ũ[j]
    Next, we apply the inductive step; suppose that, ∀ j ∈ 2:i-1,
        ũ[j] = Q₀[j] * u + Q_edirk[j] * F_imp[1] + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
    Substituting this into the last expression for ũ[i] gives us
        ũ[i] = u + Δt * a_imp[i,1] * F_imp[1] +
               ∑_{j=2}^{i-1} (
                   a_imp[i,j]/a_imp[j,j] * U[j] +
                   (a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j]) * Û[j]
               ) -
               ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * (
                   Q₀[j] * u + Q_edirk[j] * F_imp[1] + ∑_{k=2}^{j-1} (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
               )
             = (1 - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q₀[j]) * u +
               (Δt * a_imp[i,1] - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q_edirk[j]) * F_imp[1] +
               ∑_{j=2}^{i-1} (
                   a_imp[i,j]/a_imp[j,j] * U[j] +
                   (a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j]) * Û[j]
               ) -
               ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} a_exp[i,j-1]/a_exp[j,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k])
    We can rewrite the last part of this expression as
        ∑_{j=2}^{i-1} ∑_{k=2}^{j-1} a_exp[i,j-1]/a_exp[j,j-1] * (Q[j,k] * U[k] + Q̂[j,k] * Û[k]) =
             = ∑_{k=2}^{i-1} ∑_{j=2}^{k-1} a_exp[i,k-1]/a_exp[k,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
             = ∑_{j=2}^{i-2} ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
             = ∑_{j=2}^{i-1} ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
    We then have that
        ũ[i] = (1 - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q₀[j]) * u +
               (Δt * a_imp[i,1] - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q_edirk[j]) * F_imp[1] +
               ∑_{j=2}^{i-1} (
                   a_imp[i,j]/a_imp[j,j] * U[j] +
                   (a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j]) * Û[j]
               ) -
               ∑_{j=2}^{i-1} ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * (Q[k,j] * U[j] + Q̂[k,j] * Û[j])
             = (1 - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q₀[j]) * u +
               (Δt * a_imp[i,1] - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q_edirk[j]) * F_imp[1] +
               ∑_{j=2}^{i-1} (
                   (a_imp[i,j]/a_imp[j,j] - ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * Q[k,j]) * U[j] +
                   (
                       a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j] -
                       ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * Q̂[k,j]
                   ) * Û[j]
               )
    This gives us recursive formulas for Q₀, Q_edirk, Q, and Q̂: ∀ i ∈ 2:s+1,
        Q₀[i] := 1 - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q₀[j],
        Q_edirk[i] := Δt * a_imp[i,1] - ∑_{j=2}^{i-1} a_exp[i,j-1]/a_exp[j,j-1] * Q_edirk[j],
        Q[i,j] := a_imp[i,j]/a_imp[j,j] - ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * Q[k,j] ∀ j ∈ 2:i-1, and
        Q̂[i,j] := a_exp[i,j-1]/a_exp[j,j-1] - a_imp[i,j]/a_imp[j,j] -
                  ∑_{k=j+1}^{i-1} a_exp[i,k-1]/a_exp[k,k-1] * Q̂[k,j] ∀ j ∈ 2:i-1

There are two division-by-zero errors that we may encounter in these formulas.
One error is 0/0, but this can be handled by using ⧶ instead of /, which returns
0 if the numerator is 0. The other is x/0, where x is non-zero. This requires
special treatment; e.g., 

Examples:
    IMEX Euler (1,1,1):
        s = 1, a = [1], b = [1], c = [1], â = [0 0; 1 0], b̂ = [1 0], ĉ = [0 1] ==>
        u_{n+1} = U_1, where U_1 = u_n + k * (f_imp(t_n + k, U_1) + f_exp(t_n, u_n))
    IMEX Euler (1,2,1):
        s = 1, a = [1], b = [1], c = [1], â = [0 0; 1 0], b̂ = [0 1], ĉ = [0 1] ==>
        u_{n+1} = u_n + k * (f_imp(t_n + k, U_1) + f_exp(t_n + k, U_1)), where
            U_1 = u_n + k * (f_imp(t_n + k, U_1) + f_exp(t_n, u_n))

Definitions for rewriting formulas in terms of states instead of tendencies:
Let a_{s+1,i} = b_i ∀ i ∈ 1:s and â_{s+2,i} = b̂_i ∀ i ∈ 1:s+1, and, ∀ i ∈ 1:s+1, let
    ũ_i = u_n + k * ∑_{j=1}^{i-1} (a_{i,j} * f_imp(t_n + c_j * k, U_j) + â_{i+1,j} * f_exp(t_n + ĉ_j * k, U_{j-1})) and
    Û_i = ũ_i + k * â_{i+1,i} * f_exp(t_n + ĉ_i * k, U_{i-1}) ==>
u_{n+1} = Û_{s+1} and U_i = Û_i + k * a_{i,i} * f_imp(t_n + c_i * k, U_i) ∀ i ∈ 1:s

Expressing ũ_i as Q₀_i * u_n + ∑_{j=1}^{i-1} (Q_{i,j} * U_j + Q̂_{i,j} * Û_j):
Check the base case ==> ũ_1 = u_n ==> ũ_1 can be expressed in this form if Q₀_1 = 1
Rearrange the equations for Û_i and U_i ==>
f_exp(t_n + ĉ_i * k, U_{i-1}) = (Û_i - ũ_i) / (k * â_{i+1,i}) ∀ i ∈ 1:s+1 and
f_imp(t_n + c_i * k, U_i) = (U_i - Û_i) / (k * a_{i,i}) ∀ i ∈ 1:s
Substitute these into the formula for ũ_i ==>
∀ i ∈ 1:s+1, ũ_i =
    u_n + k * ∑_{j=1}^{i-1} (a_{i,j} * (U_j - Û_j) / (k * a_{j,j}) + â_{i+1,j} * (Û_j - ũ_j) / (k * â_{j+1,j})) =
    u_n + ∑_{j=1}^{i-1} (a_{i,j}/a_{j,j} * (U_j - Û_j) + â_{i+1,j}/â_{j+1,j} * (Û_j - ũ_j)) =
    u_n + ∑_{j=1}^{i-1} (a_{i,j}/a_{j,j} * U_j + (â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j}) * Û_j) -
        ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * ũ_j
Suppose that ũ_j = Q₀_j * u_n + ∑_{k=1}^{j-1} (Q_{j,k} * U_k + Q̂_{j,k} * Û_k) ∀ j ∈ 1:i-1 ==>
ũ_i =
    u_n + ∑_{j=1}^{i-1} (a_{i,j}/a_{j,j} * U_j + (â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j}) * Û_j) -
        ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * (Q₀_j * u_n + ∑_{k=1}^{j-1} (Q_{j,k} * U_k + Q̂_{j,k} * Û_k)) =
    (1 - ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * Q₀_j) * u_n +
        ∑_{j=1}^{i-1} (a_{i,j}/a_{j,j} * U_j + (â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j}) * Û_j) -
        ∑_{j=1}^{i-1} ∑_{k=1}^{j-1} â_{i+1,j}/â_{j+1,j} * (Q_{j,k} * U_k + Q̂_{j,k} * Û_k) =
    (1 - ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * Q₀_j) * u_n +
        ∑_{j=1}^{i-1} (a_{i,j}/a_{j,j} * U_j + (â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j}) * Û_j) -
        ∑_{j=1}^{i-1} ∑_{k=j+1}^{i-1} â_{i+1,k}/â_{k+1,k} * (Q_{k,j} * U_j + Q̂_{k,j} * Û_j) =
    (1 - ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * Q₀_j) * u_n +
        ∑_{j=1}^{i-1} (
            (a_{i,j}/a_{j,j} - ∑_{k=j+1}^{i-1} â_{i+1,k}/â_{k+1,k} * Q_{k,j}) * U_j +
            (â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j} - ∑_{k=j+1}^{i-1} â_{i+1,k}/â_{k+1,k} * Q̂_{k,j}) * Û_j
        ) ==>
∀ i ∈ 1:s+1,
    Q₀_i = 1 - ∑_{j=1}^{i-1} â_{i+1,j}/â_{j+1,j} * Q₀_j,
    Q_{i,j} = a_{i,j}/a_{j,j} - ∑_{k=j+1}^{i-1} â_{i+1,k}/â_{k+1,k} * Q_{k,j} ∀ j ∈ 1:i-1, and
    Q̂_{i,j} = â_{i+1,j}/â_{j+1,j} - a_{i,j}/a_{j,j} - ∑_{k=j+1}^{i-1} â_{i+1,k}/â_{k+1,k} * Q̂_{k,j} ∀ j ∈ 1:i-1

Solving U_i = Û_i + k * a_{i,i} * f_imp(t_n + c_i * k, U_i) for U_i using Newton's method:
U_i - k * a_{i,i} * f_imp(t_n + c_i * k, U_i) = Û_i ==>
F_i(U_i) = Û_i where F_i(U) = U - k * a_{i,i} * f_imp(t_n + c_i * k, U)
Let U_{i,N} denote the value of U_i on the N-th Newton iteration, with U_{i,0} = Û_i
Approximate F_i(U_{i,N+1}) with a first-order series expansion around U_{i,N} ==>
F_i(U_{i,N+1}) ≈
    F_i(U_{i,N}) + F_i'(U_{i,N}) * (U_{i,N+1} - U_{i,N}) =
    U_{i,N} - k * a_{i,i} * f_imp(t_n + c_i * k, U_{i,N}) - W_{i,N} * (U_{i,N+1} - U_{i,N}), where
    W_{i,N} = -I + k * a_{i,i} * J(t_n + c_i * k, U_{i,N}), and where J = ∂/∂U f_imp
Set the approximation equal to Û_i ==>
U_{i,N} - k * a_{i,i} * f_imp(t_n + c_i * k, U_{i,N}) - W_{i,N} * (U_{i,N+1} - U_{i,N}) = Û_i ==>
U_{i,N+1} = U_{i,N} + W_{i,N} \ residual_{i,N}, where
    residual_{i,N} = U_{i,N} - k * a_{i,i} * f_imp(t_n + c_i * k, U_{i,N}) - Û_i
    (i.e., residual_{i,N} = F_i(U_{i,N}) - Û_i)
To speed up the computation, we can approximate W_{i,N} ≈ W_{i,0} for all N
The first Newton iteration gives U_{i,1} = Û_i + W_{i,0} \ residual_{i,0}, where
    residual_{i,0} = -k * a_{i,i} * f_imp(t_n + c_i * k, Û_i)
    (Note the minus sign!)

Solving U_i = Û_i + k * a_{i,i} * f_imp(t_n + c_i * k, U_i) for U_i using a reference state:
U_i - k * a_{i,i} * f_imp(t_n + c_i * k, U_i) = Û_i ==>
F_i(U_i) = Û_i where F_i(U) = U - k * a_{i,i} * f_imp(t_n + c_i * k, U)
Approximate F_i(U_i) with a first-order series expansion around U_ref ==>
F_i(U_i) ≈
    F_i(U_ref) + F_i'(U_ref) * (U_i - U_ref) =
    U_ref - k * a_{i,i} * f_imp(t_n + c_i * k, U_ref) - W_i * (U_i - U_ref), where
    W_i = -I + k * a_{i,i} * J(t_n + c_i * k, U_ref), and where J = ∂/∂U f_imp
Set the approximation equal to Û_i ==>
U_ref - k * a_{i,i} * f_imp(t_n + c_i * k, U_ref) - W_i * (U_i - U_ref) = Û_i ==>
U_i = U_ref + W_i \ residual_i, where
    residual_i = U_ref - k * a_{i,i} * f_imp(t_n + c_i * k, U_ref) - Û_i
    (i.e., residual_i = F_i(U_ref) - Û_i)
If f_imp does not depend on time, then f_imp(_, U_ref) and J(_, U_ref) can be cached
=#
