using LinearAlgebra: Tridiagonal, lu!, ldiv!

using ClimaCore: Spaces, Fields, Operators
using ClimaCore.Utilities: half

const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

struct SchurComplementW{F, T, J1, J2, J3, S, A}
    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flags for computing the Jacobian
    flags::F

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::T

    # nonzero blocks of the Jacobian (∂ρₜ/∂𝕄, ∂𝔼ₜ/∂𝕄, ∂𝕄ₜ/∂𝔼, and ∂𝕄ₜ/∂ρ)
    ∂ρₜ∂𝕄::J1
    ∂𝔼ₜ∂𝕄::J2
    ∂𝕄ₜ∂𝔼::J3
    ∂𝕄ₜ∂ρ::J3

    # cache for the Schur complement linear solve
    S::S
    S_column_array::A

    # whether to test the Jacobian and linear solver
    test::Bool
end

function SchurComplementW(Y, transform, flags, test = false)
    FT = eltype(Y)
    dtγ_ref = Ref(zero(FT))
    center_space = axes(Y.Yc.ρ)
    face_space = axes(Y.w)

    # TODO: Automate this.
    J1_eltype = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    J2_eltype =
        flags.∂𝔼ₜ∂𝕄_mode == :exact && :ρe in propertynames(Y.Yc) ?
        Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}} : J1_eltype
    ∂ρₜ∂𝕄 = Fields.Field(J1_eltype, center_space)
    ∂𝔼ₜ∂𝕄 = Fields.Field(J2_eltype, center_space)
    ∂𝕄ₜ∂𝔼 = Fields.Field(J1_eltype, face_space)
    ∂𝕄ₜ∂ρ = Fields.Field(J1_eltype, face_space)

    # TODO: Automate this.
    S_eltype = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    S = similar(Y.w, S_eltype)
    N = Spaces.nlevels(axes(Y.w))
    S_column_array = Tridiagonal(
        Array{FT}(undef, N - 1),
        Array{FT}(undef, N),
        Array{FT}(undef, N - 1),
    )

    SchurComplementW{
        typeof(flags),
        typeof(dtγ_ref),
        typeof(∂ρₜ∂𝕄),
        typeof(∂𝔼ₜ∂𝕄),
        typeof(∂𝕄ₜ∂ρ),
        typeof(S),
        typeof(S_column_array),
    }(
        transform,
        flags,
        dtγ_ref,
        ∂ρₜ∂𝕄,
        ∂𝔼ₜ∂𝕄,
        ∂𝕄ₜ∂𝔼,
        ∂𝕄ₜ∂ρ,
        S,
        S_column_array,
        test,
    )
end

# We only use Wfact, but the implicit/IMEX solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(w::SchurComplementW) = w

#=
A = [-I         0          dtγ ∂ρₜ∂𝕄;
     0          -I         dtγ ∂𝔼ₜ∂𝕄;
     dtγ ∂𝕄ₜ∂ρ  dtγ ∂𝕄ₜ∂𝔼  -I       ]
    [-I   0    A13;
     0    -I   A23;
     A31  A32  -I ]
b = [b1; b2; b3]
x = [x1; x2; x3]
Solving A x = b:
    -x1 + A13 x3 = b1 ==> x1 = -b1 + A13 x3  (1)
    -x2 + A23 x3 = b2 ==> x2 = -b2 + A23 x3  (2)
    A31 x1 + A32 x2 - x3 = b3  (3)
Substitute (1) and (2) into (3):
    A31 (-b1 + A13 x3) + A32 (-b2 + A23 x3) - x3 = b3 ==>
    (-I + A31 A13 + A32 A23) x3 = b3 + A31 b1 + A32 b2 ==>
    x3 = (-I + A31 A13 + A32 A23) \ (b3 + A31 b1 + A32 b2)
Finally, use (1) and (2) to get x1 and x2.
Note: The matrix S = -I + A31 A13 + A32 A23 is the "Schur complement" of
[-I 0; 0 -I] (the top-left 4 blocks) in A.
=#
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        @unpack dtγ_ref, ∂ρₜ∂𝕄, ∂𝔼ₜ∂𝕄, ∂𝕄ₜ∂𝔼, ∂𝕄ₜ∂ρ, S, S_column_array = A
        dtγ = dtγ_ref[]

        xρ = x.Yc.ρ
        bρ = b.Yc.ρ
        if :ρθ in propertynames(x.Yc)
            x𝔼 = x.Yc.ρθ
            b𝔼 = b.Yc.ρθ
        elseif :ρe in propertynames(x.Yc)
            x𝔼 = x.Yc.ρe
            b𝔼 = b.Yc.ρe
        end
        if :ρw in propertynames(x)
            x𝕄 = x.ρw.components.data.:1
            b𝕄 = b.ρw.components.data.:1
        elseif :w in propertynames(x)
            x𝕄 = x.w.components.data.:1
            b𝕄 = b.w.components.data.:1
        end

        # TODO: Extend LinearAlgebra.I to work with stencil fields.
        T = eltype(eltype(S))
        I = Ref(Operators.StencilCoefs{-1, 1}((zero(T), one(T), zero(T))))
        if Operators.bandwidths(eltype(∂𝔼ₜ∂𝕄)) != (-half, half)
            str = "The linear solver cannot yet be run with the given ∂𝔼ₜ/∂𝕄 \
                block, since it has more than 2 diagonals. Setting ∂𝔼ₜ/∂𝕄 = 0 \
                for the Schur complement computation. Consider changing the \
                jacobian_mode or the energy variable."
            @warn str maxlog = 1
            @. S = -I + dtγ^2 * compose(∂𝕄ₜ∂ρ, ∂ρₜ∂𝕄)
        else
            @. S = -I + dtγ^2 * (compose(∂𝕄ₜ∂ρ, ∂ρₜ∂𝕄) + compose(∂𝕄ₜ∂𝔼, ∂𝔼ₜ∂𝕄))
        end

        @. x𝕄 = b𝕄 + dtγ * (apply(∂𝕄ₜ∂ρ, bρ) + apply(∂𝕄ₜ∂𝔼, b𝔼))

        # TODO: Do this with stencil_solve!.
        Ni, Nj, _, _, Nh = size(Spaces.local_geometry_data(axes(xρ)))
        for h in 1:Nh, j in 1:Nj, i in 1:Ni
            x𝕄_column_view = parent(Spaces.column(x𝕄, i, j, h))
            S_column = Spaces.column(S, i, j, h)
            @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
            S_column_array.d .= parent(S_column.coefs.:2)
            @views S_column_array.du .= parent(S_column.coefs.:3)[1:(end - 1)]
            ldiv!(lu!(S_column_array), x𝕄_column_view)
        end

        @. xρ = -bρ + dtγ * apply(∂ρₜ∂𝕄, x𝕄)
        @. x𝔼 = -b𝔼 + dtγ * apply(∂𝔼ₜ∂𝕄, x𝕄)

        if A.test && Operators.bandwidths(eltype(∂𝔼ₜ∂𝕄)) == (-half, half)
            Ni, Nj, _, Nv, Nh = size(Spaces.local_geometry_data(axes(xρ)))
            ∂Yₜ∂Y = Array{Float64}(undef, 3 * Nv + 1, 3 * Nv + 1)
            ΔY = Array{Float64}(undef, 3 * Nv + 1)
            ΔΔY = Array{Float64}(undef, 3 * Nv + 1)
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ∂Yₜ∂Y .= 0.0
                ∂Yₜ∂Y[1:Nv, (2 * Nv + 1):(3 * Nv + 1)] .=
                    column_matrix(∂ρₜ∂𝕄, i, j, h)
                ∂Yₜ∂Y[(Nv + 1):(2 * Nv), (2 * Nv + 1):(3 * Nv + 1)] .=
                    column_matrix(∂𝔼ₜ∂𝕄, i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), 1:Nv] .=
                    column_matrix(∂𝕄ₜ∂ρ, i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), (Nv + 1):(2 * Nv)] .=
                    column_matrix(∂𝕄ₜ∂𝔼, i, j, h)
                ΔY[1:Nv] .= column_vector(xρ, i, j, h)
                ΔY[(Nv + 1):(2 * Nv)] .= column_vector(x𝔼, i, j, h)
                ΔY[(2 * Nv + 1):(3 * Nv + 1)] .= column_vector(x𝕄, i, j, h)
                ΔΔY[1:Nv] .= column_vector(bρ, i, j, h)
                ΔΔY[(Nv + 1):(2 * Nv)] .= column_vector(b𝔼, i, j, h)
                ΔΔY[(2 * Nv + 1):(3 * Nv + 1)] .= column_vector(b𝕄, i, j, h)
                @assert (-LinearAlgebra.I + dtγ * ∂Yₜ∂Y) * ΔY ≈ ΔΔY
            end
        end

        if :ρuₕ in propertynames(x)
            @. x.ρuₕ = -b.ρuₕ
        elseif :uₕ in propertynames(x)
            @. x.uₕ = -b.uₕ
        end

        if A.transform
            x .*= dtγ
        end
    end
end
