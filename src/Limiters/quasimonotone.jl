"""
    QuasiMonotoneLimiter

This limiter is inspired by the one presented in Guba et al
[GubaOpt2014](@cite). In the reference paper, it is denoted by OP1, and is
outlined in eqs. (37)-(40). Quasimonotone here is meant to be monotone with
respect to the spectral element nodal values. This limiter involves solving a
constrained optimization problem (a weighted least square problem up to a fixed
tolerance) that is completely local to each element.

As in HOMME, the implementation idea here is the following: we need to find a
grid field which is closest to the initial field (in terms of weighted sum), but
satisfies the min/max constraints. So, first we find values that do not satisfy
constraints and bring these values to a closest constraint. This way we
introduce some change in the tracer mass, which we then redistribute so that the
l2 error is smallest. This redistribution might violate constraints; thus, we do
a few iterations (until `abs(Δtracer_mass) <= rtol * tracer_mass`).

- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit
  mass. This can be a scalar field, or a struct-valued field.
- `ρ`: fluid density Field (scalar).

# Constructor

    limiter = QuasiMonotoneLimiter(ρq::Field; rtol = eps(eltype(parent(ρq))))

Creates a limiter instance for the field `ρq` with relative tolerance `rtol`.

# Usage

Call [`compute_bounds!`](@ref) on the input fields:

    compute_bounds!(limiter, ρq, ρ)

Then call [`apply_limiter!`](@ref) on the output fields:

    apply_limiter!(ρq, ρ, limiter)
"""
struct QuasiMonotoneLimiter{D, G, FT}
    "contains the min and max of each element"
    q_bounds::D
    "contains the min and max of each element and its neighbors"
    q_bounds_nbr::D
    "communication buffer"
    ghost_buffer::G
    "relative tolerance for tracer mass change"
    rtol::FT
end


function QuasiMonotoneLimiter(ρq::Fields.Field; rtol = eps(eltype(parent(ρq))))
    q_bounds = make_q_bounds(Fields.field_values(ρq))
    ghost_buffer =
        Spaces.create_ghost_buffer(q_bounds, Spaces.topology(axes(ρq)))
    return QuasiMonotoneLimiter(q_bounds, similar(q_bounds), ghost_buffer, rtol)
end

Base.@deprecate(
    QuasiMonotoneLimiter(ρq::Fields.Field, ρ::Fields.Field),
    QuasiMonotoneLimiter(ρq::Fields.Field; rtol = eps(eltype(parent(ρq)))),
)


"""
    apply_limit_slab!(limiter, slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)

Apply the computed bounds of the tracer concentration (`slab_q_bounds`) in the
limiter to `slab_ρq`, given the total mass `slab_ρ`, metric terms `slab_WJ`,
and relative tolerance `rtol`. Return whether the tolerance condition could be
satisfied.
"""
function apply_limit_slab!(
    limiter::QuasiMonotoneLimiter,
    slab_ρq,
    slab_ρ,
    slab_WJ,
    slab_q_bounds,
    rtol,
)
    Nf = DataLayouts.ncomponents(slab_ρq)
    (Ni, Nj, _, _, _) = size(slab_ρq)
    maxiter = Ni * Nj

    array_ρq = parent(slab_ρq)
    array_ρ = parent(slab_ρ)
    array_w = parent(slab_WJ)
    array_q_bounds = parent(slab_q_bounds)

    # 1) compute ∫ρ
    total_mass = zero(eltype(array_ρ))
    for j in 1:Nj, i in 1:Ni
        total_mass += array_ρ[i, j, 1] * array_w[i, j, 1]
    end

    @assert total_mass > 0

    converged = true
    for f in 1:Nf
        q_min = array_q_bounds[1, f]
        q_max = array_q_bounds[2, f]

        # 2) compute ∫ρq
        tracer_mass = zero(eltype(array_ρq))
        for j in 1:Nj, i in 1:Ni
            tracer_mass += array_ρq[i, j, f] * array_w[i, j, 1]
        end

        # TODO: Should this condition be enforced? (It isn't in HOMME.)
        # @assert tracer_mass >= 0

        # 3) set bounds
        q_avg = tracer_mass / total_mass
        q_min = min(q_min, q_avg)
        q_max = max(q_max, q_avg)

        # 3) modify ρq
        for iter in 1:maxiter
            Δtracer_mass = zero(eltype(array_ρq))
            for j in 1:Nj, i in 1:Ni
                ρ = array_ρ[i, j, 1]
                ρq = array_ρq[i, j, f]
                ρq_max = ρ * q_max
                ρq_min = ρ * q_min
                w = array_w[i, j]
                if ρq > ρq_max
                    Δtracer_mass += (ρq - ρq_max) * w
                    array_ρq[i, j, f] = ρq_max
                elseif ρq < ρq_min
                    Δtracer_mass += (ρq - ρq_min) * w
                    array_ρq[i, j, f] = ρq_min
                end
            end

            if abs(Δtracer_mass) <= rtol * abs(tracer_mass)
                break
            end

            if Δtracer_mass > 0 # add mass
                total_mass_at_Δ_points = zero(eltype(array_ρ))
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq < ρ * q_max
                        total_mass_at_Δ_points += ρ * w
                    end
                end
                Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq < ρ * q_max
                        array_ρq[i, j, f] += ρ * Δq_at_Δ_points
                    end
                end
            else # remove mass
                total_mass_at_Δ_points = zero(eltype(array_ρ))
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq > ρ * q_min
                        total_mass_at_Δ_points += ρ * w
                    end
                end
                Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq > ρ * q_min
                        array_ρq[i, j, f] += ρ * Δq_at_Δ_points
                    end
                end
            end

            if iter == maxiter
                converged = false
            end
        end
    end
    return converged
end
