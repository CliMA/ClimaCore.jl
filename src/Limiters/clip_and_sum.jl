"""
    ClipAndSumLimiter

This limiter is inspired by the one presented HOMME's `limiter_clip_and_sum`
routine.

As in HOMME, this is the fastest limiter that (i) is assured
to find x in the constraint set if that set is not empty and (ii) is such
that the 1-norm of the update is minimal. It does not require iteration.
However, its solution quality is not established.

- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit
  mass. This can be a scalar field, or a struct-valued field.
- `ρ`: fluid density Field (scalar).

# Constructor

    limiter = ClipAndSumLimiter(ρq::Field; rtol = eps(eltype(parent(ρq))))

Creates a limiter instance for the field `ρq` with relative tolerance `rtol`.

# Usage

Call [`compute_bounds!`](@ref) on the input fields:

    compute_bounds!(limiter, ρq, ρ)

Then call [`apply_limiter!`](@ref) on the output fields:

    apply_limiter!(ρq, ρ, limiter)
"""
struct ClipAndSumLimiter{D, G}
    "contains the min and max of each element"
    q_bounds::D
    "contains the min and max of each element and its neighbors"
    q_bounds_nbr::D
    "communication buffer"
    ghost_buffer::G
end

function ClipAndSumLimiter(ρq::Fields.Field; rtol = eps(eltype(parent(ρq))))
    q_bounds = make_q_bounds(Fields.field_values(ρq))
    ghost_buffer =
        Spaces.create_ghost_buffer(q_bounds, Spaces.topology(axes(ρq)))
    return ClipAndSumLimiter(q_bounds, similar(q_bounds), ghost_buffer)
end

"""
    apply_limit_slab!(limiter, slab_ρq, slab_ρ, slab_WJ, slab_q_bounds)

Apply the computed bounds of the tracer concentration (`slab_q_bounds`) in the
limiter to `slab_ρq`, given the total mass `slab_ρ`, metric terms `slab_WJ`.
"""
function apply_limit_slab!(
    limiter::ClipAndSumLimiter,
    slab_ρq,
    slab_ρ,
    slab_WJ,
    slab_q_bounds,
)
    Nf = DataLayouts.ncomponents(slab_ρq)
    (Ni, Nj, _, _, _) = size(slab_ρq)

    array_ρq = parent(slab_ρq)
    array_ρ = parent(slab_ρ)
    array_q = @. array_ρq / array_ρ
    array_w = parent(slab_WJ)
    array_q_bounds = parent(slab_q_bounds)

    # 1) compute ∫ρ
    total_mass = zero(eltype(array_ρ))
    for j in 1:Nj, i in 1:Ni
        total_mass += array_ρ[i, j, 1] * array_w[i, j, 1]
    end

    @assert total_mass > 0

    for f in 1:Nf
        modified = false
        q_min = array_q_bounds[1, f]
        q_max = array_q_bounds[2, f]

        # 2) compute ∫ρq
        tracer_mass = zero(eltype(array_ρq))
        for j in 1:Nj, i in 1:Ni
            tracer_mass += array_ρq[i, j, f] * array_w[i, j, 1]
        end

        # 3) set bounds
        q_avg = tracer_mass / total_mass
        if q_min > q_avg
            q_min = q_avg
            modified = true
        end
        if q_max < q_avg
            q_max = q_avg
            modified = true
        end

        if !modified
            continue
        end

        # 3) modify ρq

        Δtracer = zero(eltype(array_q))
        for j in 1:Nj, i in 1:Ni
            q = array_q[i, j, f]
            w = array_w[i, j]
            if q > q_max
                Δtracer += (q - q_max) * w
                array_q[i, j, f] = q_max
            elseif q < q_min
                Δtracer += (q - q_min) * w
                array_q[i, j, f] = q_min
            end
        end

        if Δtracer != 0
            v = zero(eltype(array_q))
            if Δtracer > 0 # add mass
                for j in 1:Nj, i in 1:Ni
                    q = array_q[i, j, f]
                    v[i, j, 1] = q_max - q
                end
            else # remove mass
                for j in 1:Nj, i in 1:Ni
                    q = array_q[i, j, f]
                    v[i, j, 1] = q - q_min
                end
            end
            den = sum(v[:, :, 1] * array_w[:, :])
            if den > 0 # update
                q += (Δtracer / den) * v[:, :, 1]
            end
        end

        for j in 1:Nj, i in 1:Ni
            ρ = array_ρ[i, j, 1]
            q = array_q[i, j, f]
            ρq[i, j, f] = ρ * q
        end

    end
    return true
end
