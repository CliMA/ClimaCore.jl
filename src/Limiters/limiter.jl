"""
    quasimonotone_limiter!(ρq, ρ, min_ρq, max_ρq; rtol)

Arguments:
- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit mass
- `ρ: fluid density Field
- `min_ρq: Array of min(ρq) per element
- `max_ρq`: Array of max(ρq) per element
- `rtol`: relative tolerance needed to solve element-wise optimization problem

This limiter is inspired by the one presented in Guba et al [GubaOpt2014](@cite).
In the reference paper, it is denoted by OP1, and is outlined in eqs. (37)-(40).
Quasimonotone here is meant to be monotone with respect to the spectral element
nodal values. This limiter involves solving a constrained optimization problem
(a weighted least square problem up to a fixed tolerance denoted by `rtol`)
that is completely local to each element.
As in HOMME, the implementation idea here is the following: we need to find a grid
field which is closest to the initial field (in terms of weighted sum), but satisfies
the min/max constraints. So, first we find values that do not satisfy constraints
and bring these values to a closest constraint. This way we introduce some mass
change (`mass_change`), which we then redistribute so that the
l2 error is smallest. This redistribution might violate constraints; thus, we do
a few iterations (typically a couple).
"""
function quasimonotone_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    min_ρq,
    max_ρq;
    rtol,
)
    space = axes(ρ)
    FT = Spaces.undertype(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    n_elems = length(min_ρq)
    q = zeros(axes(Fields.slab(ρ, 1)))
    ρ_wJ = zeros(axes(Fields.slab(ρ, 1)))

    # Traverse elements
    for e in 1:n_elems
        ρ_e_slab = Fields.slab(ρ, e)
        ρq_e_slab = Fields.slab(ρq, e)
        # Compute ρ's and ρq's masses over an element e
        ρ_e_mass = sum(ρ_e_slab)
        ρq_e_mass = sum(ρq_e_slab)

        # This should never happen, but if it does, don't limit
        if ρ_e_mass <= 0
            continue
        end

        # Relax constraints to ensure limiter has a solution:
        # This is only needed if running with the SSP CFL>1 or
        # due to roundoff errors.
        if ρq_e_mass < min_ρq[e] * ρ_e_mass
            min_ρq[e] = ρq_e_mass / ρ_e_mass
        end

        if ρq_e_mass > max_ρq[e] * ρ_e_mass
            max_ρq[e] = ρq_e_mass / ρ_e_mass
        end

        local_geometry_slab = Fields.slab(space.local_geometry, e)

        q = ρq_e_slab ./ ρ_e_slab

        # Weighted least squares problem iteration loop
        for iter in 1:(Nq * Nq - 1)
            mass_change = 0.0
            # Iterate over quadrature points
            for j in 1:Nq, i in 1:Nq
                ρ_wJ_data = Fields.todata(ρ_wJ)
                q_data = Fields.todata(q)
                ρ_wJ_data[i, j] =
                    local_geometry_slab[i, j].WJ .*
                    Fields.todata(ρ_e_slab)[i, j]
                # Compute the error tolerance and project q into the
                # upper and lower bounds
                if q_data[i, j] > max_ρq[e]
                    mass_change += (q_data[i, j] - max_ρq[e]) * ρ_wJ_data[i, j]
                    q_data[i, j] = max_ρq[e]
                elseif q_data[i, j] < min_ρq[e]
                    mass_change -= (min_ρq[e] - q_data[i, j]) * ρ_wJ_data[i, j]
                    q_data[i, j] = min_ρq[e]
                end
            end

            # By this projection of q to the upper/lower bounds, we
            # have introduced a mass_change. Check if we are within the chosen
            # tolerance
            if abs(mass_change) <= rtol * abs(ρq_e_mass)
                break
            end

            weights_sum = 0.0

            # If the change was positive, the removed mass is added
            if mass_change > 0
                # Iterate over quadrature points
                for j in 1:Nq, i in 1:Nq
                    ρ_wJ_data = Fields.todata(ρ_wJ)
                    q_data = Fields.todata(q)
                    if q_data[i, j] < max_ρq[e]
                        weights_sum += ρ_wJ_data[i, j]
                    end
                end
                for j in 1:Nq, i in 1:Nq
                    q_data = Fields.todata(q)
                    if q_data[i, j] < max_ρq[e]
                        q_data[i, j] += mass_change / weights_sum
                    end
                end
            else # If the change was negative, the added mass is removed
                # Iterate over quadrature points
                for j in 1:Nq, i in 1:Nq
                    ρ_wJ_data = Fields.todata(ρ_wJ)
                    q_data = Fields.todata(q)
                    if q_data[i, j] > min_ρq[e]
                        weights_sum += ρ_wJ_data[i, j]
                    end
                end
                for j in 1:Nq, i in 1:Nq
                    q_data = Fields.todata(q)
                    if q_data[i, j] > min_ρq[e]
                        q_data[i, j] += mass_change / weights_sum
                    end
                end
            end
        end
        ρq_e_slab .= q .* ρ_e_slab
    end
end
