"""
    quasimonotone_limiter!(ρq, ρ, min_q, max_q; rtol)

Arguments:
- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit mass
- `ρ`: fluid density Field
- `min_q`: Matrix of min(q) per element, per level: shape [horz_n_elems, vert_n_elems]
- `max_q`: Matrix of max(q) per element, per level: shape [horz_n_elems, vert_n_elems]
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
    min_q,
    max_q;
    rtol,
)
    if ndims(min_q) == 1
        space = axes(ρ)
    else
        space = axes(ρ).horizontal_space
    end

    # Initialize temp variables
    FT = Spaces.undertype(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    horz_n_elems = size(min_q, 1)
    vert_n_elems = size(min_q, 2)
    slab_space = axes(Fields.slab(ρ, 1, 1))
    q_he_data = Fields.field_values(zeros(slab_space))
    ρ_wJ_data = Fields.field_values(zeros(slab_space))

    # Traverse vertical levels
    for ve in 1:vert_n_elems
        # On each level, traverse horizontal elements
        for he in 1:horz_n_elems
            ρ_he_slab = Fields.slab(ρ, ve, he)
            ρq_he_slab = Fields.slab(ρq, ve, he)
            ρ_he_data = Fields.field_values(ρ_he_slab)
            ρq_he_data = Fields.field_values(ρq_he_slab)
            # Compute ρ's and ρq's masses over an horizontal element e
            ρ_he_mass = sum(ρ_he_slab)
            ρq_he_mass = sum(ρq_he_slab)

            # This should never happen, but if it does, don't limit
            if ρ_he_mass <= zero(FT)
                error("Negative elemental mass!")
            end

            # Relax constraints to ensure limiter has a solution:
            # This is only needed if running with the SSP CFL>1 or
            # due to roundoff errors.
            if ρq_he_mass < min_q[he, ve] * ρ_he_mass
                min_q[he, ve] = ρq_he_mass / ρ_he_mass
            end

            if ρq_he_mass > max_q[he, ve] * ρ_he_mass
                max_q[he, ve] = ρq_he_mass / ρ_he_mass
            end

            local_geometry_slab = Fields.slab(space.local_geometry, he)

            q_he_data .= ρq_he_data ./ ρ_he_data

            # Weighted least squares problem iteration loop
            for iter in 1:(Nq * Nq - 1)
                mass_change = zero(FT)
                # Iterate over quadrature points
                for j in 1:Nq, i in 1:Nq
                    ρ_wJ_data[i, j] =
                        local_geometry_slab[i, j].WJ * ρ_he_data[i, j]
                    # Compute the error tolerance and project q into the
                    # upper and lower bounds
                    if q_he_data[i, j] > max_q[he, ve]
                        mass_change +=
                            (q_he_data[i, j] - max_q[he, ve]) * ρ_wJ_data[i, j]
                        q_he_data[i, j] = max_q[he, ve]
                    elseif q_he_data[i, j] < min_q[he, ve]
                        mass_change -=
                            (min_q[he, ve] - q_he_data[i, j]) * ρ_wJ_data[i, j]
                        q_he_data[i, j] = min_q[he, ve]
                    end
                end

                # By this projection of q to the upper/lower bounds, we
                # have introduced a mass_change. Check if we are within the chosen
                # tolerance
                if abs(mass_change) <= rtol * abs(ρq_he_mass)
                    break
                end

                weights_sum = zero(FT)

                # If the change was positive, the removed mass is added
                if mass_change > 0
                    # Iterate over quadrature points
                    for j in 1:Nq, i in 1:Nq
                        if q_he_data[i, j] < max_q[he, ve]
                            weights_sum += ρ_wJ_data[i, j]
                        end
                    end
                    for j in 1:Nq, i in 1:Nq
                        if q_he_data[i, j] < max_q[he, ve]
                            q_he_data[i, j] += mass_change / weights_sum
                        end
                    end
                else # If the change was negative, the added mass is removed
                    # Iterate over quadrature points
                    for j in 1:Nq, i in 1:Nq
                        if q_he_data[i, j] > min_q[he, ve]
                            weights_sum += ρ_wJ_data[i, j]
                        end
                    end
                    for j in 1:Nq, i in 1:Nq
                        if q_he_data[i, j] > min_q[he, ve]
                            q_he_data[i, j] += mass_change / weights_sum
                        end
                    end
                end
            end
            ρq_he_data .= q_he_data .* ρ_he_data
        end # end of horz elem loop
    end # end of vert level loop
    return ρq
end
