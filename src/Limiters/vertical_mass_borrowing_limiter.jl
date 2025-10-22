import .DataLayouts as DL

"""
    VerticalMassBorrowingLimiter(q_min)

A vertical-only mass borrowing limiter.

The mass borrower borrows tracer mass from an adjacent, lower layer.
It conserves the total tracer mass and can avoid negative tracers.

`q_min` should be a tuple of minimum tracer values for each tracer.

At level k, it will first borrow the mass from the layer k+1 (lower level).
If the mass is not sufficient in layer k+1, it will borrow mass from
layer k+2. The borrower will proceed this process until the bottom layer.
If the tracer mass in the bottom layer goes negative, it will repeat the
process from the bottom to the top. In this way, the borrower works for
any shape of mass profiles.

# Example usage

```julia
ρ = fill(1.0, space)
q = fill((a = 0.1, b = 0.1), space)
limiter = VerticalMassBorrowingLimiter((0.0, 0.0))
Limiters.apply_limiter!(q, ρ, limiter)
```

This code was adapted from [E3SM](https://github.com/E3SM-Project/E3SM/blob/2c377c5ec9a5585170524b366ad85074ab1b1a5c/components/eam/src/physics/cam/massborrow.F90)

References:
 - [zhang2018impact](@cite)
"""
struct VerticalMassBorrowingLimiter{T <: Tuple}
    q_min::T
end


"""
    apply_limiter!(q::Fields.Field, ρ::Fields.Field, lim::VerticalMassBorrowingLimiter)

Apply the vertical mass borrowing
limiter `lim` to field `q`, given
density `ρ`.
"""
apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    lim::VerticalMassBorrowingLimiter,
) = apply_limiter!(q, ρ, axes(q), lim, ClimaComms.device(axes(q)))

function apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    space::Spaces.FiniteDifferenceSpace,
    lim::VerticalMassBorrowingLimiter,
    device::ClimaComms.AbstractCPUDevice,
)
    (; J) = Fields.local_geometry_field(ρ)
    q_column_data = Fields.field_values(q)
    ρ_column_data = Fields.field_values(ρ)
    ΔV_column_data = Fields.field_values(J)
    for f in 1:DataLayouts.ncomponents(q_column_data)
        q_min_component = lim.q_min[f]
        column_massborrow!(
            (@view parent(q_column_data)[:, f]),
            (@view parent(ρ_column_data)[:, 1]),
            (@view parent(ΔV_column_data)[:, 1]),
            lim.q_min[f],
        )
    end
    return nothing
end

function apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    lim::VerticalMassBorrowingLimiter,
    device::ClimaComms.AbstractCPUDevice,
)
    (; J) = Fields.local_geometry_field(ρ)
    Fields.bycolumn(axes(q)) do colidx
        q_column_data = Fields.field_values(q[colidx])
        ρ_column_data = Fields.field_values(ρ[colidx])
        ΔV_column_data = Fields.field_values(J[colidx])
        for f in 1:DataLayouts.ncomponents(q_column_data)
            q_min_component = lim.q_min[f]
            column_massborrow!(
                (@view parent(q_column_data)[:, f]),
                (@view parent(ρ_column_data)[:, 1]),
                (@view parent(ΔV_column_data)[:, 1]),
                lim.q_min[f],
            )
        end
    end
    return nothing
end



"""
    column_massborrow!(
        q_data::AbstractArray,
        ρ_data::AbstractArray,
        ΔV_data::AbstractArray,
        q_min::AbstractFloat,
    )

Apply vertical mass borrowing limiter to an array backing a single column of scalar data.
"""
function column_massborrow!(
    q_data::AbstractArray,
    ρ_data::AbstractArray,
    ΔV_data::AbstractArray,
    q_min::AbstractFloat,
)
    Nv = length(q_data)
    borrowed_mass = zero(q_min)
    for i in 0:(Nv - 1) # avoid stepranges for gpu performance
        # top to bottom
        v = Nv - i
        ρΔV_lev = ρ_data[v] * ΔV_data[v]
        new_mass = q_data[v] - (borrowed_mass / ρΔV_lev)
        if new_mass > q_min
            q_data[v] = new_mass
            borrowed_mass = zero(borrowed_mass)
        else
            borrowed_mass = (q_min - new_mass) * ρΔV_lev
            q_data[v] = q_min
        end
    end
    borrowed_mass > zero(borrowed_mass) || return nothing
    for v in 1:Nv
        if borrowed_mass > zero(borrowed_mass)
            ρΔV_lev = ρ_data[v] * ΔV_data[v]
            new_mass = q_data[v] - (borrowed_mass / ρΔV_lev)
            if new_mass > q_min
                q_data[v] = new_mass
                return nothing
            else
                borrowed_mass = (q_min - new_mass) * ρΔV_lev
                q_data[v] = q_min
            end
        end
    end
    return nothing
end
