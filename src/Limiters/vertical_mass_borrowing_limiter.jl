import .DataLayouts as DL

# Matrix with one row per level of a column of data, whose columns each
# correspond to one component of the data's element type
column_matrix(data) = reshape(parent(data), size(data, 1), :)

"""
    VerticalMassBorrowingLimiter(q_min)

Vertical-only mass-borrowing limiter.

The limiter borrows tracer mass from adjacent lower layers to raise each layer's tracer
value to at least its minimum. It conserves the total tracer mass in the column.

`q_min` is a tuple with one minimum tracer value per component of the limited field.

At level `k`, the limiter first borrows mass from layer `k+1` (the lower level). If the
mass in layer `k+1` is not sufficient, it borrows from layer `k+2`, and so on down to
the bottom layer. If the tracer mass in the bottom layer goes below the minimum, the
limiter repeats the process from the bottom to the top. This makes the limiter work for
any shape of mass profile.

# Examples

```julia
ρ = fill(1.0, space)
q = fill((a = 0.1, b = 0.1), space)
limiter = VerticalMassBorrowingLimiter((0.0, 0.0))
Limiters.apply_limiter!(q, ρ, limiter)
```

Adapted from the
[E3SM mass borrower](https://github.com/E3SM-Project/E3SM/blob/2c377c5ec9a5585170524b366ad85074ab1b1a5c/components/eam/src/physics/cam/massborrow.F90);
see [zhang2018impact](@cite).
"""
struct VerticalMassBorrowingLimiter{T <: Tuple}
    q_min::T
end


"""
    apply_limiter!(q::Fields.Field, ρ::Fields.Field, lim::VerticalMassBorrowingLimiter)

Apply the vertical mass-borrowing limiter `lim` to the tracer field `q` in place, given
the density field `ρ`.

Each component of `q` is limited column by column with the corresponding entry of
`lim.q_min`, using the cell volume from the local geometry of `ρ`. Returns `nothing`.
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
            (@view column_matrix(q_column_data)[:, f]),
            (@view column_matrix(ρ_column_data)[:, 1]),
            (@view column_matrix(ΔV_column_data)[:, 1]),
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
                (@view column_matrix(q_column_data)[:, f]),
                (@view column_matrix(ρ_column_data)[:, 1]),
                (@view column_matrix(ΔV_column_data)[:, 1]),
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

Apply the vertical mass-borrowing limiter in place to `q_data`, the array backing a single
column of scalar tracer data, with column density `ρ_data`, cell volume `ΔV_data`, and
minimum value `q_min`. Returns `nothing`.
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
