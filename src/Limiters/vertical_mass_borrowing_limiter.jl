import .DataLayouts as DL

"""
    VerticalMassBorrowingLimiter(f::Fields.Field, q_min)

A vertical-only mass borrowing limiter.

The mass borrower borrows tracer mass from an adjacent, lower layer.
It conserves the total tracer mass and can avoid negative tracers.

At level k, it will first borrow the mass from the layer k+1 (lower level).
If the mass is not sufficient in layer k+1, it will borrow mass from
layer k+2. The borrower will proceed this process until the bottom layer.
If the tracer mass in the bottom layer goes negative, it will repeat the
process from the bottom to the top. In this way, the borrower works for
any shape of mass profiles.

This code was adapted from [E3SM](https://github.com/E3SM-Project/E3SM/blob/2c377c5ec9a5585170524b366ad85074ab1b1a5c/components/eam/src/physics/cam/massborrow.F90)

References:
 - [zhang2018impact](@cite)
"""
struct VerticalMassBorrowingLimiter{F, T}
    bmass::F
    ic::F
    q_min::T
end
function VerticalMassBorrowingLimiter(f::Fields.Field, q_min)
    bmass = similar(Spaces.level(f, 1))
    ic = similar(Spaces.level(f, 1))
    return VerticalMassBorrowingLimiter(bmass, ic, q_min)
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
    cache = (; bmass = lim.bmass, ic = lim.ic, q_min = lim.q_min)
    columnwise_massborrow_cpu(q, ρ, cache)
end

function apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    lim::VerticalMassBorrowingLimiter,
    device::ClimaComms.AbstractCPUDevice,
)
    Fields.bycolumn(axes(q)) do colidx
        cache = (;
            bmass = lim.bmass[colidx],
            ic = lim.ic[colidx],
            q_min = lim.q_min,
        )
        columnwise_massborrow_cpu(q[colidx], ρ[colidx], cache)
    end
end

# TODO: can we improve the performance?
# `bycolumn` on the CPU may be better here since we could multithread it.
function columnwise_massborrow_cpu(q::Fields.Field, ρ::Fields.Field, cache) # column fields
    (; bmass, ic, q_min) = cache

    Δz = Fields.Δz_field(q)
    Δz_vals = Fields.field_values(Δz)
    (; J) = Fields.local_geometry_field(ρ)
    # ΔV_vals = Fields.field_values(J)
    ΔV_vals = Δz_vals
    ρ_vals = Fields.field_values(ρ)
    #  loop over tracers
    nlevels = Spaces.nlevels(axes(q))
    @. ic = 0
    @. bmass = 0
    q_vals = Fields.field_values(q)
    # top to bottom
    for f in 1:DataLayouts.ncomponents(q_vals)
        for v in 1:nlevels
            CI = CartesianIndex(1, 1, f, v, 1)
            # new mass in the current layer
            ρΔV_lev =
                DL.getindex_field(ΔV_vals, CI) * DL.getindex_field(ρ_vals, CI)
            nmass = DL.getindex_field(q_vals, CI) + bmass[] / ρΔV_lev
            if nmass > q_min[f]
                #  if new mass in the current layer is positive, don't borrow mass any more
                DL.setindex_field!(q_vals, nmass, CI)
                bmass[] = 0
            else
                #  set mass to q_min in the current layer, and save bmass
                bmass[] = (nmass - q_min[f]) * ρΔV_lev
                DL.setindex_field!(q_vals, q_min[f], CI)
                ic[] = ic[] + 1
            end
        end

        #  bottom to top
        for v in nlevels:-1:1
            CI = CartesianIndex(1, 1, f, v, 1)
            # if the surface layer still needs to borrow mass
            if bmass[] < 0
                ρΔV_lev =
                    DL.getindex_field(ΔV_vals, CI) *
                    DL.getindex_field(ρ_vals, CI)
                # new mass in the current layer
                nmass = DL.getindex_field(q_vals, CI) + bmass[] / ρΔV_lev
                if nmass > q_min[f]
                    # if new mass in the current layer is positive, don't borrow mass any more
                    DL.setindex_field!(q_vals, nmass, CI)
                    bmass[] = 0
                else
                    # if new mass in the current layer is negative, continue to borrow mass
                    bmass[] = (nmass - q_min[f]) * ρΔV_lev
                    DL.setindex_field!(q_vals, q_min[f], CI)
                end
            end
        end
    end

    return nothing
end
