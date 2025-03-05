
import Base.Broadcast: materialize, instantiate

"""
    cw = ColumnWise(state::Field)
    z .= cw.(x .+ y)

Indicates to ClimaCore that the broadcast expression should be computed in a
column-wise fashion, leverageing shared memory for the Field, `state`.

In particular, we see this as being a powerful pattern to apply
if many operations are fused into a single kernel. For example:

```julia
cw = ColumnWise(state)
tendencies .= cw.(compute_tendencies.(state, model))
```
"""
struct ColumnWise{BC, S}
    bc::BC
    state::S
end

# User-facing method
ColumnWise(state) = ColumnWise(nothing, state)

function Base.Broadcast.broadcasted(cop::ColumnWise, x)
	ColumnWise(x, cop.state)
end
function Base.materialize(x::ColumnWise)
	instantiate(ColumnWise(x.bc, x.state))
end
function Base.materialize!(out, x::ColumnWise)
	bci = instantiate(ColumnWise(x.bc, x.state))
	Base.copyto!(out, bci)
end

function Base.copyto!(
    field_out::Field,
    bc::ColumnWise,
)
    @info "Calling ColumnWise copyto!"
    space = axes(bc)
    local_geometry = Spaces.local_geometry_data(space)
    (Ni, Nj, _, _, Nh) = size(local_geometry)
    context = ClimaComms.context(axes(field_out))
    device = ClimaComms.device(context)
    return _serial_copyto_shmem!(field_out, bc, Ni, Nj, Nh)
end

function _serial_copyto_shmem!(field_out::Field, bc, Ni::Int, Nj::Int, Nh::Int)
    space = axes(field_out)
    bounds = window_bounds(space, bc)
    bcs = bc # strip_space(bc, space)
    @inbounds for h in 1:Nh, j in 1:Nj, i in 1:Ni
        apply_stencil_shmem!(space, field_out, bcs, (i, j, h), bounds)
    end
    call_post_op_callback() &&
        post_op_callback(field_out, field_out, bc, Ni, Nj, Nh)
    return field_out
end

Base.@propagate_inbounds function apply_stencil_shmem!(
    space,
    field_out,
    bc,
    hidx,
    (li, lw, rw, ri) = window_bounds(space, bc),
)
    if !Topologies.isperiodic(Spaces.vertical_topology(space))
        # left window
        lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
        @inbounds for idx in li:(lw - 1)
            setidx!(
                space,
                field_out,
                idx,
                hidx,
                getidx(space, bc, lbw, idx, hidx),
            )
        end
    end
    # interior
    @inbounds for idx in lw:rw
        setidx!(
            space,
            field_out,
            idx,
            hidx,
            getidx(space, bc, Interior(), idx, hidx),
        )
    end
    if !Topologies.isperiodic(Spaces.vertical_topology(space))
        # right window
        rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
        @inbounds for idx in (rw + 1):ri
            setidx!(
                space,
                field_out,
                idx,
                hidx,
                getidx(space, bc, rbw, idx, hidx),
            )
        end
    end
    return field_out
end

#=
state = rand(3)
x = rand(3)
y = rand(3)
z = rand(3)
cw = ColumnWise(state)
z .= cw.(x .+ y)
=#
