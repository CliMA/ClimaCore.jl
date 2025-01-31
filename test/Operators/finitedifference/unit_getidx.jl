#=
julia --check-bounds=yes --project=.buildkite
using Revise; include("test/Operators/finitedifference/unit_getidx.jl")
=#
using Test
import LazyBroadcast: lazy
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Utilities, Spaces, Fields, Operators, Meshes, Topologies
using ClimaCore.Utilities: half
using ClimaCore.CommonSpaces

function get_getidx_args(bc)
    space = axes(bc)
    hidx = (1, 1, 1)
    loc_i = Operators.Interior()
    loc_l = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    loc_r = Operators.RightBoundaryWindow{Spaces.left_boundary_name(space)}()
    return (; space, bc, loc_l, loc_i, loc_r, hidx)
end

function get_getidx_args(bc, idx)
    space = axes(bc)
    vtopology = Spaces.vertical_topology(space)
    hidx = (1, 1, 1)
    (li, lw, rw, ri) = Operators.window_bounds(space, bc)
    not_periodic = !Topologies.isperiodic(vtopology)
    if not_periodic && idx in li:(lw - 1)
        loc = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    elseif idx in (rw + 1):ri
        loc = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    elseif idx in lw:rw
        loc = Operators.Interior()
    else
        error("Uncaught case")
    end
    return (; space, bc, loc, hidx)
end

# @testset "FD getidx tests" begin
    FT = Float64
    ᶜspace = ColumnSpace(FT;
        z_elem = 10,
        z_min = 0,
        z_max = 10,
        staggering = CellCenter(),
        stretch = Meshes.ExponentialStretching{FT}(2),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠϕ = Fields.Field(FT, ᶠspace)
    ᶜϕ = Fields.Field(FT, ᶜspace)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠΔz = Fields.Δz_field(ᶠspace)
    ᶜΔz = Fields.Δz_field(ᶜspace)
    ᶠϕ_vals = Fields.field_values(ᶠϕ)
    ᶜϕ_vals = Fields.field_values(ᶜϕ)
    ᶜJ_vals = Fields.field_values(Fields.local_geometry_field(ᶜspace).J)
    ᶠJ_vals = Fields.field_values(Fields.local_geometry_field(ᶠspace).J)
    @. ᶠϕ .= ᶠz
    @. ᶠϕ .= ᶠz
    op = Operators.GradientF2C()
    bc = @. lazy(op(ᶠϕ))
    (; space, bc, loc_l, loc_i, loc_r, hidx) = get_getidx_args(bc)
    @test_throws BoundsError Operators.getidx(space, bc, loc_l, 0, hidx)[]

    ᶜ∇ϕ = map(1:Spaces.nlevels(space)) do idx
        (; loc) = get_getidx_args(bc, idx)
        Operators.getidx(space, bc, loc, idx, hidx)[]
    end
    ᶜ∇ϕ_lir = (
        Operators.getidx(space, bc, loc_l, 1, hidx)[],
        Operators.getidx(space, bc, loc_i, 5, hidx)[],
        Operators.getidx(space, bc, loc_r, 10, hidx)[],
    )
    @show ᶜ∇ϕ_lir
    ᶜ∇ϕ_lir_expected = (
        (ᶠϕ_vals[2] - ᶠϕ_vals[1]) / ᶜJ_vals[1],
        (ᶠϕ_vals[2] - ᶠϕ_vals[1]) / ᶜJ_vals[1],
        (ᶠϕ_vals[2] - ᶠϕ_vals[1]) / ᶜJ_vals[1],
    )
    @show ᶜ∇ϕ_lir_expected

    @test Operators.getidx(space, bc, loc_l, 1, hidx)[] == Fields.level(ᶠΔz, 0+half)[]
    @show Operators.stencil_interior(op, loc_l, space, 1, hidx, bc.args...)[]
    @show Operators.stencil_interior(op, loc_i, space, 5, hidx, bc.args...)[]
    @test Operators.getidx(space, bc, loc_l, 1, hidx)[] == Fields.level(ᶠΔz, 0+half)[]
    # @show val
    # @show Fields.level(ᶠΔz, half)[]
    @show ᶠΔz
    val = Operators.getidx(space, bc, loc_i, 5, hidx)[]
    @show val
    val = Operators.getidx(space, bc, loc_r, 10, hidx)[]
    @show val
    @test_throws BoundsError Operators.getidx(space, bc, loc_r, 11, hidx)[]
# end
