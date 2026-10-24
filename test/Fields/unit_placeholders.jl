import ClimaCore
import ClimaCore: Fields, Spaces, Grids, Operators
import LazyBroadcast: lazy
using ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

using Test

const PlaceholderGrid = Grids.PlaceholderGrid
toggle(x, space) = Grids.toggle_placeholder_grid(x, space)

@testset "toggle_placeholder_grid" begin
    FT = Float64
    center_space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    face_space = Spaces.face_space(center_space)
    column_space = TU.ColumnCenterFiniteDifferenceSpace(FT)
    horizontal_space = Spaces.horizontal_space(center_space)
    ᶜfield = ones(center_space)
    ᶠfield = ones(face_space)

    for (field, space) in (
        (ᶜfield, center_space),
        (ᶠfield, center_space),
        (ones(column_space), column_space),
        (ones(horizontal_space), horizontal_space),
    )
        stripped = toggle(field, space)
        @test Spaces.grid(axes(stripped)) === PlaceholderGrid()
        @test Fields.field_values(stripped) === Fields.field_values(field)
        @test toggle(stripped, space) === field
    end

    # Level and column grids keep their wrappers, along with their indices.
    for field in (Fields.level(ᶜfield, 2), Fields.column(ᶜfield, 1, 1, 1))
        stripped_grid = Spaces.grid(axes(toggle(field, center_space)))
        @test stripped_grid.full_grid === PlaceholderGrid()
        @test stripped_grid isa typeof(Spaces.grid(axes(field))).name.wrapper
        @test toggle(toggle(field, center_space), center_space) === field
    end

    # A level destination uses the full grid of its level, so that a level
    # field from a different level keeps its own level.
    level_space = axes(Fields.level(ᶜfield, 1))
    other_level_field = Fields.level(ᶜfield, 3)
    @test toggle(toggle(other_level_field, level_space), level_space) ===
          other_level_field

    # Grids of other types are not replaced.
    horizontal_field = ones(horizontal_space)
    @test toggle(horizontal_field, center_space) === horizontal_field

    # Every node of a broadcast is toggled, including the axes of pointwise
    # nodes and the fields in boundary conditions.
    interp = Operators.InterpolateC2F(;
        bottom = Operators.SetValue(Fields.level(ᶜfield, 1)),
        top = Operators.Extrapolate(),
    )
    grad = Operators.GradientF2C()
    bc = Base.Broadcast.instantiate(
        @. lazy(grad(interp(ᶜfield * other_level_field) * ᶠfield))
    )
    stripped_bc = toggle(bc, center_space)
    @test !occursin("ExtrudedFiniteDifferenceGrid", string(typeof(stripped_bc)))
    @test toggle(stripped_bc, center_space) === bc

    # toggle_compact_args compacts duplicate grids across multiple arguments,
    # inside a single Broadcasted, when a 2D horizontal field precedes 3D
    # extruded fields, and inside Tuple and NamedTuple argument bundles.
    (compact_bc,) = Grids.toggle_compact_args(bc)
    @test Spaces.grid(compact_bc.axes) === Spaces.grid(center_space)
    @test !occursin("ExtrudedFiniteDifferenceGrid", string(typeof(compact_bc.args)))
    @test Grids.toggle_compact_args(compact_bc) === (bc,)

    mixed_args = (horizontal_field, ᶜfield, ᶠfield)
    compact_mixed = Grids.toggle_compact_args(mixed_args...)
    @test compact_mixed[1] === horizontal_field
    @test compact_mixed[2] === ᶜfield
    @test Spaces.grid(axes(compact_mixed[3])) === PlaceholderGrid()
    @test Grids.toggle_compact_args(compact_mixed...) === mixed_args

    bundled_args = ((; sfc = horizontal_field, c = ᶜfield, f = ᶠfield), (ᶜfield, bc))
    compact_bundled = Grids.toggle_compact_args(bundled_args...)
    @test compact_bundled[1].sfc === horizontal_field
    @test compact_bundled[1].c === ᶜfield
    @test Spaces.grid(axes(compact_bundled[1].f)) === PlaceholderGrid()
    @test Spaces.grid(axes(compact_bundled[2][1])) === PlaceholderGrid()
    @test Spaces.grid(compact_bundled[2][2].axes) === PlaceholderGrid()
    @test Grids.toggle_compact_args(compact_bundled...) === bundled_args
end
