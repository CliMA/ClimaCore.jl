using Test
using ClimaComms
ClimaComms.@import_required_backends
import Adapt
import ClimaCore
import ClimaCore: Spaces, Fields, Grids, Geometry, Operators
import ClimaCore.Utilities: half

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

# A space is a handle to its grid plus a staggering, so it must stay within a
# couple of words regardless of how much the grid itself holds; a Field then
# adds only its space to its DataLayout, and a lazy broadcast only its
# arguments and its space.
const MAX_SPACE_BYTES = 2 * sizeof(Int)

has_grid(space) = !(space isa Spaces.PointSpace)
values_bytes(field) = sizeof(typeof(Fields.field_values(field)))
args_bytes(bc) = sum(arg -> sizeof(typeof(arg)), bc.args)

@testset "Spaces and Fields are pointer-sized" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        has_grid(space) || continue
        field = fill((; x = FT(1), y = FT(2)), space)
        # With the grid stored inline these were 176 and 184 bytes.
        @test sizeof(typeof(space)) ≤ MAX_SPACE_BYTES
        @test sizeof(typeof(field)) ≤ values_bytes(field) + MAX_SPACE_BYTES
        @test sizeof(typeof(field.x)) ≤ values_bytes(field.x) + MAX_SPACE_BYTES
        bc = Base.Broadcast.broadcasted(+, field.x, field.y, field.x, field.y)
        @test sizeof(typeof(bc)) ≤ args_bytes(bc) + MAX_SPACE_BYTES
        if TU.levelable(space)
            level_space = Spaces.level(space, TU.fc_index(1, space))
            # Levels of single columns are point spaces, which hold their
            # local geometry directly.
            if has_grid(level_space)
                @test sizeof(typeof(level_space)) ≤ MAX_SPACE_BYTES
            end
        end
    end
end

@testset "Spaces expose the grid" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        has_grid(space) || continue
        grid = Spaces.grid(space)
        @test grid isa Grids.AbstractGrid
        # Field access is kept for downstream code that reads `space.grid`.
        @test space.grid === grid
        if TU.levelable(space)
            level_space = Spaces.level(space, TU.fc_index(1, space))
            if has_grid(level_space)
                level_grid = Spaces.grid(level_space)
                @test level_grid isa Grids.LevelGrid
                @test level_grid.full_grid === grid
            end
        end
    end
end

@testset "Spaces built from the same grid are identical" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        has_grid(space) || continue
        grid = Spaces.grid(space)
        staggering = Spaces.staggering(space)
        # Rebuilding the space from its grid shares the cached box.
        @test Spaces.space(grid, staggering) === space
        if TU.levelable(space)
            @test Spaces.face_space(Spaces.center_space(space)) ===
                  Spaces.face_space(space)
            @test Spaces.center_space(Spaces.face_space(space)) ===
                  Spaces.center_space(space)
            @test Spaces.space(space, staggering) === space
            v = TU.fc_index(1, space)
            @test Spaces.level(space, v) === Spaces.level(space, v)
            field = fill((; x = FT(1)), space)
            # Fields compare by space identity, so two extractions of the same
            # level are equal, and so are their spaces read back from a level
            # field.
            @test Fields.level(field, v) == Fields.level(field, v)
            @test axes(Fields.level(field, v)) === Spaces.level(space, v)
            if space isa Spaces.ExtrudedFiniteDifferenceSpace
                hspace = Spaces.horizontal_space(space)
                @test hspace === Spaces.horizontal_space(space)
                @test hspace === Spaces.space(Spaces.grid(hspace), nothing)
            end
        end
    end
end

# `Fields.level` builds a level grid around a pointer to the parent grid, so it
# should cost no more than the sliced DataLayout view and the level grid itself.
level_allocs(field, v) = @allocated Fields.level(field, v)

@testset "Levels and columns do not copy the grid" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        TU.levelable(space) || continue
        field = fill((; x = FT(1)), space)
        v = TU.fc_index(1, space)
        level_allocs(field, v) # compile
        # With the grid stored inline this was 272 bytes (a copy of the grid);
        # with the grid held by pointer it is the ~100-byte DataLayout view.
        @test level_allocs(field, v) ≤ 160
        if space isa Spaces.ExtrudedFiniteDifferenceSpace ||
           space isa Spaces.MultiColumnFiniteDifferenceSpace
            indices = space isa Spaces.ExtrudedFiniteDifferenceSpace2D ? (1, 1) : (1, 1, 1)
            column_field = Fields.column(field, indices...)
            # A column grid holds a pointer to the parent grid and its indices.
            @test sizeof(typeof(axes(column_field))) ≤
                  MAX_SPACE_BYTES + sizeof(indices)
        end
    end
end

@testset "Spaces are copied with host grids" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        has_grid(space) || continue
        field = fill((; x = FT(1)), space)
        # Adapting to a host array type returns an equivalent space on a host grid.
        host_field = Adapt.adapt(Array, field)
        host_space = axes(host_field)
        @test host_space isa typeof(space).name.wrapper
        @test sizeof(typeof(host_space)) ≤ MAX_SPACE_BYTES
        @test Spaces.grid(host_space) isa typeof(Spaces.grid(space)).name.wrapper
        @test Spaces.local_geometry_data(host_space).coordinates ==
              Adapt.adapt(Array, Spaces.local_geometry_data(space).coordinates)
    end
end

if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    @testset "Kernels receive the immutable grid twin" begin
        FT = Float64
        for space in TU.all_spaces(FT)
            has_grid(space) || continue
            device_space = Adapt.adapt(CUDA.KernelAdaptor(), space)
            @test isbits(device_space)
            @test Spaces.grid(device_space) isa Grids.AbstractGrid
            @test !ismutable(Spaces.grid(device_space))
            if TU.levelable(space)
                device_level = Spaces.level(device_space, TU.fc_index(1, space))
                @test isbits(device_level)
            end
            field = fill((; x = FT(1)), space)
            @test isbits(Adapt.adapt(CUDA.KernelAdaptor(), field))
        end
    end
end
