import ClimaCore
import ClimaCore: Spaces, Operators
using ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

using Test

@testset "placeholder_space" begin
    FT = Float64

    center_space = TU.ColumnCenterFiniteDifferenceSpace(FT)
    face_space = TU.ColumnFaceFiniteDifferenceSpace(FT)
    extruded_center_space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    extruded_face_space = TU.FaceExtrudedFiniteDifferenceSpace(FT)
    hspace = Spaces.horizontal_space(extruded_center_space)

    @test Operators.placeholder_space(center_space, center_space) ===
          Operators.PlaceholderSpace()

    @test Operators.placeholder_space(hspace, extruded_center_space) ===
          Operators.LevelPlaceholderSpace()

    @test Operators.placeholder_space(center_space, face_space) ===
          Operators.CenterPlaceholderSpace()
    @test Operators.placeholder_space(extruded_center_space, extruded_face_space) ===
          Operators.CenterPlaceholderSpace()

    @test Operators.placeholder_space(face_space, center_space) ===
          Operators.FacePlaceholderSpace()
    @test Operators.placeholder_space(extruded_face_space, extruded_center_space) ===
          Operators.FacePlaceholderSpace()
end

@testset "reconstruct_placeholder_space" begin
    FT = Float64

    center_space = TU.ColumnCenterFiniteDifferenceSpace(FT)
    face_space = TU.ColumnFaceFiniteDifferenceSpace(FT)
    extruded_center_space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    extruded_face_space = TU.FaceExtrudedFiniteDifferenceSpace(FT)

    @test Operators.reconstruct_placeholder_space(
        Operators.PlaceholderSpace(),
        center_space,
    ) == center_space

    r_level = Operators.reconstruct_placeholder_space(
        Operators.LevelPlaceholderSpace(),
        center_space,
    )
    @test r_level == Spaces.level(center_space, Operators.left_idx(center_space))

    @test Operators.reconstruct_placeholder_space(
        Operators.CenterPlaceholderSpace(),
        face_space,
    ) isa Spaces.CenterFiniteDifferenceSpace
    @test Operators.reconstruct_placeholder_space(
        Operators.CenterPlaceholderSpace(),
        extruded_face_space,
    ) isa Spaces.CenterExtrudedFiniteDifferenceSpace
    @test Operators.reconstruct_placeholder_space(
        Operators.CenterPlaceholderSpace(),
        center_space,
    ) == center_space
    @test Operators.reconstruct_placeholder_space(
        Operators.CenterPlaceholderSpace(),
        extruded_center_space,
    ) == extruded_center_space

    @test Operators.reconstruct_placeholder_space(
        Operators.FacePlaceholderSpace(),
        center_space,
    ) isa Spaces.FaceFiniteDifferenceSpace
    @test Operators.reconstruct_placeholder_space(
        Operators.FacePlaceholderSpace(),
        extruded_center_space,
    ) isa Spaces.FaceExtrudedFiniteDifferenceSpace
    @test Operators.reconstruct_placeholder_space(
        Operators.FacePlaceholderSpace(),
        face_space,
    ) == face_space
    @test Operators.reconstruct_placeholder_space(
        Operators.FacePlaceholderSpace(),
        extruded_face_space,
    ) == extruded_face_space
end
