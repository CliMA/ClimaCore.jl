#=
julia --check-bounds=yes --project
julia -g2 --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Fields", "unit_field_multi_broadcast_fusion.jl"))
=#
include("utils_field_multi_broadcast_fusion.jl")

@testset "FusedMultiBroadcast - restrict to only similar fields" begin
    FT = Float64
    dev = ClimaComms.device()
    cspace = CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 3,
        helem = 4,
        context = ClimaComms.context(dev),
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    x = rand_field(FT, cspace)
    y = rand_field(FT, fspace)
    # Cannot fuse center and face-spaced broadcasting
    @test_throws ErrorException begin
        @fused_direct begin
            @. x += 1
            @. y += 1
        end
    end
    nothing
end

@testset "FusedMultiBroadcast - restrict to only similar broadcast types" begin
    FT = Float64
    dev = ClimaComms.device()
    cspace = CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 3,
        helem = 4,
        context = ClimaComms.context(dev),
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    x = rand_field(FT, cspace)
    sd = Fields.Field(SomeData{FT}, cspace)
    x2 = rand_field(FT, cspace)
    y = rand_field(FT, fspace)
    # Error when the axes of the RHS are incompatible
    @test_throws ErrorException("Broacasted spaces are not the same.") begin
        @fused_direct begin
            @. x += 1
            @. x += y
        end
    end
    @test_throws ErrorException("Broacasted spaces are not the same.") begin
        @fused_direct begin
            @. x += y
            @. x += y
        end
    end
    # Different but compatible broadcasts
    @fused_direct begin
        @. x += 1
        @. x += x2
    end
    # Different fields but same spaces
    @fused_direct begin
        @. x += 1
        @. sd = SomeData{FT}(1, 2, 3)
    end
    @fused_direct begin
        @. x += 1
        @. sd.b = 3
    end
    nothing
end

@testset "FusedMultiBroadcast VIJFH and VF" begin
    FT = Float64
    device = ClimaComms.device()
    space = CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 3,
        helem = 4,
        context = ClimaComms.context(device),
    )
    X = Fields.FieldVector(
        x1 = rand_field(FT, space),
        x2 = rand_field(FT, space),
        x3 = rand_field(FT, space),
    )
    Y = Fields.FieldVector(
        y1 = rand_field(FT, space),
        y2 = rand_field(FT, space),
        y3 = rand_field(FT, space),
    )
    test_kernel!(; fused!, unfused!, X, Y)
    test_kernel!(; fused! = fused_bycolumn!, unfused! = unfused_bycolumn!, X, Y)

    nothing
end

@testset "FusedMultiBroadcast VIFH" begin
    FT = Float64
    device = ClimaComms.device()
    # Add GPU test when https://github.com/CliMA/ClimaCore.jl/issues/1383 is fixed
    if device isa ClimaComms.CPUSingleThreaded
        space = CenterExtrudedFiniteDifferenceSpaceLineHSpace(
            FT;
            zelem = 3,
            helem = 4,
            context = ClimaComms.context(device),
        )
        X = Fields.FieldVector(
            x1 = rand_field(FT, space),
            x2 = rand_field(FT, space),
            x3 = rand_field(FT, space),
        )
        Y = Fields.FieldVector(
            y1 = rand_field(FT, space),
            y2 = rand_field(FT, space),
            y3 = rand_field(FT, space),
        )
        test_kernel!(; fused!, unfused!, X, Y)
        test_kernel!(;
            fused! = fused_bycolumn!,
            unfused! = unfused_bycolumn!,
            X,
            Y,
        )

        nothing
    end
end

@testset "FusedMultiBroadcast IJFH" begin
    FT = Float64
    device = ClimaComms.device()
    sem_space =
        SphereSpectralElementSpace(FT; context = ClimaComms.context(device))
    IJFH_data() = Fields.Field(FT, sem_space)
    X = Fields.FieldVector(;
        x1 = IJFH_data(),
        x2 = IJFH_data(),
        x3 = IJFH_data(),
    )
    Y = Fields.FieldVector(;
        y1 = IJFH_data(),
        y2 = IJFH_data(),
        y3 = IJFH_data(),
    )
    test_kernel!(; fused!, unfused!, X, Y)
    nothing
end

@testset "FusedMultiBroadcast VF" begin
    FT = Float64
    device = ClimaComms.device()
    colspace = ColumnCenterFiniteDifferenceSpace(
        FT;
        zelem = 3,
        context = ClimaComms.context(device),
    )
    VF_data() = Fields.Field(FT, colspace)

    X = Fields.FieldVector(; x1 = VF_data(), x2 = VF_data(), x3 = VF_data())
    Y = Fields.FieldVector(; y1 = VF_data(), y2 = VF_data(), y3 = VF_data())
    test_kernel!(; fused!, unfused!, X, Y)
    nothing
end

@testset "FusedMultiBroadcast DataF" begin
    FT = Float64
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    DataF_data() = DataF{FT}(ArrayType(ones(FT, 1)))
    X = Fields.FieldVector(;
        x1 = DataF_data(),
        x2 = DataF_data(),
        x3 = DataF_data(),
    )
    Y = Fields.FieldVector(;
        y1 = DataF_data(),
        y2 = DataF_data(),
        y3 = DataF_data(),
    )
    test_kernel!(; fused!, unfused!, X, Y)
    nothing
end
