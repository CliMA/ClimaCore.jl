#=
julia --check-bounds=yes --project=test
julia --project=test
using Revise; include(joinpath("test", "Fields", "field_multi_broadcast_fusion.jl"))
=#
using Test
using JET
using BenchmarkTools

using ClimaComms
using OrderedCollections
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts: IJFH
import ClimaCore.DataLayouts
import ClimaCore:
    Fields,
    slab,
    Domains,
    Topologies,
    Meshes,
    Operators,
    Spaces,
    Geometry,
    Quadratures

import ClimaCore.Fields: @fused_direct
using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff
using CUDA
using CUDA: @allowscalar

util_file =
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl")
if !(@isdefined(TU))
    include(util_file)
    import .TestUtilities as TU
end

function CenterExtrudedFiniteDifferenceSpaceLineHSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
    helem = 4,
    Nq = 4,
) where {FT}
    radius = FT(128)
    zlim = (0, 1)
    domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(FT)),
        Geometry.XPoint(FT(1));
        periodic = true,
    )
    hmesh = Meshes.IntervalMesh(domain; nelems = helem)

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(context, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    quad = Quadratures.GLL{Nq}()
    htopology = Topologies.IntervalTopology(context, hmesh)
    hspace = Spaces.SpectralElementSpace1D(htopology, quad)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
end

function benchmark_kernel!(f!, X, Y)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = benchmark_kernel!(f!, X, Y, ClimaComms.device(X.x1))
    show(stdout, MIME("text/plain"), trial)
end
benchmark_kernel!(f!, X, Y, ::ClimaComms.CUDADevice) =
    CUDA.@sync BenchmarkTools.@benchmark $f!($X, $Y);
benchmark_kernel!(f!, X, Y, ::ClimaComms.AbstractCPUDevice) =
    BenchmarkTools.@benchmark $f!($X, $Y);

function show_diff(A, B)
    for pn in propertynames(A)
        Ai = getproperty(A, pn)
        Bi = getproperty(B, pn)
        println("==================== Comparing $pn")
        @show Ai
        @show Bi
        @show abs.(Ai .- Bi)
    end
end

function compare(A, B)
    pass = true
    for pn in propertynames(A)
        pass =
            pass &&
            all(parent(getproperty(A, pn)) .== parent(getproperty(B, pn)))
    end
    pass || show_diff(A, B)
    return pass
end
function test_kernel!(; fused!, unfused!, X, Y)
    for pn in propertynames(X)
        rand_field!(getproperty(X, pn))
    end
    for pn in propertynames(Y)
        rand_field!(getproperty(Y, pn))
    end
    X_fused = similar(X)
    X_fused .= X
    X_unfused = similar(X)
    X_unfused .= X
    Y_fused = similar(Y)
    Y_fused .= Y
    Y_unfused = similar(Y)
    Y_unfused .= Y
    unfused!(X_unfused, Y_unfused)
    fused!(X_fused, Y_fused)
    @testset "Test correctness of $(nameof(typeof(fused!)))" begin
        @test compare(X_fused, X_unfused)
        @test compare(Y_fused, Y_unfused)
    end
end

function fused!(X, Y)
    (; x1, x2, x3) = X
    (; y1, y2, y3) = Y
    @fused_direct begin
        @. y1 = x1 + x2 + x3
        @. y2 = x1 + x2 + x3
    end
    return nothing
end
function unfused!(X, Y)
    (; x1, x2, x3) = X
    (; y1, y2, y3) = Y
    @. y1 = x1 + x2 + x3
    @. y2 = x1 + x2 + x3
    return nothing
end
function fused_bycolumn!(X, Y)
    (; x1, x2, x3) = X
    (; y1, y2, y3) = Y
    Fields.bycolumn(axes(x1)) do colidx
        @fused_direct begin
            @. y1[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
            @. y2[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
        end
    end
    return nothing
end
function unfused_bycolumn!(X, Y)
    (; x1, x2, x3) = X
    (; y1, y2, y3) = Y
    Fields.bycolumn(axes(x1)) do colidx
        @. y1[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
        @. y2[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
    end
    return nothing
end

function rand_field(FT, space)
    f = Fields.Field(FT, space)
    rand_field!(f)
end

function rand_field!(f)
    parent(f) .= map(x -> rand(), parent(f))
    return f
end

@testset "FusedMultiBroadcast - restrict to only similar fields" begin
    FT = Float64
    dev = ClimaComms.device()
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
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

struct SomeData{FT}
    a::FT
    b::FT
    c::FT
end
@testset "FusedMultiBroadcast - restrict to only similar broadcast types" begin
    FT = Float64
    dev = ClimaComms.device()
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
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
    @test_throws DimensionMismatch begin
        @fused_direct begin
            @. x += 1
            @. x += y
        end
    end
    @test_throws DimensionMismatch begin
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
    space = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 3,
        helem = 4,
        context = ClimaComms.context(),
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

    benchmark_kernel!(unfused!, X, Y)
    benchmark_kernel!(fused!, X, Y)

    benchmark_kernel!(unfused_bycolumn!, X, Y)
    benchmark_kernel!(fused_bycolumn!, X, Y)
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

        benchmark_kernel!(unfused!, X, Y)
        benchmark_kernel!(fused!, X, Y)

        benchmark_kernel!(unfused_bycolumn!, X, Y)
        benchmark_kernel!(fused_bycolumn!, X, Y)
        nothing
    end
end
