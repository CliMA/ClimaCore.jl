#=
julia --check-bounds=yes --project
julia -g2 --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Fields", "utils_field_multi_broadcast_fusion.jl"))
=#
using Test
using JET
using BenchmarkTools

using ClimaComms
ClimaComms.@import_required_backends
using OrderedCollections
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts: IJFH, VF, DataF
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

if !(@isdefined(TU))
    include(
        joinpath(
            pkgdir(ClimaCore),
            "test",
            "TestUtilities",
            "TestUtilities.jl",
        ),
    )
    import .TestUtilities as TU
end

@show ClimaComms.device()

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

function benchmark_kernel!(f!, X, Y, device)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device $f!($X, $Y)
    show(stdout, MIME("text/plain"), trial)
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
        Fields.@rprint_diff(X_fused, X_unfused)
        Fields.@rprint_diff(Y_fused, Y_unfused)
        @test X_fused == X_unfused
        @test Y_fused == Y_unfused
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
    var = (2,)
    Fields.bycolumn(axes(x1)) do colidx
        @fused_direct begin
            @. y1[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
            @. y2[colidx] = var # tests Base.Broadcast.AbstractArrayStyle{0}
            @. y2[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
        end
    end
    return nothing
end
function unfused_bycolumn!(X, Y)
    (; x1, x2, x3) = X
    (; y1, y2, y3) = Y
    var = (2,)
    Fields.bycolumn(axes(x1)) do colidx
        @. y1[colidx] = x1[colidx] + x2[colidx] + x3[colidx]
        @. y2[colidx] = var # tests Base.Broadcast.AbstractArrayStyle{0}
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
struct SomeData{FT}
    a::FT
    b::FT
    c::FT
end
