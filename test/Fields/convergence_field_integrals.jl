#=
julia --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Fields", "convergence_field_integrals.jl"))
=#
using Test
using JET

using ClimaComms
ClimaComms.@import_required_backends
import DataStructures
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts: IJFH
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

using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "Definite column integrals bycolumn" begin
    FT = Float64
    results = DataStructures.OrderedDict()
    ∫y_analytic = 1 - cos(1) - (0 - cos(0))
    function col_field_copy(y)
        col_copy = similar(y[Fields.ColumnIndex((1, 1), 1)])
        return Fields.Field(Fields.field_values(col_copy), axes(col_copy))
    end
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    for space_constructor in [
        TU.ColumnCenterFiniteDifferenceSpace,
        TU.ColumnFaceFiniteDifferenceSpace,
        TU.CenterExtrudedFiniteDifferenceSpace,
        TU.FaceExtrudedFiniteDifferenceSpace,
    ]
        device isa ClimaComms.CUDADevice && continue # broken on gpu
        for zelem in (2^2, 2^3, 2^4, 2^5)
            space = space_constructor(FT; zelem, context)
            # # Filter out spaces without z coordinates:
            # TU.has_z_coordinates(space) || continue
            # # Skip spaces incompatible with Fields.bycolumn:
            # TU.bycolumnable(space) || continue

            Y = fill((; y = FT(1)), space)
            zcf = Fields.coordinate_field(Y.y).z
            Δz = Fields.Δz_field(axes(zcf))
            Δz_col = Δz[Fields.ColumnIndex((1, 1), 1)]
            Δz_1 = ClimaComms.allowscalar(device) do
                parent(Δz_col)[1]
            end
            key = zelem
            if !haskey(results, key)
                results[key] =
                    Dict(:maxerr => 0, :Δz_1 => FT(0), :∫y => [], :y => [])
            end
            ∫y = Spaces.level(similar(Y.y), TU.fc_index(1, space))
            ∫y .= 0
            y = Y.y
            @. y .= 1 + sin(zcf)
            # Compute max error against analytic solution
            maxerr = FT(0)
            if space isa Spaces.ExtrudedFiniteDifferenceSpace
                Fields.bycolumn(axes(Y.y)) do colidx
                    Operators.column_integral_definite!(∫y[colidx], y[colidx])
                    maxerr = max(
                        maxerr,
                        maximum(abs.(parent(∫y[colidx]) .- ∫y_analytic)),
                    )
                    nothing
                end
            else
                Operators.column_integral_definite!(∫y, y)
                maxerr = max(maxerr, maximum(abs.(parent(∫y) .- ∫y_analytic)))
            end
            results[key][:Δz_1] = Δz_1
            results[key][:maxerr] = maxerr
            push!(results[key][:∫y], ∫y)
            push!(results[key][:y], y)
            nothing
        end
        maxerrs = map(key -> results[key][:maxerr], collect(keys(results)))
        Δzs_1 = map(key -> results[key][:Δz_1], collect(keys(results)))
        cr = convergence_rate(maxerrs, Δzs_1)
        if nameof(space_constructor) == :ColumnCenterFiniteDifferenceSpace
            @test 2 < sum(abs.(cr)) / length(cr) < 2.01
        elseif nameof(space_constructor) == :ColumnFaceFiniteDifferenceSpace
            @test_broken 2 < sum(abs.(cr)) / length(cr) < 2.01
        elseif nameof(space_constructor) == :CenterExtrudedFiniteDifferenceSpace
            @test 2 < sum(abs.(cr)) / length(cr) < 2.01
        elseif nameof(space_constructor) == :FaceExtrudedFiniteDifferenceSpace
            @test_broken 2 < sum(abs.(cr)) / length(cr) < 2.01
        else
            error("Uncaught case")
        end
    end
end
