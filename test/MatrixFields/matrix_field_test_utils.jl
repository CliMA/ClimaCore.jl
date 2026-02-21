using Test
using JET
import Dates
import Random: seed!
import Base.Broadcast: materialize, materialize!
import LazyBroadcast: @lazy
import BenchmarkTools as BT

import ClimaComms
import BenchmarkTools as BT
ClimaComms.@import_required_backends
import ClimaCore:
    Utilities,
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Hypsography,
    Spaces,
    Fields,
    Operators,
    Quadratures
using ClimaCore.MatrixFields
import ClimaCore.Utilities: half, inferred_const, auto_broadcast
import LinearAlgebra: I, norm, ldiv!, mul!
import ClimaCore.MatrixFields: @name

# Test that an expression is true and that it is also type-stable.
macro test_all(expression)
    return quote
        local test_func() = $(esc(expression))
        @test test_func()                   # correctness
        @test (@allocated test_func()) == 0 # allocations
        @test_opt test_func()               # type instabilities
    end
end

# Compute the minimum time (in seconds) required to run an expression after it
# has been compiled. This macro is used instead of @benchmark from
# BenchmarkTools.jl because the latter is extremely slow (it appears to keep
# triggering recompilations and allocating a lot of memory in the process).
macro benchmark(expression)
    return quote
        $(esc(expression)) # Compile the expression first. Use esc for hygiene.
        best_time = Inf
        start_time = time_ns()
        while time_ns() - start_time < 1e8 # Benchmark for 0.1 s (1e8 ns).
            best_time = min(best_time, @elapsed $(esc(expression)))
        end
        best_time
    end
end

const comms_device = ClimaComms.device()
# comms_device = ClimaComms.CPUSingleThreaded()
@show comms_device
const using_cuda = comms_device isa ClimaComms.CUDADevice
cuda_module(ext) = using_cuda ? ext.CUDA : ext
const cuda_mod = cuda_module(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))
const cuda_frames = using_cuda ? (AnyFrameModule(cuda_mod),) : ()
const cublas_frames = using_cuda ? (AnyFrameModule(cuda_mod.CUBLAS),) : ()
const invalid_ir_error = using_cuda ? cuda_mod.InvalidIRError : ErrorException

# Test the allocating and non-allocating versions of a field broadcast against
# a reference non-allocating implementation. Ensure that they are performant,
# correct, and type-stable, and print some useful information. If a reference
# implementation is not available, the performance and correctness checks are
# skipped.
function test_field_broadcast(;
    test_name,
    get_result,
    set_result,
    ref_set_result = nothing,
    time_ratio_limit = 10,
    max_eps_error_limit = 10,
    test_broken_with_cuda = false,
)
    @testset "$test_name" begin
        if test_broken_with_cuda && using_cuda
            @test_throws invalid_ir_error materialize(get_result)
            @warn "$test_name:\n\tCUDA.InvalidIRError"
            return
        end

        result = materialize(get_result)
        result_copy = copy(result)
        time = @benchmark materialize!(result, set_result)
        time_rounded = round(time; sigdigits = 2)

        # Test that set_result! sets the same value as get_result.
        @test result == result_copy

        if isnothing(ref_set_result)
            @info "$test_name:\n\tTime = $time_rounded s (reference \
                   implementation unavailable)"
        else
            ref_result = similar(result)
            ref_time = @benchmark materialize!(ref_result, ref_set_result)
            ref_time_rounded = round(ref_time; sigdigits = 2)
            time_ratio = time / ref_time
            time_ratio_rounded = round(time_ratio; sigdigits = 2)
            max_error = mapreduce(
                (a, b) -> (abs(a - b)),
                max,
                parent(result),
                parent(ref_result),
            )
            max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

            @info "$test_name:\n\tTime Ratio = $time_ratio_rounded \
                   ($time_rounded s vs. $ref_time_rounded s for reference) \
                   \n\tMaximum Error = $max_eps_error eps"

            # Test that set_result! is performant and correct when compared
            # against ref_set_result.
            @test time / ref_time <= time_ratio_limit
            @test max_eps_error <= max_eps_error_limit
        end

        # Test get_result and set_result! for type instabilities, and test
        # set_result! for allocations. Ignore the type instabilities in CUDA and
        # the allocations they incur.
        @test_opt ignored_modules = cuda_frames materialize(get_result)
        @test_opt ignored_modules = cuda_frames materialize!(result, set_result)
        using_cuda || @test (@allocated materialize!(result, set_result)) == 0

        if !isnothing(ref_set_result)
            # Test ref_set_result! for type instabilities and allocations to
            # ensure that the performance comparison is fair.
            @test_opt ignored_modules = cuda_frames materialize!(
                ref_result,
                ref_set_result,
            )
            using_cuda ||
                @test (@allocated materialize!(ref_result, ref_set_result)) == 0
        end
    end
end

# Create a field matrix for a similar solve to ClimaAtmos's moist dycore + prognostic,
# EDMF + prognostic surface temperature with implicit acoustic waves and SGS fluxes
# also returns corresponding FieldVector
function dycore_prognostic_EDMF_FieldMatrix(
    ::Type{FT},
    center_space = nothing,
    face_space = nothing,
) where {FT}
    seed!(1) # For reproducibility with random fields
    if isnothing(center_space) || isnothing(face_space)
        center_space, face_space = test_spaces(FT)
    end
    surface_space = Spaces.level(face_space, half)
    surface_space = Spaces.level(face_space, half)
    sfc_vec = random_field(FT, surface_space)
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    λ = 10
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)
    # Geometry.Covariant123Vector(1, 2, 3) * Geometry.Covariant12Vector(1, 2)'
    e¹² = Geometry.Covariant12Vector(1, 1)
    e₁₂ = Geometry.Contravariant12Vector(1, 1)
    e³ = Geometry.Covariant3Vector(1)
    e₃ = Geometry.Contravariant3Vector(1)

    ρχ_unit = (; ρq_tot = 1, ρq_liq = 1, ρq_ice = 1, ρq_rai = 1, ρq_sno = 1)
    ρaχ_unit =
        (; ρaq_tot = 1, ρaq_liq = 1, ρaq_ice = 1, ρaq_rai = 1, ρaq_sno = 1)


    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶠmat2_ρχ_u₃ = ᶜᶠmat2 .* (ρχ_unit,) .* (e₃',)
    ᶜᶜmat3_uₕ_scalar = ᶜᶜmat3 .* (e¹²,)
    ᶜᶜmat3_uₕ_uₕ =
        ᶜᶜmat3 .* (
            Geometry.Covariant12Vector(1, 0) *
            Geometry.Contravariant12Vector(1, 0)' +
            Geometry.Covariant12Vector(0, 1) *
            Geometry.Contravariant12Vector(0, 1)',
        )
    ᶜᶠmat2_uₕ_u₃ = ᶜᶠmat2 .* (e¹² * e₃',)
    ᶜᶜmat3_ρχ_scalar = ᶜᶜmat3 .* (ρχ_unit,)
    ᶜᶜmat3_ρaχ_scalar = ᶜᶜmat3 .* (ρaχ_unit,)
    ᶜᶠmat2_ρaχ_u₃ = ᶜᶠmat2 .* (ρaχ_unit,) .* (e₃',)

    dry_center_gs_unit = (; ρ = 1, ρe_tot = 1, uₕ = e¹²)
    center_gs_unit = (; dry_center_gs_unit..., ρatke = 1, ρχ = ρχ_unit)
    center_sgsʲ_unit = (; ρa = 1, ρae_tot = 1, ρaχ = ρaχ_unit)

    b = Fields.FieldVector(;
        sfc = sfc_vec .* ((; T = 1),),
        c = ᶜvec .* ((; center_gs_unit..., sgsʲs = (center_sgsʲ_unit,)),),
        f = ᶠvec .* ((; u₃ = e³, sgsʲs = ((; u₃ = e³),)),),
    )
    A = MatrixFields.FieldMatrix(
        # GS-GS blocks:
        (@name(sfc), @name(sfc)) => I,
        (@name(c.ρ), @name(c.ρ)) => I,
        (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.ρatke)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.ρχ)) => ᶜᶜmat3,
        (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3_uₕ_uₕ,
        (@name(c.ρ), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
        (@name(c.ρe_tot), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
        (@name(c.ρatke), @name(f.u₃)) => ᶜᶠmat2_scalar_u₃,
        (@name(c.ρχ), @name(f.u₃)) => ᶜᶠmat2_ρχ_u₃,
        (@name(f.u₃), @name(c.ρ)) => ᶠᶜmat2_u₃_scalar,
        (@name(f.u₃), @name(c.ρe_tot)) => ᶠᶜmat2_u₃_scalar,
        (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        # GS-SGS blocks:
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) => ᶜᶜmat3,
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_ρχ_scalar,
        (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_uₕ_scalar,
        (@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_scalar_u₃,
        (@name(c.ρatke), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_scalar_u₃,
        (@name(c.ρχ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_ρχ_u₃,
        (@name(c.uₕ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_uₕ_u₃,
        (@name(f.u₃), @name(c.sgsʲs.:(1).ρa)) => ᶠᶜmat2_u₃_scalar,
        (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
        # SGS-SGS blocks:
        (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)) => I,
        (@name(c.sgsʲs.:(1).ρae_tot), @name(c.sgsʲs.:(1).ρae_tot)) => I,
        (@name(c.sgsʲs.:(1).ρaχ), @name(c.sgsʲs.:(1).ρaχ)) => I,
        (@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)) =>
            ᶜᶠmat2_scalar_u₃,
        (@name(c.sgsʲs.:(1).ρae_tot), @name(f.sgsʲs.:(1).u₃)) =>
            ᶜᶠmat2_scalar_u₃,
        (@name(c.sgsʲs.:(1).ρaχ), @name(f.sgsʲs.:(1).u₃)) => ᶜᶠmat2_ρaχ_u₃,
        (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρa)) =>
            ᶠᶜmat2_u₃_scalar,
        (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).ρae_tot)) =>
            ᶠᶜmat2_u₃_scalar,
        (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
    )
    return A, b
end

function scaling_only_dycore_prognostic_EDMF_FieldMatrix(
    ::Type{FT},
    center_space = nothing,
    face_space = nothing,
) where {FT}
    seed!(1) # For reproducibility with random fields
    if isnothing(center_space) || isnothing(face_space)
        center_space, face_space = test_spaces(FT)
    end
    surface_space = Spaces.level(face_space, half)
    surface_space = Spaces.level(face_space, half)
    sfc_vec = random_field(FT, surface_space)
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    λ = 10
    # Geometry.Covariant123Vector(1, 2, 3) * Geometry.Covariant12Vector(1, 2)'
    e¹² = Geometry.Covariant12Vector(FT(1), FT(1))
    e₁₂ = Geometry.Contravariant12Vector(FT(1), FT(1))
    e³ = Geometry.Covariant3Vector(FT(1))
    e₃ = Geometry.Contravariant3Vector(FT(1))

    ρχ_unit = (;
        ρq_tot = FT(1),
        ρq_liq = FT(1),
        ρq_ice = FT(1),
        ρq_rai = FT(1),
        ρq_sno = FT(1),
    )
    ρaχ_unit = (;
        ρaq_tot = FT(1),
        ρaq_liq = FT(1),
        ρaq_ice = FT(1),
        ρaq_rai = FT(1),
        ρaq_sno = FT(1),
    )



    ᶠᶠu₃_u₃ = DiagonalMatrixRow(e³ * e₃')
    ᶜᶜuₕ_scalar = DiagonalMatrixRow(e¹²)
    ᶜᶜuₕ_uₕ = DiagonalMatrixRow(
        Geometry.Covariant12Vector(FT(1), FT(0)) *
        Geometry.Contravariant12Vector(FT(1), FT(0))' +
        Geometry.Covariant12Vector(FT(0), FT(1)) *
        Geometry.Contravariant12Vector(FT(0), FT(1))',
    )
    ᶜᶜρχ_scalar = DiagonalMatrixRow(ρχ_unit)
    ᶜᶜρaχ_scalar = DiagonalMatrixRow(ρaχ_unit)

    dry_center_gs_unit = (; ρ = FT(1), ρe_tot = FT(1), uₕ = e¹²)
    center_gs_unit = (; dry_center_gs_unit..., ρatke = FT(1), ρχ = ρχ_unit)
    center_sgsʲ_unit = (; ρa = FT(1), ρae_tot = FT(1), ρaχ = ρaχ_unit)

    b = Fields.FieldVector(;
        sfc = sfc_vec .* ((; T = 1),),
        c = ᶜvec .* ((; center_gs_unit..., sgsʲs = (center_sgsʲ_unit,)),),
        f = ᶠvec .* ((; u₃ = e³, sgsʲs = ((; u₃ = e³),)),),
    )
    A = MatrixFields.FieldMatrix(
        # GS-GS blocks:
        (@name(sfc), @name(sfc)) => I,
        (@name(c.ρ), @name(c.ρ)) => I,
        (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜuₕ_uₕ,
        (@name(f.u₃), @name(f.u₃)) => ᶠᶠu₃_u₃,
        # GS-SGS blocks:
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) =>
            DiagonalMatrixRow(rand(FT)),
        (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜρχ_scalar,
        (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜuₕ_scalar,
        (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠu₃_u₃,
        # SGS-SGS blocks:
        (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)) => I,
        (@name(c.sgsʲs.:(1).ρae_tot), @name(c.sgsʲs.:(1).ρae_tot)) => I,
        (@name(c.sgsʲs.:(1).ρaχ), @name(c.sgsʲs.:(1).ρaχ)) => I,
        (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠu₃_u₃,
    )
    return A, b
end

# Generate extruded finite difference spaces for testing. Include topography
# when possible.
function test_spaces(::Type{FT}) where {FT}
    velem = 20 # This should be big enough to test high-bandwidth matrices.
    helem = npoly = 1 # These should be small enough for the tests to be fast.

    comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
    hdomain = Domains.SphereDomain(FT(10))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(comms_ctx, hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vtopology = Topologies.IntervalTopology(comms_ctx, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    sfc_coord = Fields.coordinate_field(hspace)
    hypsography =
        using_cuda ? Hypsography.Flat() :
        Hypsography.LinearAdaption(
            Geometry.ZPoint.(@. cosd(sfc_coord.lat) + cosd(sfc_coord.long) + 1),
        ) # TODO: FD operators don't currently work with hypsography on GPUs.
    center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

# Generate a random field with elements of type T.
function random_field(::Type{T}, space) where {T}
    FT = Spaces.undertype(space)
    field = Fields.Field(T, space)
    parent(field) .= rand.(FT)
    return field
end

# Analogue of zero for nested iterators
const nested_zero = inferred_const(auto_broadcast(zero))

# Construct a nested iterator for testing compatibility with generic data types.
nested_type(value) = nested_type(value, value, value)
nested_type(value1, value2, value3) =
    (; a = (), b = value1, c = (value2, (; d = (value3,)), (;)))

# A shorthand for typeof(nested_type(::FT)).
const NestedType{FT} = NamedTuple{
    (:a, :b, :c),
    Tuple{
        Tuple{},
        FT,
        Tuple{FT, NamedTuple{(:d,), Tuple{Tuple{FT}}}, NamedTuple{(), Tuple{}}},
    },
}

function call_ref_set_result!(
    ref_set_result!::F,
    ref_result_arrays,
    inputs_arrays,
    temp_values_arrays,
) where {F}
    for arrays in
        zip(ref_result_arrays, inputs_arrays..., temp_values_arrays...)
        ref_set_result!(arrays...)
    end
    return nothing
end

function print_time_comparison(; time, ref_time)
    time_rounded = round(time; sigdigits = 2)
    ref_time_rounded = round(ref_time; sigdigits = 2)
    time_ratio = time / ref_time
    time_ratio_rounded = round(time_ratio; sigdigits = 2)
    @info "Times (ClimaCore,Array,ClimaCore/Array): = ($time_rounded, $ref_time_rounded, $time_ratio_rounded)."
    return nothing
end

function compute_max_error(result_arrays, ref_result_arrays)
    return mapreduce(max, result_arrays, ref_result_arrays) do array, ref_array
        mapreduce((a, b) -> (abs(a - b)), max, array, ref_array)
    end
end

set_result!(result, bc) = (materialize!(result, bc); nothing)

function call_getidx(space, bc, idx, hidx)
    @inbounds Operators.getidx(space, bc, idx, hidx)
    return nothing
end

time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))

"""
    compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}

A canonicalized `Dates.CompoundPeriod` given a real value
`x`, and its units via the period type `T`.
"""
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end

trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s

function get_getidx_args(bc)
    space = axes(bc)
    # TODO: change this to idx_l, idx_i, idx_r
    # may need to define a helper
    (li, lw, rw, ri) = Operators.window_bounds(space, bc)
    idx_l, idx_r = if Topologies.isperiodic(space)
        li, ri
    else
        lw, rw
    end
    idx_i = if space.staggering isa Spaces.CellCenter
        Int(round((idx_l + idx_r) / 2; digits = 0))
    else
        Utilities.PlusHalf(Int(round((idx_l + idx_r) / 2; digits = 0)))
    end
    hidx = (1, 1, 1)
    return (; space, bc, idx_l, idx_i, idx_r, hidx)
end

import JET
function perf_getidx(bc; broken = false)
    (; space, bc, idx_l, idx_i, idx_r, hidx) = get_getidx_args(bc)
    call_getidx(space, bc, idx_l, hidx)
    call_getidx(space, bc, idx_i, hidx)
    call_getidx(space, bc, idx_r, hidx)

    bel =
        time_and_units_str(BT.@belapsed call_getidx($space, $bc, $idx_l, $hidx))
    bei =
        time_and_units_str(BT.@belapsed call_getidx($space, $bc, $idx_i, $hidx))
    ber =
        time_and_units_str(BT.@belapsed call_getidx($space, $bc, $idx_r, $hidx))
    JET.@test_opt call_getidx(space, bc, idx_l, hidx)
    JET.@test_opt call_getidx(space, bc, idx_i, hidx)
    JET.@test_opt call_getidx(space, bc, idx_r, hidx)
    @info "getidx times max(left,interior,right) = ($bel,$bei,$ber)"
    return nothing
end
