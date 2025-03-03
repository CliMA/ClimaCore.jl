using Test
using PrettyTables
using StatsBase: countmap


"""
    UnitTest(name, filename; meta)

A unit test, given:
 - `name::String` the name of the unit test
 - `filename::String` the filename of the unit test
 - `meta::Any` meta information for the test
"""
mutable struct UnitTest
    name::String
    filename::String
    elapsed::Float64
    compile_time::Float64
    recompile_time::Float64
    test_id::Int
    meta::Any
end
UnitTest(name, filename; meta = nothing) =
    UnitTest(name, filename, 0.0, 0.0, 0.0, 0, meta)

"""
    validate_tests(unit_tests::Vector{UnitTest}; test_path)

Given:
 - `unit_tests` a vector of `UnitTest`s
 - `test_path` the path to the test directory (for checking that files exist).
               Typically this should be `test_path = @__DIR__`.

Returns `err::Symbol` indicating the validation results:

 - `:duplicate_file` duplicate files found (and info statements are printed)
 - `:non_existent_file` found non-existent files (and info statements are printed)
 - `:pass` passes

Checking for non-existent files can help
prevent situations where a user adds a new
unit test to the end of their test suite,
but misspells the name. Instead of finding
out at the end of the test suite, users can
fail immediately due to a non-existent file.

Checking for duplicate files can be helpful
by avoiding unexpected duplicate work.
"""
function validate_tests(unit_tests::Vector{UnitTest}; test_path)
    # Test uniqueness of included files
    err = :pass
    filenames = map(x -> x.filename, unit_tests)
    if !allunique(filenames) # let's not do things more than once
        for (key, val) in countmap(filenames)
            val > 1 || continue
            @info "Duplicate file found: $key, ($val times)"
            err = :duplicate_file
        end
    end
    # Test that files exist
    if !all(map(x -> isfile(x.filename), unit_tests))
        for filename in filenames
            rfile = joinpath(test_path, filename)
            isfile(rfile) && continue
            @warn "Filename: $rfile does not exist"
            err = :non_existent_file
        end
    end
    return err
end


"""
    tabulate_tests(
        unit_tests::Vector{UnitTest};
        include_timings::Bool = true,
        time_format::Symbol = :second, # one of (:second, :compoundperiod)
    )

 - `include_timings::Bool` indicates whether or not to include the timings/percent columns
 - `time_format::Symbol` specify the time format. Valid values include [:second (default), :compoundperiod].
                         If we do not match either of these, then we warn & default to `:second`.

Tabulate the given unit tests. The `include_timings` kwarg
extends the table to include timings of the tests.
"""
function tabulate_tests(
    unit_tests::Vector{UnitTest};
    include_timings::Bool = true,
    time_format::Symbol = :second,
)
    title =
        include_timings ? "Tests results" :
        "Running the following unit tests..."
    if include_timings
        sort!(unit_tests; by = x -> x.compile_time, rev = true)
        local time_header
        elapsed_times = if time_format == :compoundperiod
            time_header = "Time"
            map(x -> time_and_units_str(x.elapsed), unit_tests)
        elseif time_format == :second
            time_header = "Time (s)"
            map(x -> x.elapsed, unit_tests)
        else
            @warn "Invalid time format `$time_format`. Falling back on `:second`"
            time_header = "Time (s)"
            map(x -> x.elapsed, unit_tests)
        end
        ∑times = sum(x -> x.elapsed, unit_tests)
        time_percent = map(x -> x.elapsed / ∑times * 100, unit_tests)
        test_id = map(x -> x.test_id, unit_tests)
        compile_time = map(x -> x.compile_time, unit_tests)
        ∑compile_time = sum(x -> x.compile_time, unit_tests)
        compile_time_percent =
            map(x -> x.compile_time / ∑compile_time * 100, unit_tests)
        header = [
            "% Comp",
            "Comp (s)",
            "% Time",
            time_header,
            "ID",
            "Name",
            "Filename",
        ]
        data = hcat(
            compile_time_percent,
            compile_time,
            time_percent,
            elapsed_times,
            test_id,
            map(x -> x.name, unit_tests),
            map(x -> x.filename, unit_tests),
        )
    else
        header = ["Name", "Filename"]
        data =
            hcat(map(x -> x.name, unit_tests), map(x -> x.filename, unit_tests))
    end
    PrettyTables.pretty_table(
        data;
        title,
        header,
        alignment = :l,
        crop = :horizontal,
    )
end


"""
    @timevd ex

A combination of `@timev` and `@timed`, which returns
everything computed in `@timev` into a NamedTuple.


This is similar to `Base.@timev`, but stores the result
in a NamedTuple.

!!! warn
    The compile-time measurements have been added for
    temporary use, in order to track latency. We will
    soon remove this as we fix some latency issues.
"""
macro timevd(ex)
    quote
        @timevd nothing $(esc(ex))
    end
end

macro timevd(msg, ex)
    # From julia Base:
    quote
        Base.Experimental.@force_compile
        local stats = Base.gc_num()
        local elapsedtime = Base.time_ns()
        Base.cumulative_compile_timing(true)
        local compile_elapsedtimes = Base.cumulative_compile_time_ns()
        local val = Base.@__tryfinally(
            $(esc(ex)),
            (
                elapsedtime = Base.time_ns() - elapsedtime;
                Base.cumulative_compile_timing(false);
                compile_elapsedtimes =
                    Base.cumulative_compile_time_ns() .- compile_elapsedtimes
            )
        )
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        local _msg = $(esc(msg))
        (;
            value = val,
            elapsed = elapsedtime / 1e9,
            bytes = diff.allocd,
            gctime = diff.total_time / 1e9,
            gcstats = diff,
            compile_elapsedtimes,
        )
    end
end

"""
    run_unit_tests!(
        unit_tests::Vector{UnitTest};
        fail_fast::Bool = true,
        prevent_leaky_tests::Bool = false
    )

Run all given unit tests, and updates each of the `UnitTest`'s elapsed time.

Options:
 - `fail_fast::Bool` indicates to fail fast (and not run additional tests after failing).
                     if `false`, then all tests will run and errors will be summarized at
                     the end
 - `prevent_leaky_tests::Bool` indicates to wrap all individual tests in a module to prevent
                     test from leaking.

Note:
    for `fail_fast = false`, the tests are all wrapped in `@testset "Unit tests"`
    so output is suppressed until all tests are complete.
"""
function run_unit_test!(
    test::UnitTest,
    test_id;
    prevent_leaky_tests::Bool = false,
)
    @debug "--- About to test $(test.filename)"
    stats = if prevent_leaky_tests
        # This is a home-brewed `SafeTestsets`, but it allows us to (inside a module)
        @timevd eval(Meta.parse(test_expr_safe(test)))
    else
        @timevd eval(Meta.parse(test_expr(test)))
        # test.elapsed = @elapsed include(test.filename)
    end
    (; compile_elapsedtimes, elapsed) = stats
    compile_time = first(compile_elapsedtimes) / 1e9
    recompile_time = last(compile_elapsedtimes) / 1e9
    perc_compile = compile_time / elapsed
    perc_recompile = recompile_time / compile_time

    test.elapsed = elapsed
    test.compile_time = compile_time
    test.recompile_time = recompile_time
    test.test_id = test_id
    @debug "--- Finished running test $(test.filename) in $(time_and_units_str(elapsed)) ($(100*perc_compile)% compilation time, $(100*perc_recompile)% recompilation)"
end

"""
    run_unit_tests!(
        unit_tests::Vector{UnitTest};
        fail_fast::Bool = true,
        prevent_leaky_tests::Bool = false
    )

Run all given unit tests, and updates each of the `UnitTest`'s elapsed time.

Note:
    for `fail_fast = false`, the tests are all wrapped in `@testset "Unit tests"`
    so output is suppressed until all tests are complete.
"""
function run_unit_tests!(
    unit_tests::Vector{UnitTest};
    fail_fast::Bool = true,
    prevent_leaky_tests::Bool = false,
)
    if fail_fast
        for (test_id, test) in enumerate(unit_tests)
            run_unit_test!(test, test_id; prevent_leaky_tests)
        end
    else
        @testset "Unit tests" begin
            for (test_id, test) in enumerate(unit_tests)
                run_unit_test!(test, test_id)
            end
        end
    end
end

nameify(name) = replace(name, "/" => "", ".jl" => "", " " => "")
# Hashes can result in `cannot assign a value to imported variable Main.include` error.
gensym_no_hashes(x) = replace(string(gensym(x)), "#" => "")
modulename(name) = gensym_no_hashes(nameify(name))

test_expr(test) =
    "@testset \"$(test.name)\" begin include(\"$(test.filename)\")\nend"
test_expr_safe(test) =
    "module $(modulename(test.filename))\nusing Test;@testset \"$(test.name)\" begin \ninclude(\"$(test.filename)\")\nend\nend"

import Dates

"""
    time_and_units_str(x::Real)

Returns a truncated string of time and units,
given a time `x` in Seconds.
"""
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
