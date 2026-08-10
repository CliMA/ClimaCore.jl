using Test
using ClimaCore
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    ua = Aqua.detect_unbound_args_recursively(ClimaCore)
    length(ua) > 0 && @show ua
    @test length(ua) == 0

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ClimaCore; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ClimaCore", pkgdir(last(x).module)), ambs)

    # If the number of ambiguities is less than the limit below,
    # then please lower the limit based on the new number of ambiguities.
    # We're trying to drive this number down to zero to reduce latency.
    n_existing_ambiguities = 26
    if !(length(ambs) ≤ n_existing_ambiguities)
        for method_ambiguity in ambs
            @show method_ambiguity
        end
    end
    @test length(ambs) ≤ n_existing_ambiguities
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(ClimaCore)
    # julia-downgrade-compat' rewrites the Project.toml, promoting
    # our `[extras]` test dependencies into `[deps]` so that the resolved floors
    # survive `Pkg.test`. Those promoted deps aren't loaded by ClimaCore itself,
    # so they look stale to Aqua; skip this one check under the Downgrade
    # workflow. The other checks below are unaffected by the rewrite.
    if get(ENV, "CLIMACORE_DOWNGRADE_TESTS", "false") != "true"
        Aqua.test_stale_deps(ClimaCore)
    end
    Aqua.test_deps_compat(ClimaCore)
    Aqua.test_project_extras(ClimaCore)
    # Aqua.test_project_toml_formatting(ClimaCore) # failing
    Aqua.test_piracies(ClimaCore)
end

# Init values must be passed to the unrolled functions positionally, either
# directly or wrapped in UnrolledUtilities.Init, since a keyword argument is
# lowered into a call to Core.kwcall, which does not always specialize during
# GPU compilation of wide broadcast expressions and is a dynamic invocation
# when it does not. The pattern below covers every unrolled function that
# accepts an init value, and it tolerates one level of nested parentheses
# before the semicolon of a kwcall.
@testset "Kernel-reachable unrolled functions" begin
    kernel_reachable_dirs = [
        joinpath(pkgdir(ClimaCore), "src", "DataLayouts"),
        joinpath(pkgdir(ClimaCore), "src", "Geometry"),
        joinpath(pkgdir(ClimaCore), "src", "Utilities"),
        joinpath(pkgdir(ClimaCore), "ext", "cuda"),
    ]
    kwcall_pattern =
        r"\bunrolled_(reduce|mapreduce|accumulate|sum|prod)\((?:[^;()]|\([^()]*\))*;"
    offending_files = String[]
    for dir in kernel_reachable_dirs, (root, _, files) in walkdir(dir)
        for file in filter(endswith(".jl"), files)
            source_lines = readlines(joinpath(root, file))
            code_lines = map(line -> replace(line, r"#.*" => ""), source_lines)
            code = join(code_lines, '\n')
            contains(code, kwcall_pattern) &&
                push!(offending_files, joinpath(root, file))
        end
    end
    isempty(offending_files) || @show offending_files
    @test isempty(offending_files)
end

nothing
