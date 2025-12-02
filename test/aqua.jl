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
    # Uncomment for debugging:
    n_existing_ambiguities = 25
    if !(length(ambs) ≤ n_existing_ambiguities)
        for method_ambiguity in ambs
            @show method_ambiguity
        end
    end
    @test length(ambs) ≤ n_existing_ambiguities
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(ClimaCore)
    Aqua.test_stale_deps(ClimaCore)
    Aqua.test_deps_compat(ClimaCore)
    Aqua.test_project_extras(ClimaCore)
    # Aqua.test_project_toml_formatting(ClimaCore) # failing
    Aqua.test_piracies(ClimaCore)
end

nothing
