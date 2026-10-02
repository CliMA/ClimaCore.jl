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
    # When the whole suite runs in one process, the Operators benchmark
    # utilities load StatsBase before this file, and `StatsBase.TestStat(v)`
    # is then ambiguous with `(::Type{<:Number})(::AutoBroadcaster)`. The
    # constructor disambiguators in Utilities/auto_broadcaster.jl are generated
    # when ClimaCore is precompiled, so they cannot cover packages that are not
    # ClimaCore dependencies. StatsBase is excluded so that the count does not
    # depend on which tests ran first.
    from_statsbase(m) = nameof(Base.moduleroot(m.module)) === :StatsBase
    filter!(x -> !any(from_statsbase, x), ambs)

    # If the number of ambiguities is less than the limit below,
    # then please lower the limit based on the new number of ambiguities.
    # We're trying to drive this number down to zero to reduce latency.
    # ClimaCoreCUDAExt adds ambiguities of its own, so the count has to be
    # checked with CUDA loaded (CLIMACOMMS_DEVICE=CUDA) as well.
    n_existing_ambiguities = 0
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
