using Test
import ClimaCore
using Aqua

@testset "Aqua tests - unbound args" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    Aqua.test_unbound_args(ClimaCore)

    # returns a vector of all unbound args
    # ua = Aqua.detect_unbound_args_recursively(ClimaCore)
end
