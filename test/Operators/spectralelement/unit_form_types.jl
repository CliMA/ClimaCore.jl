#=
julia --project
using Revise; include(joinpath("test", "Operators", "spectralelement", "unit_form_types.jl"))
=#
using Test
using ClimaComms
import ClimaCore:
    Domains, Meshes, Topologies, Spaces, Quadratures, Operators as O

function test_space()
    domain = Domains.SphereDomain(1.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 2)
    topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{3}())
end

# The strong and weak variants of an operator differ only in a FormType type
# parameter, with the weak variants available under their original names.
@testset "FormType parameter and weak-form aliases" begin
    for (strong, weak) in
        (
        (O.Divergence, O.WeakDivergence),
        (O.Gradient, O.WeakGradient),
        (O.Curl, O.WeakCurl),
    )
        # The alias is the WeakForm variant, not a distinct type.
        @test weak{()} === strong{(), O.WeakForm}
        @test weak{(1, 2)} === strong{(1, 2), O.WeakForm}

        # Constructors default to the strong form, and every spelling of a form
        # produces the same singleton.
        @test strong() === strong{(), O.StrongForm}()
        @test strong{(1, 2)}() === strong{(1, 2), O.StrongForm}()
        @test weak() === strong{(), O.WeakForm}()
        @test weak{(1, 2)}() === strong{(1, 2), O.WeakForm}()

        # Unifying the types makes the weak variants subtypes of the strong
        # names, so code that must distinguish forms has to name the FormType.
        @test weak() isa strong
        @test !(weak() isa strong{(), O.StrongForm})
        @test !(strong() isa strong{(), O.WeakForm})

        # rebuild_operator resets the operator axes to those of the space while
        # preserving the form, which is what Base.Broadcast.instantiate relies on.
        space = test_space()
        for (form, op) in ((O.StrongForm, strong()), (O.WeakForm, weak()))
            @test O.rebuild_operator(op, space) === strong{(1, 2), form}()
        end
    end
end
