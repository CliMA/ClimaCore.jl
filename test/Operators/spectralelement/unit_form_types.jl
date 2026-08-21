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

# The form-dependent factors that let a single operator body serve both forms.
@testset "form-dependent factors" begin
    for FT in (Float32, Float64)
        D = FT[1 2 3; 4 5 6; 7 8 9]
        local_geometry = (; J = FT(2), invJ = FT(0.5), WJ = FT(3))
        W = local_geometry.WJ * local_geometry.invJ
        x = FT(7)

        # The weak form transposes D and flips its sign, which is the only
        # change integration by parts makes to the accumulation loop.
        @test O.form_deriv_entry(O.StrongForm(), D, 1, 3) == D[1, 3]
        @test O.form_deriv_entry(O.WeakForm(), D, 1, 3) == -D[3, 1]

        # Only the weak form weights its argument, by W = WJ J⁻¹.
        @test O.form_weighted_arg(O.StrongForm(), local_geometry, x) === x
        @test O.form_weighted_arg(O.WeakForm(), local_geometry, x) == W * x

        @test O.form_jacobian(O.StrongForm(), local_geometry) == local_geometry.J
        @test O.form_jacobian(O.WeakForm(), local_geometry) == local_geometry.WJ

        @test O.form_jacobian_rescale(O.StrongForm(), local_geometry, x) ==
              x / local_geometry.J
        @test O.form_jacobian_rescale(O.WeakForm(), local_geometry, x) ==
              x / local_geometry.WJ

        # Only the weak gradient divides out W; the strong form is the identity,
        # so it must return its argument untouched rather than rescale by one.
        @test O.form_weight_rescale(O.StrongForm(), local_geometry, x) === x
        @test O.form_weight_rescale(O.WeakForm(), local_geometry, x) == x / W

        # Unified operator bodies accumulate with form_deriv_entry and then
        # apply the strong form's sign pattern to the result. Folding the weak
        # form's sign flip into each term must give bitwise the same value as
        # negating the completed transposed sum, or the unified bodies would
        # change the weak form's rounding.
        values = FT.([0.1, 0.2, 0.3])
        weak_sum = O.form_deriv_entry(O.WeakForm(), D, 1, 1) * values[1]
        transposed_sum = D[1, 1] * values[1]
        for k in 2:3
            weak_sum += O.form_deriv_entry(O.WeakForm(), D, 1, k) * values[k]
            transposed_sum += D[k, 1] * values[k]
        end
        @test weak_sum === -transposed_sum
    end
end
