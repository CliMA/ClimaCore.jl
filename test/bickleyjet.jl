using ClimateMachineCore.Geometry, LinearAlgebra, UnPack

state = (ρ=1.0, ρu=Cartesian12Vector(1.2,2.4), ρθ=0.1)

function flux(state, g)
    @unpack ρ, ρu, ρθ = state

    u = ρu ./ ρ

    return (
        ρ  = ρu,
        ρu = (ρu ⊗ u) + (g * ρ^2 / 2) * I,
        ρθ = ρθ .* u,
    )
end

flux(state, 9.81)

# Need to define:
#  - broadcasting on VectorPoint
#  - ⊗ on VectorPoint
#  - promotiion of I  (UniformScaling)
