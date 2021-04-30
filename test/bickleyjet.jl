

function flux(state, g)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    u = ρu ./ ρ

    return (
        ρ  = ρu,
        ρu = (ρu ⊗ u) .+ (g * ρ^2 / 2) * I,
        ρθ = ρθ .* u,
    )
end

