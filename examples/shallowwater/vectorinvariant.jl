

using ClimateMachineCore



function tendency!(dY, Y, _, t)
    hu = Y.h .* Y.u
    divergence!(dY.v, hu)

    norm.(Y.y)


end