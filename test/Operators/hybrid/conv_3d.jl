include("utils_3d.jl")

device = ClimaComms.device()

@testset "Spatially varying boundary conditions" begin
    FT = Float64
    n_elems_seq = (5, 6, 7, 8)

    err = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        cs, fs = hvspace_3D((-pi, pi), (-pi, pi), (-pi, pi), 4, 4, 2^n)
        coords = Fields.coordinate_field(cs)

        c = sin.(coords.x .+ coords.z)

        bottom_face = level(fs, half)
        top_face = level(fs, 2^n + half)
        bottom_coords = Fields.coordinate_field(bottom_face)
        top_coords = Fields.coordinate_field(top_face)
        flux_bottom =
            @. Geometry.WVector(sin(bottom_coords.x + bottom_coords.z))
        flux_top = @. Geometry.WVector(sin(top_coords.x + top_coords.z))
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(flux_bottom),
            top = Operators.SetValue(flux_top),
        )
        Ic2f = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )

        div = divf2c.(Ic2f.(c) .* Geometry.WVector.(Fields.ones(fs)))

        err[k] = norm(div .- cos.(coords.x .+ coords.z))
    end
    @show err
    @test err[4] ≤ err[3] ≤ err[2] ≤ err[1]
end
