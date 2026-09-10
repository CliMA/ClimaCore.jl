# Applying the same operator to a MultiColumnFiniteDifferenceSpace or a
# FiniteDifferenceSpace should produce the same results
using Test
import ClimaCore.MatrixFields: ⋅
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains, Fields, Geometry, Grids, MatrixFields, Meshes, Operators, Spaces
import ClimaCore.CommonSpaces: MultiColumnSpace
import ClimaCore.Operators:
    SetValue, SetGradient, Extrapolate,
    column_integral_definite!, column_integral_indefinite!,
    column_reduce!, column_accumulate!

# Run every operation on `ᶜspace` and return the results by name
function results(ᶜspace)
    FT = Spaces.undertype(ᶜspace)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜf = @. ᶜz^2
    ᶠw = @. Geometry.Covariant3Vector(ᶠz)

    # Vector conversions at every point
    ᶜuw = @. Geometry.UWVector(ᶜz, ᶜz^2)
    ᶜcov13 = @. Geometry.Covariant13Vector(ᶜuw)
    ᶜuvw = @. Geometry.UVWVector(Geometry.Covariant123Vector(1, 2, ᶜz))
    ᶜcontra3 = @. Geometry.Contravariant3Vector(Geometry.Covariant3Vector(ᶜf))
    ᶜw = @. Geometry.WVector(Geometry.Covariant3Vector(ᶜf))

    # Vertical finite-difference operators
    divf2c = Operators.DivergenceF2C(;
        bottom = SetValue(zero(eltype(ᶠw))),
        top = SetValue(zero(eltype(ᶠw))),
    )
    gradc2f = Operators.GradientC2F(;
        bottom = SetValue(zero(FT)),
        top = SetGradient(Geometry.Covariant3Vector(one(FT))),
    )
    interpc2f = Operators.InterpolateC2F(; bottom = Extrapolate(), top = Extrapolate())
    interpf2c = Operators.InterpolateF2C()
    upwind = Operators.UpwindBiasedProductC2F(; bottom = Extrapolate(), top = Extrapolate())
    div_mat = MatrixFields.operator_matrix(divf2c)
    interp_mat = MatrixFields.operator_matrix(interpc2f)

    # Horizontal (spectral) operators, which have no effect on a column
    ᶜcov12 = @. Geometry.Covariant12Vector(ᶜz, 1)

    # Column reductions.
    ᶠ∫f = similar(ᶠz)
    column_integral_indefinite!(ᶠ∫f, ᶜf)
    ∫f = similar(Fields.level(ᶠz, Operators.right_idx(ᶠspace)))
    column_integral_definite!(∫f, ᶜf)
    max_f = similar(Fields.level(ᶜz, 1))
    column_reduce!(max, max_f, ᶜf)
    ᶠacc = similar(ᶠz)
    column_accumulate!(+, ᶠacc, ᶜf; init = zero(FT))

    return (;
        ᶜcov13, ᶜuw_roundtrip = Geometry.UWVector.(ᶜcov13), ᶜuvw, ᶜcontra3, ᶜw,
        ᶜdiv = divf2c.(ᶠw),
        ᶠgrad = gradc2f.(ᶜf),
        ᶠinterp = interpc2f.(ᶜf),
        ᶜinterp = interpf2c.(ᶠ∫f),
        ᶠupwind = upwind.(ᶠw, ᶜf),
        ᶜdiv_mat = @.(div_mat() ⋅ ᶠw),
        ᶠinterp_mat = @.(interp_mat() ⋅ ᶜf),
        ᶜdivₕ = Operators.Divergence().(ᶜcov12),
        ᶜwdivₕ = Operators.WeakDivergence().(ᶜcov12),
        ᶜgradₕ = Operators.Gradient().(ᶜf),
        ᶠ∫f, ∫f, max_f, ᶠacc,
        ᶜΔz = Fields.Δz_field(ᶜspace),
        ᶜlevel = Fields.level(ᶜf, 3),
        ᶜsum = sum(ᶜf),
    )
end

values_of(x::Fields.Field) = vec(Array(parent(x)))
values_of(x::Number) = x
column_values(x::Fields.Field, h) = values_of(Spaces.column(x, 1, 1, h))

@testset "Multi-column space matches single columns" begin
    FT = Float64
    device = ClimaComms.CPUSingleThreaded()
    lats = FT[0, 45, -60]
    points = [Geometry.LatLongPoint(lat, zero(FT)) for lat in lats]
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(FT(4));
        boundary_names = (:bottom, :top),
    )
    # Stretched so that the vertical metric terms are not all equal
    z_mesh = Meshes.IntervalMesh(z_domain, Meshes.ExponentialStretching(FT(2)); nelems = 8)
    ᶜmulti = MultiColumnSpace(;
        points, z_elem = 8, z_min = zero(FT), z_max = FT(4), radius = FT(100),
        z_mesh, staggering = Grids.CellCenter(), device,
    )
    ᶜsingle = Spaces.CenterFiniteDifferenceSpace(device, z_mesh)

    multi = results(ᶜmulti)
    single = results(ᶜsingle)
    for name in keys(multi), h in eachindex(points)
        name == :ᶜsum && continue
        @test column_values(multi[name], h) ≈ values_of(single[name])
    end
    @test multi.ᶜsum ≈ length(points) * single.ᶜsum
end
