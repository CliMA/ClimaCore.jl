using ClimaCore
using ClimaCore.CommonSpaces, ClimaCore.Geometry
using ClimaCore: Fields, Spaces, Operators, Grids
using ClimaCore.Utilities: PlusHalf
using Test

const FT = Float64

# ---------------------------------------------------------------
# Setup: N=3 columns at arbitrary (lat, lon) locations
# Compare with a single ColumnSpace for behaviour parity.
# ---------------------------------------------------------------
points = [
    Geometry.LatLongPoint(FT(0),  FT(0)),
    Geometry.LatLongPoint(FT(10), FT(20)),
    Geometry.LatLongPoint(FT(-5), FT(90)),
]
N = length(points)

ᶜspace = PointColumnEnsembleSpace(FT;
    points  = points,
    z_elem  = 10,
    z_min   = FT(0),
    z_max   = FT(10_000),
    radius  = FT(6.371229e6),
    staggering = CellCenter(),
)
ᶠspace = Spaces.face_space(ᶜspace)

# ColumnSpace reference for single-column behaviour
col_cspace = ColumnSpace(FT; z_elem = 10, z_min = FT(0), z_max = FT(10_000), staggering = CellCenter())
col_fspace = Spaces.face_space(col_cspace)

@testset "PointColumnEnsembleSpace – construction" begin
    @test ᶜspace isa Spaces.CenterExtrudedFiniteDifferenceSpace
    @test ᶠspace isa Spaces.FaceExtrudedFiniteDifferenceSpace
    @test Spaces.ncolumns(ᶜspace) == N
    @test Spaces.nlevels(ᶜspace) == 10
    @test Spaces.nlevels(ᶠspace) == 11
end

@testset "PointColumnEnsembleSpace – coordinate field" begin
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)

    ᶜz = ᶜcoords.z
    ᶠz = ᶠcoords.z

    # z values should be the same across all columns (uniform z mesh)
    col_ᶜz = Fields.coordinate_field(col_cspace).z
    col_ᶠz = Fields.coordinate_field(col_fspace).z

    # Each column's z values should match the reference column
    ᶜz_arr = parent(ᶜz)   # shape: (Nv, 1, 1, N) under VIFH
    col_ᶜz_arr = parent(col_ᶜz)

    @test size(ᶜz_arr, 1) == size(col_ᶜz_arr, 1)   # same number of levels
    for h in 1:N
        @test ᶜz_arr[:, 1, 1, h] ≈ col_ᶜz_arr[:, 1, 1, 1]
    end

    ᶠz_arr = parent(ᶠz)
    col_ᶠz_arr = parent(col_ᶠz)
    for h in 1:N
        @test ᶠz_arr[:, 1, 1, h] ≈ col_ᶠz_arr[:, 1, 1, 1]
    end
end

@testset "PointColumnEnsembleSpace – fill and ones" begin
    f_ones = Fields.ones(ᶜspace)
    @test all(==(FT(1)), parent(f_ones))

    f_fill = fill(FT(3), ᶜspace)
    @test all(==(FT(3)), parent(f_fill))
end

@testset "PointColumnEnsembleSpace – arithmetic" begin
    ᶜz = Fields.coordinate_field(ᶜspace).z

    f1 = Fields.ones(ᶜspace)
    f2 = ᶜz

    @test parent(f1 .+ f2) ≈ parent(f1) .+ parent(f2)
    @test parent(f1 .- f2) ≈ parent(f1) .- parent(f2)
    @test parent(FT(2) .* f2) ≈ FT(2) .* parent(f2)
    @test parent(sin.(f2)) ≈ sin.(parent(f2))
    @test parent(f2 .^ 2) ≈ parent(f2) .^ 2
end

@testset "PointColumnEnsembleSpace – local_geometry_field" begin
    lg = Fields.local_geometry_field(ᶜspace)
    @test eltype(lg) == Spaces.local_geometry_type(typeof(ᶜspace))

    # J and WJ include the vertical metric (element height = z_max-z_min / z_elem).
    # All values must be positive and finite.
    @test all(>(0), parent(lg.J))
    @test all(>(0), parent(lg.WJ))
    @test all(isfinite, parent(lg.J))

    # J must be identical across all columns (same vertical mesh).
    J_arr = parent(lg.J)
    for h in 2:N
        @test J_arr[:, :, :, h] ≈ J_arr[:, :, :, 1]
    end

    # J must match the single-column reference.
    col_lg = Fields.local_geometry_field(col_cspace)
    @test parent(lg.J)[:, 1, 1, 1] ≈ parent(col_lg.J)[:, 1, 1, 1]
end

@testset "PointColumnEnsembleSpace – face_space / center_space round-trip" begin
    @test Spaces.center_space(ᶠspace) === ᶜspace  ||
          Spaces.grid(Spaces.center_space(ᶠspace)) === Spaces.grid(ᶜspace)
    @test Spaces.face_space(ᶜspace) === ᶠspace ||
          Spaces.grid(Spaces.face_space(ᶜspace)) === Spaces.grid(ᶠspace)
end

@testset "PointColumnEnsembleSpace – Fields.level" begin
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z

    # Each level slice has N values (one per column)
    lev1 = Fields.level(ᶜz, 1)
    @test length(parent(lev1)) == N

    face_lev1 = Fields.level(ᶠz, PlusHalf(1))
    @test length(parent(face_lev1)) == N

    # Bounds checks (matches ColumnSpace behaviour — same guards as unit_field.jl)
    if Base.JLOptions().check_bounds == 1
        @test_throws BoundsError Fields.level(ᶜz, 0)
        @test_throws BoundsError Fields.level(ᶜz, 11)
        @test_throws BoundsError Fields.level(ᶠz, PlusHalf(-1))
    end
end

@testset "PointColumnEnsembleSpace – bycolumn" begin
    ᶜz = Fields.coordinate_field(ᶜspace).z
    result = similar(ᶜz)

    # Apply sin column-by-column; result must equal sin.(ᶜz) applied globally
    Fields.bycolumn(ᶜspace) do colidx
        result[colidx] .= sin.(ᶜz[colidx])
    end
    @test parent(result) ≈ sin.(parent(ᶜz))
end

@testset "PointColumnEnsembleSpace – finite difference operators" begin
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z

    # GradientC2F: gradient of z should be ≈ 1 everywhere (dz/dz = 1)
    grad = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(1))),
        top    = Operators.SetGradient(Geometry.Covariant3Vector(FT(1))),
    )
    ᶠgrad_z = @. grad(ᶜz)
    # Covariant3 component should be ~1 (dz/dξ₃ scaling)
    @test all(x -> abs(x) > 0, parent(ᶠgrad_z))

    # DivergenceF2C: divergence of a face field
    div = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top    = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    ᶜdiv = @. div(Geometry.Contravariant3Vector(ᶠz))
    # result should be finite
    @test all(isfinite, parent(ᶜdiv))

    # InterpolateC2F / InterpolateF2C round-trip
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top    = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()
    ᶠᶜz = @. interp_c2f(ᶜz)
    ᶜᶠᶜz = @. interp_f2c(ᶠᶜz)
    # round-trip interpolation should be close to original (not exact at boundaries)
    interior_levels = 2:(Spaces.nlevels(ᶜspace) - 1)
    for lev in interior_levels
        @test parent(Fields.level(ᶜᶠᶜz, lev)) ≈ parent(Fields.level(ᶜz, lev))
    end
end

@testset "PointColumnEnsembleSpace – similar and zero" begin
    ᶜz = Fields.coordinate_field(ᶜspace).z
    s = similar(ᶜz)
    @test axes(s) === ᶜspace
    fill!(s, FT(0))
    @test all(==(FT(0)), parent(s))
end

@testset "PointColumnEnsembleSpace – z_min / z_max" begin
    @test Spaces.z_min(ᶜspace) ≈ FT(0)
    @test Spaces.z_max(ᶜspace) ≈ FT(10_000)
end
