using ClimaCore
using ClimaCore.CommonSpaces, ClimaCore.Geometry
using ClimaCore: Fields, Spaces, Operators, Grids, Domains, Meshes
using ClimaCore.Utilities: PlusHalf
using Test

const FT = Float64

import ClimaComms
ClimaComms.@import_required_backends

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

# CenterFiniteDifferenceSpace reference for single-column behaviour
_col_domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(FT(0)),
    Geometry.ZPoint{FT}(FT(10_000)),
    boundary_names = (:bottom, :top),
)
_col_mesh = Meshes.IntervalMesh(_col_domain; nelems = 10)
col_cspace = Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), _col_mesh)
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

@testset "PointColumnEnsembleSpace – lat/lon coordinate field" begin
    ᶜcoords = Fields.coordinate_field(ᶜspace)

    ᶜlat  = ᶜcoords.lat
    ᶜlong = ᶜcoords.long

    lat_arr  = parent(ᶜlat)   # shape: (Nv, 1, 1, N)
    long_arr = parent(ᶜlong)

    # Each column's lat/lon should be constant across all vertical levels
    # and match the corresponding input point.
    for h in 1:N
        expected_lat  = FT(points[h].lat)
        expected_long = FT(points[h].long)
        @test all(≈(expected_lat),  lat_arr[:, 1, 1, h])
        @test all(≈(expected_long), long_arr[:, 1, 1, h])
    end

    # Columns must differ from each other (sanity check that lat/lon aren't broadcast-collapsed)
    for h in 2:N
        @test !(lat_arr[:, 1, 1, h] ≈ lat_arr[:, 1, 1, 1]) ||
              !(long_arr[:, 1, 1, h] ≈ long_arr[:, 1, 1, 1])
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
        @test Array(parent(Fields.level(ᶜᶠᶜz, lev))) ≈ Array(parent(Fields.level(ᶜz, lev)))
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

# ---------------------------------------------------------------
# Single-point PointColumnEnsembleSpace vs CenterFiniteDifferenceSpace
#
# A PointColumnEnsembleSpace with one column uses a 3×3 local geometry
# (LocalGeometry{(1,2,3)}) while CenterFiniteDifferenceSpace uses a 1×1
# one (LocalGeometry{(3,)}).  Because the horizontal block of ∂x∂ξ is
# the identity and J_h = 1, all vertical-physics quantities must be
# numerically identical.  The tests below verify this for:
#   1. FD operators  (GradientC2F, DivergenceF2C, Interpolate round-trip)
#   2. Vector-projection paths that call Jcontravariant3
#      – Covariant3Vector branch: reads gⁱʲ[3,3] vs gⁱʲ[1,1]
#      – WVector branch         : reads ∂ξ∂x[3,3] vs ∂ξ∂x[1,1]
#   3. Metric tensor scalars: J, WJ, invJ; and the (3,3)/(1,1) diagonal
#      entries of ∂x∂ξ and gⁱʲ extracted via Δz_metric_component.
# ---------------------------------------------------------------
@testset "single-point PointColumnEnsembleSpace ≡ CenterFiniteDifferenceSpace" begin
    # Build a single-point ensemble on the same z mesh as col_cspace
    ᶜsp1 = PointColumnEnsembleSpace(FT;
        points     = [Geometry.LatLongPoint(FT(0), FT(0))],
        z_elem     = 10,
        z_min      = FT(0),
        z_max      = FT(10_000),
        radius     = FT(6.371229e6),
        staggering = CellCenter(),
    )
    ᶠsp1 = Spaces.face_space(ᶜsp1)

    # Flatten field data to a plain Vector{FT} for shape-agnostic comparison
    fvec(f) = vec(Array(parent(f)))

    ᶜz1   = Fields.coordinate_field(ᶜsp1).z
    ᶠz1   = Fields.coordinate_field(ᶠsp1).z
    ᶜzcol = Fields.coordinate_field(col_cspace).z
    ᶠzcol = Fields.coordinate_field(col_fspace).z

    # ── 1. FD operators ──────────────────────────────────────────────────────
    grad = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(1))),
        top    = Operators.SetGradient(Geometry.Covariant3Vector(FT(1))),
    )
    @test fvec(@. grad(ᶜz1)) ≈ fvec(@. grad(ᶜzcol))

    div_ct3 = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top    = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    @test fvec(@. div_ct3(Geometry.Contravariant3Vector(ᶠz1))) ≈
          fvec(@. div_ct3(Geometry.Contravariant3Vector(ᶠzcol)))

    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(), top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()
    @test fvec(@. interp_c2f(ᶜz1)) ≈ fvec(@. interp_c2f(ᶜzcol))
    @test fvec(@. interp_f2c(ᶠz1)) ≈ fvec(@. interp_f2c(ᶠzcol))

    # ── 2. Vector projections via Jcontravariant3 ────────────────────────────
    # Covariant3Vector path: specialised to J * gⁱʲ[3,3] * u[1] (ensemble)
    #                                    vs J * gⁱʲ[1,1] * u[1] (column)
    div_cov3 = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
        top    = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
    )
    @test fvec(@. div_cov3(Geometry.Covariant3Vector(ᶠz1))) ≈
          fvec(@. div_cov3(Geometry.Covariant3Vector(ᶠzcol)))

    # WVector path: specialised to J * ∂ξ∂x[3,3] * u[1] (ensemble)
    #                           vs J * ∂ξ∂x[1,1] * u[1] (column)
    div_wvec = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top    = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    @test fvec(@. div_wvec(Geometry.WVector(ᶠz1))) ≈
          fvec(@. div_wvec(Geometry.WVector(ᶠzcol)))

    # ── 3. Metric-tensor scalars and tensor diagonal ──────────────────────────
    # J, WJ, invJ are scalar fields on both spaces; they must be identical.
    lg_1   = Fields.local_geometry_field(ᶜsp1)
    lg_col = Fields.local_geometry_field(col_cspace)
    @test fvec(lg_1.J)    ≈ fvec(lg_col.J)
    @test fvec(lg_1.WJ)   ≈ fvec(lg_col.WJ)
    @test fvec(lg_1.invJ) ≈ fvec(lg_col.invJ)

    # Δz_data extracts ∂x∂ξ[3,3] for the ensemble (LatLongZPoint, idx=9)
    # and ∂x∂ξ[1,1] for the column (ZPoint, idx=1); both equal Δz.
    Δz_1   = vec(Array(parent(Spaces.Δz_data(ᶜsp1))))
    Δz_col = vec(Array(parent(Spaces.Δz_data(col_cspace))))
    @test Δz_1 ≈ Δz_col

    # Same index-selection trick for gⁱʲ: vertical diagonal entry = (1/Δz)².
    # For ensemble (3×3 gⁱʲ): component 9 = (3,3); for column (1×1): component 1.
    _gij_zz(sp) = let lg = Spaces.local_geometry_data(sp)
        getproperty(
            lg.gⁱʲ.components.data,
            Geometry.Δz_metric_component(eltype(lg.coordinates)),
        )
    end
    @test vec(Array(parent(_gij_zz(ᶜsp1)))) ≈ vec(Array(parent(_gij_zz(col_cspace))))

    # ── 4. Space properties ───────────────────────────────────────────────────
    @test Spaces.nlevels(ᶜsp1)   == Spaces.nlevels(col_cspace)
    @test Spaces.ncolumns(ᶜsp1)  == Spaces.ncolumns(col_cspace)
    @test Spaces.left_boundary_name(ᶜsp1)  == Spaces.left_boundary_name(col_cspace)
    @test Spaces.right_boundary_name(ᶜsp1) == Spaces.right_boundary_name(col_cspace)

    # ── 5. Face-space metrics ─────────────────────────────────────────────────
    lg1f   = Fields.local_geometry_field(ᶠsp1)
    lgcolf = Fields.local_geometry_field(col_fspace)
    @test fvec(lg1f.J)    ≈ fvec(lgcolf.J)
    @test fvec(lg1f.WJ)   ≈ fvec(lgcolf.WJ)
    @test fvec(lg1f.invJ) ≈ fvec(lgcolf.invJ)

    # ── 6. GradientF2C ────────────────────────────────────────────────────────
    grad_f2c = Operators.GradientF2C(
        bottom = Operators.SetValue(FT(0)),
        top    = Operators.SetValue(FT(10_000)),
    )
    @test fvec(@. grad_f2c(ᶠz1)) ≈ fvec(@. grad_f2c(ᶠzcol))

    # ── 7. Biased interpolation (C2F) ─────────────────────────────────────────
    lb_c2f = Operators.LeftBiasedC2F(bottom = Operators.SetValue(FT(0)))
    rb_c2f = Operators.RightBiasedC2F(top   = Operators.SetValue(FT(10_000)))
    @test fvec(@. lb_c2f(ᶜz1)) ≈ fvec(@. lb_c2f(ᶜzcol))
    @test fvec(@. rb_c2f(ᶜz1)) ≈ fvec(@. rb_c2f(ᶜzcol))

    # Biased interpolation (F2C)
    lb_f2c = Operators.LeftBiasedF2C(bottom = Operators.SetValue(FT(0)))
    rb_f2c = Operators.RightBiasedF2C(top   = Operators.SetValue(FT(10_000)))
    @test fvec(@. lb_f2c(ᶠz1)) ≈ fvec(@. lb_f2c(ᶠzcol))
    @test fvec(@. rb_f2c(ᶠz1)) ≈ fvec(@. rb_f2c(ᶠzcol))

    # ── 8. Weighted interpolation ─────────────────────────────────────────────
    # Use ones as weight so the result is a regular average (non-zero everywhere).
    ᶠones1   = Fields.ones(ᶠsp1)
    ᶠonescol = Fields.ones(col_fspace)
    ᶜones1   = Fields.ones(ᶜsp1)
    ᶜonescol = Fields.ones(col_cspace)

    wt_f2c = Operators.WeightedInterpolateF2C()
    wt_c2f = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top    = Operators.Extrapolate(),
    )
    @test fvec(@. wt_f2c(ᶠones1, ᶠz1))     ≈ fvec(@. wt_f2c(ᶠonescol, ᶠzcol))
    @test fvec(@. wt_c2f(ᶜones1, ᶜz1))     ≈ fvec(@. wt_c2f(ᶜonescol, ᶜzcol))

    # ── 9. AdvectionC2C ───────────────────────────────────────────────────────
    # Uniform unit Contravariant3 velocity; θ = z.
    # A(v, θ)[i] = ½ { (θ[i+1]-θ[i]) v³[i+½] + (θ[i]-θ[i-1]) v³[i-½] }
    adv = Operators.AdvectionC2C(
        bottom = Operators.SetValue(FT(0)),
        top    = Operators.SetValue(FT(0)),
    )
    ᶠv1   = fill(Geometry.Contravariant3Vector(FT(1)), ᶠsp1)
    ᶠvcol = fill(Geometry.Contravariant3Vector(FT(1)), col_fspace)
    adv1   = similar(ᶜz1);   @. adv1   = adv(ᶠv1,   ᶜz1)
    advcol = similar(ᶜzcol); @. advcol = adv(ᶠvcol, ᶜzcol)
    @test fvec(adv1) ≈ fvec(advcol)

    # ── 10. Fields.sum (weighted integral ∫ f WJ dξ) ─────────────────────────
    # WJ is identical for both spaces (= Δz for a uniform mesh), so the
    # integral of any function of z must agree.
    @test sum(ᶜz1)            ≈ sum(ᶜzcol)
    @test sum(x -> x^2, ᶜz1) ≈ sum(x -> x^2, ᶜzcol)

    # ── 11. Fields.Δz_field ───────────────────────────────────────────────────
    @test fvec(Fields.Δz_field(ᶜsp1)) ≈ fvec(Fields.Δz_field(col_cspace))
end
