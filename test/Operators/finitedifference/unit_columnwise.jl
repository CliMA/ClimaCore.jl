#=
julia --project=.buildkite
using Revise; include("test/Operators/finitedifference/unit_columnwise.jl")
ENV["CLIMACOMMS_DEVICE"] = "CPU";
ENV["CLIMACOMMS_DEVICE"] = "CUDA";
=#
high_res = true;
@info "high_res: $high_res"

# using CUDA
import ClimaComms
using ClimaParams # needed in environment to load convenience parameter struct wrappers
import Thermodynamics as TD
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
import NullBroadcasts: NullBroadcasted
using LazyBroadcast: lazy
using LinearAlgebra: ×, dot, norm
import Thermodynamics as TD
import SciMLBase
import ClimaCore.Grids
import ClimaCore
import ClimaCore.Geometry
import ClimaCore.MatrixFields: @name, ⋅
import LinearAlgebra: adjoint
import LinearAlgebra as LA
import ClimaCore: Operators, Topologies, DataLayouts
import ClimaCore.MatrixFields
import ClimaCore.Spaces
import ClimaCore.Fields

# Operators.fd_shmem_is_supported(bc::Base.Broadcast.Broadcasted) = false
# ClimaCore.Operators.use_fd_shmem() = false
# The existing implementation limits our ability to apply
# the same expressions from within kernels
# Fields.error_mismatched_spaces(::Type, ::Type) = nothing # causes unsupported dynamic function invocation

const C1 = Geometry.Covariant1Vector
const C2 = Geometry.Covariant2Vector
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT12 = Geometry.Contravariant12Vector
const CT3 = Geometry.Contravariant3Vector
const CT123 = Geometry.Contravariant123Vector
const UVW = Geometry.UVWVector

const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.SetValue(CT3(0)),
)

const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

const ᶜdivᵥ = Operators.DivergenceF2C()
const ᶜinterp = Operators.InterpolateF2C()

ᶜtendencies(ρ, uₕ, ρe_tot) = (; ρ, uₕ, ρe_tot)
ᶠtendencies(u₃) = (; u₃)

Base.@kwdef struct RayleighSponge{FT}
    zd::FT
    α_uₕ::FT
    α_w::FT
end
Base.Broadcast.broadcastable(x::RayleighSponge) = tuple(x)

αₘ(s::RayleighSponge{FT}, z, α) where {FT} = ifelse(z > s.zd, α, FT(0))
ζ_rayleigh(s::RayleighSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_rayleigh_uₕ(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z, zmax)
β_rayleigh_w(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_w) * ζ_rayleigh(s, z, zmax)

function rayleigh_sponge_tendency_uₕ(ᶜuₕ, s)
    s isa Nothing && return NullBroadcasted()
    ᶜz = Fields.coordinate_field(axes(ᶜuₕ)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(axes(ᶜuₕ))).z
    # (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_uₕ(s, ᶜz, zmax) * ᶜuₕ)
end

function compute_kinetic(uₕ, uᵥ)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    FT = Spaces.undertype(axes(uₕ))
    # for some reason, on the gpu, this needs to be hoisted, and FT(1 / 2)
    # does not work inside the broadcast expression.
    onehalf = FT(1 / 2)
    return @. lazy(
        onehalf * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        ),
    )
end

function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠu³ * ᶠinterp(ᶜχ))))
end

function implicit_tendency_bc!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    (; rayleigh_sponge, dt) = p
    (; ᶜh_tot, ᶠu³, ᶜp) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    FT = Spaces.undertype(axes(Y.c))
    grav = FT(9.8)
    (; zmax) = p

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠu³)
    # Central advection of active tracers (e_tot and q_tot)
    Yₜ.c.ρe_tot .+= vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ(Φ(grav, ᶜz))

    @. Yₜ.f.u₃ -= β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * Y.f.u₃
    return nothing
end


function ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t)
    (; rayleigh_sponge, dt, zmax, thermo_params) = p
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶜJ = Fields.local_geometry_field(ᶜY).J
    ᶠz = Fields.coordinate_field(ᶠY).z
    FT = Spaces.undertype(axes(ᶜY))
    grav = FT(9.8)
    ᶜρ = ᶜY.ρ
    ᶜρe_tot = ᶜY.ρe_tot
    ᶜuₕ = ᶜY.uₕ
    ᶠu₃ = ᶠY.u₃

    ᶜK = compute_kinetic(ᶜuₕ, ᶠu₃)
    ᶜe_int = @. lazy(ᶜρe_tot / ᶜρ - ᶜK - Φ(grav, ᶜz))
    ᶜT = @. lazy(TD.air_temperature(thermo_params, ᶜe_int))
    ᶜp = @. lazy(TD.air_pressure(thermo_params, ᶜT, ᶜρ))
    ᶜh_tot =
        @. lazy(TD.total_enthalpy(thermo_params, ᶜρe_tot / ᶜρ, ᶜT))
    # Central advection of active tracers (e_tot and q_tot)
    ᶠuₕ³ = @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
    ᶠu³ = @. lazy(ᶠuₕ³ + CT3(ᶠu₃))
    tend_ρ_1 = @. lazy(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠuₕ³))
    tend_ρe_tot_1 = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    ᶜuₕ₀ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)

    return @. lazy(ᶜtendencies(-tend_ρ_1, ᶜuₕ₀, tend_ρe_tot_1))
end

function ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t)
    (; rayleigh_sponge, thermo_params, zmax) = p
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶠz = Fields.coordinate_field(ᶠY).z
    FT = Spaces.undertype(axes(ᶜY))
    grav = FT(9.8)
    ᶜρ = ᶜY.ρ
    ᶜρe_tot = ᶜY.ρe_tot
    ᶜuₕ = ᶜY.uₕ
    ᶠu₃ = ᶠY.u₃
    ᶜK = compute_kinetic(ᶜuₕ, ᶠu₃)
    ᶜe_int = @. lazy(ᶜρe_tot / ᶜρ - ᶜK - Φ(grav, ᶜz))
    ᶜT = @. lazy(TD.air_temperature(thermo_params, ᶜe_int))
    ᶜp = @. lazy(TD.air_pressure(thermo_params, ᶜT, ᶜρ))
    bc1 = @. lazy(-(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(Φ(grav, ᶜz))))
    bc2 = @. lazy(-β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * ᶠu₃)
    return @. lazy(ᶠtendencies(bc1 + bc2))
end

function compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    return @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
end

Φ(grav, z) = grav * z

function set_precomputed_quantities!(Y, p, t)
    (; thermo_params) = p
    (; ᶜu, ᶠu³, ᶠu, ᶜK, ᶜT, ᶜp) = p.precomputed

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶜz = Fields.coordinate_field(Y.c).z
    FT = Spaces.undertype(axes(Y.c))
    grav = FT(9.8)
    ᶠu₃ = Y.f.u₃
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    ᶠu³ .= compute_ᶠuₕ³(ᶜuₕ, ᶜρ) .+ CT3.(ᶠu₃)
    ᶜK .= compute_kinetic(ᶜuₕ, ᶠu₃)

    ᶜe_int = @. Y.c.ρe_tot / Y.c.ρ - ᶜK - Φ(grav, ᶜz)
    @. ᶜT = TD.air_temperature(thermo_params, ᶜe_int)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜT, Y.c.ρ)

    (; ᶜh_tot) = p.precomputed
    @. ᶜh_tot =
        TD.total_enthalpy(thermo_params, Y.c.ρe_tot / Y.c.ρ, ᶜT)
    return nothing
end

FT = Float32;
if high_res
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 63,
        z_min = 0,
        z_max = 30000.0,
        radius = 6.371e6,
        h_elem = 30,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
else
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 8,
        z_min = 0,
        z_max = 30000.0,
        radius = 6.371e6,
        h_elem = 2,
        n_quad_points = 2,
        staggering = CellCenter(),
    )
end
# ᶜspace = SliceXZSpace(FT;
#     z_elem = 10,
#     x_min = 0,
#     x_max = 1,
#     z_min = 0,
#     z_max = 30000.0,
#     periodic_x = false,
#     n_quad_points = 4,
#     x_elem = 4,
#     staggering = CellCenter()
# )
# ᶜspace = Box3DSpace(FT;
#     z_elem = 10,
#     x_min = 0,
#     x_max = 1,
#     y_min = 0,
#     y_max = 1,
#     z_min = 0,
#     z_max = 30000.0,
#     periodic_x = false,
#     periodic_y = false,
#     n_quad_points = 4,
#     x_elem = 3,
#     y_elem = 4,
#     staggering = CellCenter()
# )
# ᶜspace = ColumnSpace(
#     FT;
#     z_elem = 10,
#     z_min = 0,
#     z_max = 30000.0,
#     staggering = CellCenter()
# )

ᶠspace = Spaces.face_space(ᶜspace);
cnt = (; ρ = zero(FT), uₕ = zero(C12{FT}), ρe_tot = zero(FT));
Yc = Fields.fill(cnt, ᶜspace);
fill!(parent(Yc.ρ), 1)
fill!(parent(Yc.uₕ), 0.01)
fill!(parent(Yc.ρe_tot), 1000.0)
Yf = Fields.fill((; u₃ = zero(C3{FT})), ᶠspace);
Y = Fields.FieldVector(; c = Yc, f = Yf);

thermo_params = TD.Parameters.ThermodynamicsParameters(FT);

ᶠcoord = Fields.coordinate_field(ᶠspace);
ᶜcoord = Fields.coordinate_field(ᶜspace);
precomputed = (;
    ᶜh_tot = Fields.Field(FT, ᶜspace),
    ᶠu³ = Fields.Field(CT3{FT}, ᶠspace),
    ᶜp = Fields.Field(FT, ᶜspace),
    ᶜK = Fields.Field(FT, ᶜspace),
    ᶜT = Fields.Field(FT, ᶜspace),
    ᶠu = Fields.Field(C123{FT}, ᶠspace),
    ᶜu = Fields.Field(C123{FT}, ᶜspace),
)
dt = FT(0.1)

p = (;
    zmax = Spaces.z_max(axes(Y.f)),
    rayleigh_sponge = RayleighSponge{FT}(15000.0, 0.0, 1.0),
    thermo_params,
    dt,
    precomputed,
)
Yc = Y.c;
Yf = Y.f;
zc = Fields.coordinate_field(Yc).z;
zf = Fields.coordinate_field(Yf).z;
Yₜ = similar(Y);
Yₜ_bc = similar(Yₜ);

@. Yₜ_bc = 0
@. Yₜ = 0
fill!(parent(Yc.ρ), 1);
@. Yc.ρ += 0.1 * sin(zc);
parent(Yf.u₃) .+= 0.001 .* sin.(parent(zf));
fill!(parent(Yc.uₕ), 0.01);
fill!(parent(Yc.ρe_tot), 100000.0);

t₀ = zero(Spaces.undertype(axes(Yc)))

using Test

dev = ClimaComms.device(axes(Yc))
Operators.columnwise!(
    dev,
    ᶜimplicit_tendency_bc,
    ᶠimplicit_tendency_bc,
    Yₜ.c,
    Yₜ.f,
    Yc,
    Yf,
    p,
    t₀,
)
implicit_tendency_bc!(Yₜ_bc, Y, p, t₀)
abs_err_c = maximum(Array(abs.(parent(Yₜ.c) .- parent(Yₜ_bc.c))))
abs_err_f = maximum(Array(abs.(parent(Yₜ.f) .- parent(Yₜ_bc.f))))
results_match = abs_err_c < 6e-9 && abs_err_c < 6e-9
if !results_match
    @show norm(Array(parent(Yₜ_bc.c))), norm(Array(parent(Yₜ.c)))
    @show norm(Array(parent(Yₜ_bc.f))), norm(Array(parent(Yₜ.f)))
    @show abs_err_c
    @show abs_err_f
end
@test results_match
#! format: off
@static if ClimaComms.device() isa ClimaComms.CUDADevice
    println(
        CUDA.@profile begin
            Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
            Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
            Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
            Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
        end
    )
    println(CUDA.@profile begin
        @. Yₜ += 1
        @. Yₜ += 1
        @. Yₜ += 1
        @. Yₜ += 1
    end)
else
    @info "CPU timings"
    @time "columnwise!" Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
    @time "columnwise!" Operators.columnwise!(dev,ᶜimplicit_tendency_bc,ᶠimplicit_tendency_bc,Yₜ.c,Yₜ.f,Yc,Yf,p,t₀)
    @time "implicit_tendency_bc!" implicit_tendency_bc!(Yₜ_bc, Y, p, t₀)
    @time "implicit_tendency_bc!" implicit_tendency_bc!(Yₜ_bc, Y, p, t₀)
    @info "Done!"
end
#! format: off

#=
Analysis:

julia> DataLayouts.ncomponents(Fields.field_values(Fields.local_geometry_field(Yₜ.c)))
42
julia> DataLayouts.ncomponents(Fields.field_values(Yₜ.c))
4
julia> DataLayouts.ncomponents(Fields.field_values(Yₜ.f))
1

So,
all LG: 2*42 (faces + centers)
only needed LG: 12 (found by trial-and-error)

state: 1+4 (faces + centers)
nreads: 12 + 5 * n_points
nwrites: 5 * n_points

Conclusion:
 - We first need to (generally) minimize the number of LocalGeometry variables
   that are read into shared memory. This is possible because we can
   ahead-of-time scan the broadcasted object and, at compile-time, return a list
   of components to read into shmem, and unroll the loop over those variables

   Doing this manually yielded:

```
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.coordinates.z[ᶜui] = ᶜlg.coordinates.z[ᶜui]) # needed
is_valid_index_cw(ᶠus, ᶠui) && (ᶠlg_col.coordinates.z[ᶠui] = ᶠlg.coordinates.z[ᶠui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.J[ᶜui] = ᶜlg.J[ᶜui]) # needed
is_valid_index_cw(ᶠus, ᶠui) && (ᶠlg_col.J[ᶠui] = ᶠlg.J[ᶠui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.invJ[ᶜui] = ᶜlg.invJ[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:1[ᶜui] = ᶜlg.gⁱʲ.components.data.:1[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:2[ᶜui] = ᶜlg.gⁱʲ.components.data.:2[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:3[ᶜui] = ᶜlg.gⁱʲ.components.data.:3[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:4[ᶜui] = ᶜlg.gⁱʲ.components.data.:4[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:5[ᶜui] = ᶜlg.gⁱʲ.components.data.:5[ᶜui]) # needed
is_valid_index_cw(ᶜus, ᶜui) && (ᶜlg_col.gⁱʲ.components.data.:6[ᶜui] = ᶜlg.gⁱʲ.components.data.:6[ᶜui]) # needed
is_valid_index_cw(ᶠus, ᶠui) && (ᶠlg_col.gⁱʲ.components.data.:9[ᶠui] = ᶠlg.gⁱʲ.components.data.:9[ᶠui]) # needed
```

=#

nothing
