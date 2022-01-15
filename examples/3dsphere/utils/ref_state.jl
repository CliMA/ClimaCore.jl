import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators

const FT = Float64
const grav = 9.8 # gravitational constant
const MSLP = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)

const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height

function decaying_temperature_profile(
	z;
	T_virt_surf = FT(280.0),
	T_min_ref = FT(230.0),
)
	# Scale height for surface temperature
	H_sfc = R_d * T_virt_surf / grav
	H_t = H_sfc

	z′ = z / H_t
	tanh_z′ = tanh(z′)

	ΔTv = T_virt_surf - T_min_ref
	Tv = T_virt_surf - ΔTv * tanh_z′

	ΔTv′ = ΔTv / T_virt_surf
	p =
		MSLP * exp(
			(
				-H_t *
				(z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
			) / (H_sfc * (1 - ΔTv′^2)),
		)
	ρ = p / (R_d * Tv)
	return (ρ = ρ, p = p, Tv = Tv)
end

Φ(z) = grav * z

function isothermal_profile(z)

    p = MSLP * exp(-z / H)
    ρ = 1 / R_d / T_0 * p

    e = cv_d * (T_0 - T_tri) + Φ(z)

    return (ρ = ρ, p = p, Tv = T_0)
end

function discrete_hydrostatic_balance!(ρ, Tv, zc, grav)
    for i in 1:(length(ρ) - 1)
        ρ[i + 1] = ρ[i] * (R_d*Tv[i] - grav*(zc[i+1]-zc[i])/2) / (R_d*Tv[i+1]+grav*(zc[i+1]-zc[i])/2)
    end
end

function calc_ref_state(c_coords, profile::Function)
	zc_vec = parent(c_coords.z) |> unique

	N = length(zc_vec)
	ρ = zeros(Float64, N)
	Tv = zeros(Float64, N)

	for i in 1:N
		var = profile(zc_vec[i])
		ρ[i] = var.ρ
		Tv[i] = var.Tv
	end
	discrete_hydrostatic_balance!(ρ, Tv, zc_vec, grav)

	ρe = @. ρ * cv_d * (Tv - T_tri) + ρ * Φ(zc_vec)
	p = @. ρ * R_d * Tv

	ref_ρ = map(_ -> 0.0, c_coords)
	ref_ρe = map(_ -> 0.0, c_coords)
	ref_p = map(_ -> 0.0, c_coords)

	parent(ref_ρ) .= ρ
	parent(ref_ρe) .= ρe
	parent(ref_p) .= p

	return (ρ = ref_ρ, ρe = ref_ρe, p = ref_p)
end