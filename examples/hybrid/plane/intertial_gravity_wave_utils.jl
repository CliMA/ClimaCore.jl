module InertialGravityWaveUtils

import ClimaCore.Geometry as Geometry

# min_λx = 2 * (x_max / x_elem) / upsampling_factor # this should include npoly
# min_λz = 2 * (FT( / z_)elem) / upsampling_factor
# min_λx = 2 * π / max_kx = x_max / max_ikx
# min_λz = 2 * π / max_kz = 2 * z_max / max_ikz
# max_ikx = x_max / min_λx = upsampling_factor * x_elem / 2
# max_ikz = 2 * z_max / min_λz = upsampling_factor * z_elem
function ρfb_init_coefs!(::Type{FT}, params) where {FT}
    (; max_ikz, max_ikx, x_max, z_max, unit_integral) = params
    (; ρfb_init_array, ᶜρb_init_xz) = params
    # Since the coefficients are for a modified domain of height 2 * z_max, the
    # unit integral over the domain must be multiplied by 2 to ensure correct
    # normalization. On the other hand, ᶜρb_init is assumed to be 0 outside of
    # the "true" domain, so the integral of
    # ᶜintegrand (`ᶜintegrand = ᶜρb_init / ᶜfourier_factor`) should not be modified.
    # where `ᶜfourier_factor = exp(im * (kx * x + kz * z))`.
    @inbounds begin
        Threads.@threads for ikx in (-max_ikx):max_ikx
            for ikz in (-max_ikz):max_ikz
                kx::FT = 2 * π / x_max * ikx
                kz::FT = 2 * π / (2 * z_max) * ikz
                ρfb_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1] =
                    sum(ᶜρb_init_xz) do nt
                        (; ρ, x, z) = nt
                        ρ / exp(im * (kx * x + kz * z))
                    end / unit_integral

            end
        end
    end
    return nothing
end

function Bretherton_transforms!(lin_cache, t, ::Type{FT}) where {FT}
    # Bretherton_transforms_partial_sums! is fastest because
    # we can multithread across
    #       `Iterators.product((-max_ikx):max_ikx, (-max_ikz):max_ikz)`
    # and apply sums for center and face fields. Using mapreduce requires
    # two calls and, as a result in ~20 slower.

    Bretherton_transforms_original!(lin_cache, t, FT)
    # Bretherton_transforms_partial_sums!(lin_cache, t, FT)
    # Bretherton_transforms_threaded_mapreduce!(lin_cache, t, FT)
end

function Bretherton_transforms_original!(lin_cache, t, ::Type{FT}) where {FT}
    (; ᶜx, ᶠx, ᶜz, ᶠz) = lin_cache
    (; x_max, z_max, u₀, δ, cₛ², grav, f, ρₛ) = lin_cache
    (; ρfb_init_array, ᶜfourier_factor, ᶠfourier_factor) = lin_cache
    (; ᶜpb, ᶜρb, ᶜub, ᶜvb, ᶠwb) = lin_cache

    ᶜpb .= FT(0)
    ᶜρb .= FT(0)
    ᶜub .= FT(0)
    ᶜvb .= FT(0)
    ᶠwb .= FT(0)
    max_ikx, max_ikz = (size(ρfb_init_array) .- 1) .÷ 2
    @inbounds for ikx in (-max_ikx):max_ikx, ikz in (-max_ikz):max_ikz
        kx = 2 * π / x_max * ikx
        kz = 2 * π / (2 * z_max) * ikz

        # Fourier coefficient of ᶜρb_init (for current kx and kz)
        ρfb_init = ρfb_init_array[ikx + max_ikx + 1, ikz + max_ikz + 1]

        # Fourier factors, shifted by u₀ * t along the x-axis
        @. ᶜfourier_factor = exp(im * (kx * (ᶜx - u₀ * t) + kz * ᶜz))
        @. ᶠfourier_factor = exp(im * (kx * (ᶠx - u₀ * t) + kz * ᶠz))

        # roots of a₁(s)
        p₁ = cₛ² * (kx^2 + kz^2 + δ^2 / 4) + f^2
        q₁ = grav * kx^2 * (cₛ² * δ - grav) + cₛ² * f^2 * (kz^2 + δ^2 / 4)
        α² = p₁ / 2 - sqrt(p₁^2 / 4 - q₁)
        β² = p₁ / 2 + sqrt(p₁^2 / 4 - q₁)
        α = sqrt(α²)
        β = sqrt(β²)

        # inverse Laplace transform of s^p/((s^2 + α^2)(s^2 + β^2)) for p ∈ -1:3
        if α == 0
            L₋₁ = (β² * t^2 / 2 - 1 + cos(β * t)) / β^4
            L₀ = (β * t - sin(β * t)) / β^3
        else
            L₋₁ =
                (-cos(α * t) / α² + cos(β * t) / β²) / (β² - α²) + 1 / (α² * β²)
            L₀ = (sin(α * t) / α - sin(β * t) / β) / (β² - α²)
        end
        L₁ = (cos(α * t) - cos(β * t)) / (β² - α²)
        L₂ = (-sin(α * t) * α + sin(β * t) * β) / (β² - α²)
        L₃ = (-cos(α * t) * α² + cos(β * t) * β²) / (β² - α²)

        # Fourier coefficients of Bretherton transforms of final perturbations
        C₁ = grav * (grav - cₛ² * (im * kz + δ / 2))
        C₂ = grav * (im * kz - δ / 2)
        pfb = -ρfb_init * (L₁ + L₋₁ * f^2) * C₁
        ρfb =
            ρfb_init *
            (L₃ + L₁ * (p₁ + C₂) + L₋₁ * f^2 * (cₛ² * (kz^2 + δ^2 / 4) + C₂))
        ufb = ρfb_init * L₀ * im * kx * C₁ / ρₛ
        vfb = -ρfb_init * L₋₁ * im * kx * f * C₁ / ρₛ
        wfb = -ρfb_init * (L₂ + L₀ * (f^2 + cₛ² * kx^2)) * grav / ρₛ

        # Bretherton transforms of final perturbations
        @. ᶜpb += real(pfb * ᶜfourier_factor)
        @. ᶜρb += real(ρfb * ᶜfourier_factor)
        @. ᶜub += real(ufb * ᶜfourier_factor)
        @. ᶜvb += real(vfb * ᶜfourier_factor)
        @. ᶠwb += real(wfb * ᶠfourier_factor)
        # The imaginary components should be 0 (or at least very close to 0).
    end
    return nothing
end

function linear_solution!(Y, lin_cache, t, ::Type{FT}) where {FT}
    (; ᶜz, ᶜp₀, ᶜρ₀, ᶜu₀, ᶜv₀, ᶠw₀) = lin_cache
    (; ᶜinterp) = lin_cache
    (; R_d, ᶜ𝔼_name, x_max, z_max, p_0, cp_d, cv_d, grav, T_tri) = lin_cache
    (; ᶜbretherton_factor_pρ) = lin_cache
    (; ᶜbretherton_factor_uvwT, ᶠbretherton_factor_uvwT) = lin_cache
    (; ᶜpb, ᶜρb, ᶜub, ᶜvb, ᶠwb, ᶜp, ᶜρ, ᶜu, ᶜv, ᶠw, ᶜT) = lin_cache

    Bretherton_transforms!(lin_cache, t, FT)

    # final state
    @. ᶜp = ᶜp₀ + ᶜpb * ᶜbretherton_factor_pρ
    @. ᶜρ = ᶜρ₀ + ᶜρb * ᶜbretherton_factor_pρ
    @. ᶜu = ᶜu₀ + ᶜub * ᶜbretherton_factor_uvwT
    @. ᶜv = ᶜv₀ + ᶜvb * ᶜbretherton_factor_uvwT
    @. ᶠw = ᶠw₀ + ᶠwb * ᶠbretherton_factor_uvwT
    @. ᶜT = ᶜp / (R_d * ᶜρ)

    @. Y.c.ρ = ᶜρ
    if ᶜ𝔼_name == :ρθ
        @. Y.c.ρθ = ᶜρ * ᶜT * (p_0 / ᶜp)^(R_d / cp_d)
    elseif ᶜ𝔼_name == :ρe
        @. Y.c.ρe =
            ᶜρ * (
                cv_d * (ᶜT - T_tri) +
                (ᶜu^2 + ᶜv^2 + ᶜinterp(ᶠw)^2) / 2 +
                grav * ᶜz
            )
    elseif ᶜ𝔼_name == :ρe_int
        @. Y.c.ρe_int = ᶜρ * cv_d * (ᶜT - T_tri)
    end
    # NOTE: The following two lines are a temporary workaround b/c Covariant12Vector won't accept a non-zero second component in an XZ-space.
    # So we temporarily set it to zero and then reassign its intended non-zero value (since in case of large-scale config ᶜv is non-zero)
    @. Y.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(ᶜu, FT(0.0)))
    @. Y.c.uₕ.components.data.:2 .= ᶜv
    @. Y.f.w = Geometry.Covariant3Vector(Geometry.WVector(ᶠw))
    return nothing
end

end # module
