#### Gradients
function op_GradientF2C!(c, f, bcs = (;))
    ∇f = Operators.GradientF2C(bcs)
    @. c.∇x = ∇f(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_GradientF2C!)}) = 2 # 1 write + 1 read (0 metric terms)
function op_GradientC2F!(c, f, bcs)
    ∇f = Operators.GradientC2F(bcs)
    @. f.∇x = ∇f(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_GradientC2F!)}) = 2 # 1 write + 1 read (0 metric terms)
#### Divergences
function op_DivergenceF2C!(c, f, bcs = (;))
    div = Operators.DivergenceF2C(bcs)
    @. c.x = div(Geometry.WVector(f.y))
    return nothing
end
n_reads_writes(::Type{typeof(op_DivergenceF2C!)}) = 3 # 1 write + 2 reads (1 metric term)
function op_DivergenceC2F!(c, f, bcs)
    div = Operators.DivergenceC2F(bcs)
    @. f.x = div(Geometry.WVector(c.y))
    return nothing
end
n_reads_writes(::Type{typeof(op_DivergenceC2F!)}) = 3 # 1 write + 2 reads (1 metric term)
#### Interpolations
function op_InterpolateF2C!(c, f, bcs = (;))
    interp = Operators.InterpolateF2C(bcs)
    @. c.x = interp(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_InterpolateF2C!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_InterpolateC2F!(c, f, bcs)
    interp = Operators.InterpolateC2F(bcs)
    @. f.x = interp(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_InterpolateC2F!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_BottomBiasedC2F!(c, f, bcs)
    interp = Operators.BottomBiasedC2F(bcs)
    @. f.x = interp(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_BottomBiasedC2F!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_BottomBiasedF2C!(c, f, bcs = (;))
    interp = Operators.BottomBiasedF2C(bcs)
    @. c.x = interp(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_BottomBiasedF2C!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_TopBiasedC2F!(c, f, bcs)
    interp = Operators.TopBiasedC2F(bcs)
    @. f.x = interp(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_TopBiasedC2F!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_TopBiasedF2C!(c, f, bcs = (;))
    interp = Operators.TopBiasedF2C(bcs)
    @. c.x = interp(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_TopBiasedF2C!)}) = 2 # 1 write + 1 reads (0 metric terms)
#### Curl
function op_CurlC2F!(c, f, bcs = (;))
    curl = Operators.CurlC2F(bcs)
    @. f.curluₕ = curl(c.uₕ)
    return nothing
end
n_reads_writes(::Type{typeof(op_CurlC2F!)}) = -1 # todo
#### Mixed/adaptive
function op_UpwindBiasedProductC2F!(c, f, bcs = (;))
    upwind = Operators.UpwindBiasedProductC2F(bcs)
    @. f.contra3 = upwind(f.w, c.x)
    return nothing
end
n_reads_writes(::Type{typeof(op_UpwindBiasedProductC2F!)}) = -1 # todo
function op_Upwind3rdOrderBiasedProductC2F!(c, f, bcs = (;))
    upwind = Operators.Upwind3rdOrderBiasedProductC2F(bcs)
    @. f.contra3 = upwind(f.w, c.x)
    return nothing
end
n_reads_writes(::Type{typeof(op_Upwind3rdOrderBiasedProductC2F!)}) = -1 # todo
#### Simple composed (non-exhaustive due to combinatorial explosion)
function op_divgrad_CC!(c, f, bcs)
    grad = Operators.GradientC2F(bcs.inner)
    div = Operators.DivergenceF2C(bcs.outer)
    @. c.y = div(grad(c.x))
    return nothing
end
n_reads_writes(::Type{typeof(op_divgrad_CC!)}) = 3 # 1 write, 2 reads (1 metric term)
function op_divgrad_FF!(c, f, bcs)
    grad = Operators.GradientF2C(bcs.inner)
    div = Operators.DivergenceC2F(bcs.outer)
    @. f.y = div(grad(f.x))
    return nothing
end
n_reads_writes(::Type{typeof(op_divgrad_FF!)}) = 3 # 1 write, 2 reads (1 metric term)
function op_div_interp_CC!(c, f, bcs)
    interp = Operators.InterpolateC2F(bcs.inner)
    div = Operators.DivergenceF2C(bcs.outer)
    @. c.y = div(interp(c.contra3))
    return nothing
end
n_reads_writes(::Type{typeof(op_div_interp_CC!)}) = -1 # todo
function op_div_interp_FF!(c, f, bcs)
    interp = Operators.InterpolateF2C(bcs.inner)
    div = Operators.DivergenceC2F(bcs.outer)
    @. f.y = div(interp(f.contra3))
    return nothing
end
n_reads_writes(::Type{typeof(op_div_interp_FF!)}) = -1 # todo
function op_divgrad_uₕ!(c, f, bcs)
    grad = Operators.GradientC2F(bcs.inner)
    div = Operators.DivergenceF2C(bcs.outer)
    @. c.uₕ2 = div(f.y * grad(c.uₕ))
    return nothing
end
n_reads_writes(::Type{typeof(op_divgrad_uₕ!)}) = -1 # todo
function op_divUpwind3rdOrderBiasedProductC2F!(c, f, bcs)
    upwind = Operators.Upwind3rdOrderBiasedProductC2F(bcs.inner)
    divf2c = Operators.DivergenceF2C(bcs.outer)
    @. c.y = divf2c(upwind(f.w, c.x))
    return nothing
end
n_reads_writes(::Type{typeof(op_divUpwind3rdOrderBiasedProductC2F!)}) = -1 # todo

#### Nested and fused expressions (boundary conditions fixed inside each op)
function op_nest_interp_4!(c, f, bcs)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜinterp = Operators.InterpolateF2C()
    @. c.y = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(c.x))))
    return nothing
end
n_reads_writes(::Type{typeof(op_nest_interp_4!)}) = -1 # todo
function op_nest_interp_8!(c, f, bcs)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜinterp = Operators.InterpolateF2C()
    @. c.y = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(c.x))))))))
    return nothing
end
n_reads_writes(::Type{typeof(op_nest_interp_8!)}) = -1 # todo
costly(x) = exp(sin(x))
function op_nest_costly_8!(c, f, bcs)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜinterp = Operators.InterpolateF2C()
    @. c.y = ᶜinterp(
        costly(
            ᶠinterp(
                costly(
                    ᶜinterp(
                        costly(
                            ᶠinterp(
                                costly(
                                    ᶜinterp(
                                        costly(
                                            ᶠinterp(
                                                costly(
                                                    ᶜinterp(costly(ᶠinterp(costly(c.x)))),
                                                ),
                                            ),
                                        ),
                                    )),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return nothing
end
n_reads_writes(::Type{typeof(op_nest_costly_8!)}) = -1 # todo
function op_biharmonic!(c, f, bcs)
    FT = Spaces.undertype(axes(c))
    ᶠgrad = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    ᶜdiv = Operators.DivergenceF2C()
    @. c.y = ᶜdiv(ᶠgrad(ᶜdiv(ᶠgrad(c.x))))
    return nothing
end
n_reads_writes(::Type{typeof(op_biharmonic!)}) = -1 # todo
function op_wide_sum_8!(c, f, bcs)
    ᶜinterp = Operators.InterpolateF2C()
    @. c.y =
        ᶜinterp(f.x) + ᶜinterp(f.y) + ᶜinterp(f.D) + ᶜinterp(f.U) +
        ᶜinterp(f.s1) + ᶜinterp(f.s2) + ᶜinterp(f.s3) + ᶜinterp(f.s4)
    return nothing
end
n_reads_writes(::Type{typeof(op_wide_sum_8!)}) = 9 # 1 write + 8 reads (0 metric terms)
function op_diffusion_like!(c, f, bcs)
    FT = Spaces.undertype(axes(c))
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠgrad = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    ᶜdiv = Operators.DivergenceF2C()
    @. c.y = ᶜdiv(ᶠinterp(c.s1) * ᶠgrad(c.x))
    return nothing
end
n_reads_writes(::Type{typeof(op_diffusion_like!)}) = -1 # todo
function op_advection_like!(c, f, bcs)
    FT = Spaces.undertype(axes(c))
    CT3 = Geometry.Contravariant3Vector
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜdiv = Operators.DivergenceF2C(
        bottom = Operators.SetValue(CT3(FT(0))),
        top = Operators.SetValue(CT3(FT(0))),
    )
    @. c.y = -(ᶜdiv(ᶠinterp(c.D) * f.ᶠu³ * ᶠinterp(c.U)))
    return nothing
end
n_reads_writes(::Type{typeof(op_advection_like!)}) = -1 # todo
function op_atmos_like!(c, f, bcs)
    FT = Spaces.undertype(axes(c))
    CT3 = Geometry.Contravariant3Vector
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠgrad = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    ᶜdiv = Operators.DivergenceF2C(
        bottom = Operators.SetValue(CT3(FT(0))),
        top = Operators.SetValue(CT3(FT(0))),
    )
    ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F()
    @. c.y =
        -(ᶜdiv(ᶠinterp(c.D) * f.ᶠu³ * ᶠinterp(c.U))) - ᶜdiv(ᶠupwind3(f.w, c.x)) +
        ᶜdiv(ᶠinterp(c.s1) * ᶠgrad(c.s2))
    return nothing
end
n_reads_writes(::Type{typeof(op_atmos_like!)}) = -1 # todo

function op_broadcast_example0!(c, f, bcs)
    Fields.bycolumn(axes(f.ᶠu³)) do colidx
        @. f.ᶠu³[colidx] = f.ᶠu³[colidx] + f.ᶠu³[colidx]
    end
    return nothing
end
n_reads_writes(::Type{typeof(op_broadcast_example0!)}) = 3 # 1 write, 2 reads (0 metric term)

function op_broadcast_example1!(c, f, bcs)
    Fields.bycolumn(axes(f.ᶠu³)) do colidx
        CT3 = Geometry.Contravariant3Vector
        @. f.ᶠu³[colidx] = f.ᶠuₕ³[colidx] + CT3(f.ᶠw[colidx])
    end
    return nothing
end
n_reads_writes(::Type{typeof(op_broadcast_example1!)}) = 4 # 1 write, 3 reads (1 metric term)

function op_broadcast_example2!(c, f, bcs)
    CT3 = Geometry.Contravariant3Vector
    @. f.ᶠu³ = f.ᶠuₕ³ + CT3(f.ᶠw)
    return nothing
end
n_reads_writes(::Type{typeof(op_broadcast_example2!)}) = 4 # 1 write, 3 reads (1 metric term)

#=
#####
##### Remaining TODOs
#####

# Collect common examples in ClimaAtmos:
norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
ᶜdivᵥ(ᶠinterp(ρe_tot + ᶜp) * ᶠw)
ᶜdivᵥ(ᶠinterp(ρ) * ᶠupwind_product(ᶠw, (ρe_tot + ᶜp) / ρ))
Yₜ.c.uₕ -= Geometry.Covariant12Vector(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × (project(Contravariant12Axis(), ᶜuvw))
@. ᶠK_E = eddy_diffusivity_coefficient(norm(ᶠv_a), ᶠz_a, ᶠinterp(ᶜp))

# Collect examples in TurbulenceConvection
(TODO)

# Composing with non-stencil operations
 - Example: `Geometry.project(Geometry.Contravariant12Axis(), ᶠinterp(ᶜuvw))`

# Full "core" operator list (not including composed):

```
# 2-point stencils
DivergenceF2C
DivergenceC2F
GradientF2C
GradientC2F
InterpolateF2C
InterpolateC2F
BottomBiasedC2F
BottomBiasedF2C
TopBiasedC2F
TopBiasedF2C

# Additional operators
UpwindBiasedProductC2F
Upwind3rdOrderBiasedProductC2F
CurlC2F
```
=#
