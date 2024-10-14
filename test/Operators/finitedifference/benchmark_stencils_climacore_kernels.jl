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
function op_LeftBiasedC2F!(c, f, bcs)
    interp = Operators.LeftBiasedC2F(bcs)
    @. f.x = interp(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_LeftBiasedC2F!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_LeftBiasedF2C!(c, f, bcs = (;))
    interp = Operators.LeftBiasedF2C(bcs)
    @. c.x = interp(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_LeftBiasedF2C!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_RightBiasedC2F!(c, f, bcs)
    interp = Operators.RightBiasedC2F(bcs)
    @. f.x = interp(c.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_RightBiasedC2F!)}) = 2 # 1 write + 1 reads (0 metric terms)
function op_RightBiasedF2C!(c, f, bcs = (;))
    interp = Operators.RightBiasedF2C(bcs)
    @. c.x = interp(f.y)
    return nothing
end
n_reads_writes(::Type{typeof(op_RightBiasedF2C!)}) = 2 # 1 write + 1 reads (0 metric terms)
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

function op_broadcast_example0!(c, f, bcs)
    Fields.bycolumn(axes(f.ᶠu³)) do colidx
        CT3 = Geometry.Contravariant3Vector
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

# Add tests for Operator2Stencil's
 - how to add Operator2Stencil tests?
 - can we leverage the existing tests?
 - Example: `ᶜdivᵥ_stencil(ᶠinterp(ᶜρ) * one(ᶠw))`

# Full "core" operator list (not including composed):

```
# 2-point stencils
DivergenceF2C
DivergenceC2F
GradientF2C
GradientC2F
InterpolateF2C
InterpolateC2F
LeftBiasedC2F
LeftBiasedF2C
RightBiasedC2F
RightBiasedF2C

# Additional operators
UpwindBiasedProductC2F
Upwind3rdOrderBiasedProductC2F
CurlC2F

# Unused (at the moment) operators
LeftBiased3rdOrderC2F  # unused
LeftBiased3rdOrderF2C  # unused
RightBiased3rdOrderC2F # unused
RightBiased3rdOrderF2C # unused
AdvectionF2F           # only used in ClimaAtmos src/
AdvectionC2C           # only used in ClimaAtmos src/
FluxCorrectionC2C      # unused at the moment
FluxCorrectionF2F      # only used in ClimaAtmos src/
```
=#
