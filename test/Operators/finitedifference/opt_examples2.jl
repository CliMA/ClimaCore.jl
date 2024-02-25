#=
julia --project=test
using Revise; include("test/Operators/finitedifference/opt_examples2.jl")
=#
using Revise
import ClimaCore;
import ClimaComms;
using Test
import ClimaCore.Fields
import ClimaCore.Operators
import ClimaCore.Spaces
import ClimaCore.Geometry
using BenchmarkTools;
@isdefined(TU) || include(joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"));
import .TestUtilities as TU;
FT = Float64;
zelem=25
cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT;zelem, helem=10);
fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace);
const CT3 = Geometry.Contravariant3Vector
const C12 = ClimaCore.Geometry.Covariant12Vector
@show ClimaComms.device(cspace)
ᶜx = fill((;uₕ=zero(C12{FT}) ,ρ=FT(0)), cspace);
ᶠx = fill((;ᶠuₕ³=zero(CT3{FT})), fspace);

const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

function set_ᶠuₕ³!(ᶜx, ᶠx)
    ᶜJ = Fields.local_geometry_field(ᶜx).J
    @. ᶠx.ᶠuₕ³ = ᶠwinterp(ᶜx.ρ * ᶜJ, CT3(ᶜx.uₕ))
    return nothing
end

set_ᶠuₕ³!(ᶜx, ᶠx) # compile
p_allocated = @allocated set_ᶠuₕ³!(ᶜx, ᶠx)
@show p_allocated

using BenchmarkTools, CUDA
CUDA.@sync @benchmark set_ᶠuₕ³!(ᶜx, ᶠx)

