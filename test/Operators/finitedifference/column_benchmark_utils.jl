using Test
using StaticArrays, IntervalSets, LinearAlgebra
using BenchmarkTools

import ClimaCore
ClimaCore.enable_threading() = false

import ClimaCore: Domains, Meshes, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

field_vars(::Type{FT}) where {FT} = (;
    x = FT(0),
    y = FT(0),
    D = FT(0),
    U = FT(0),
    ∇x = Geometry.Covariant3Vector(FT(0)),
)

function get_fields(n_elems, ::Type{FT}) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = n_elems)
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)
    cfield = field_wrapper(cs, field_vars(FT))
    ffield = field_wrapper(fs, field_vars(FT))

    D = zeros(FT, n_elems)
    U = zeros(FT, n_elems)
    xarr = rand(FT, n_elems)
    yarr = rand(FT, n_elems + 1)
    vars_contig = (; D, U, xarr, yarr)

    return (; cfield, ffield, vars_contig)
end

function field_wrapper(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

#####
##### Array implementation
#####
function op_raw_arrays_loop!(x, y, D, U)
    @inbounds for i in eachindex(x)
        x[i] = D[i] * y[i] + U[i] * y[i + 1]
    end
    return nothing
end
function op_raw_arrays_loop_views!(x, y, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:end]
    @inbounds for i in eachindex(x)
        x[i] = D[i] * y1[i] + U[i] * y2[i]
    end
    return nothing
end

function op_raw_arrays_bc!(x, y, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:end]
    @. x = D * y1 + U * y2
    return nothing
end

#####
##### Field implementation (`Base.copyto!`)
#####
function op_field_only_Basecopyto!(x, y, D, U)
    y1 = @view parent(y)[1:(end - 1)]
    y2 = @view parent(y)[2:end]
    parent(x) .= parent(D) .* y1 .+ parent(U) .* y2
    return nothing
end

#####
##### Field implementation (with `apply_stencil!`)
#####
function op_field_apply_stencil!(c, f)
    ∇f = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    @. c.x = ∇f(wvec(f.y))
    return nothing
end

function op_field_apply_stencil_grad!(c∇x, fy)
    ∇f = Operators.GradientF2C()
    @. c∇x = ∇f(fy)
    return nothing
end

#####
##### Field implementation (with `apply_stencil!` and boundary window)
#####
function op_field_bcs!(c, f)
    wvec = Geometry.WVector
    FT = Spaces.undertype(axes(c))
    ∇f = Operators.DivergenceC2F(;
        bottom = Operators.SetValue(wvec(FT(0))),
        top = Operators.SetValue(wvec(FT(0))),
    )
    @. f.x = ∇f(wvec(c.y))
    return nothing
end

function benchmark_cases(vars_contig, cfield, ffield)
    println("\n############################ Arrays loop contiguous")
    (; D, U, xarr, yarr) = vars_contig
    trial = @benchmark op_raw_arrays_loop!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Arrays loop views")
    (; D, U, xarr, yarr) = vars_contig
    trial = @benchmark op_raw_arrays_loop_views!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Arrays bc contiguous")
    (; D, U, xarr, yarr) = vars_contig
    trial = @benchmark op_raw_arrays_bc!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Base copyto!")
    trial = @benchmark op_field_only_Basecopyto!(
        $(cfield.x),
        $(ffield.y),
        $(cfield.D),
        $(cfield.U),
    )
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil!")
    trial = @benchmark op_field_apply_stencil!($cfield, $ffield)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil! grad")
    trial = @benchmark op_field_apply_stencil_grad!($(cfield.∇x), $(ffield.y))
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil! with BC")
    trial = @benchmark op_field_bcs!($cfield, $ffield)
    show(stdout, MIME("text/plain"), trial)
end

function benchmark_grad(vars_contig, cfield, ffield)
    println("\n############################ Arrays loop views")
    (; D, U, xarr, yarr) = vars_contig
    trial = @benchmark op_raw_arrays_loop_views!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil! grad")
    trial = @benchmark op_field_apply_stencil_grad!($(cfield.∇x), $(ffield.y))
    show(stdout, MIME("text/plain"), trial)
end
