using Test
using StaticArrays, IntervalSets, LinearAlgebra
using BenchmarkTools

import ClimaCore: Domains, Meshes, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

@testset "Performance of FD operator instructions" begin
    FT = Float64
    n_elems = 1000
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
    function field_wrapper(space, nt::NamedTuple)
        cmv(z) = nt
        return cmv.(Fields.coordinate_field(space))
    end
    field_vars() = (; x = FT(0), y = FT(0), D = FT(0), U = FT(0))
    cfield = field_wrapper(cs, field_vars())
    ffield = field_wrapper(fs, field_vars())
    wvec_glob = Geometry.WVector

    #####
    ##### Array implementation
    #####
    function second_order_op_raw_arrays_loop!(x, y, D, U)
        @inbounds for i in eachindex(x)
            x[i] = D[i]*y[i] + U[i]*y[i]
        end
        return nothing
    end

    function second_order_op_raw_arrays_bc!(x, y, D, U)
        @. x = D*y + U*y
        return nothing
    end
    D = zeros(FT, n_elems)
    U = zeros(FT, n_elems)
    xarr = rand(FT, n_elems)
    yarr = rand(FT, n_elems)

    #####
    ##### Field implementation (`Base.copyto!`)
    #####
    function second_order_op_field_only_Basecopyto!(x, y, D, U)
        parent(x) .= parent(D) .* parent(y) .+ parent(U) .* parent(y)
        return nothing
    end

    #####
    ##### Field implementation (only `copyto!`)
    #####
    function second_order_op_field_only_copyto!(x, y, D, U)
        @. x = D*y + U*y
        return nothing
    end

    #####
    ##### Field implementation (with `apply_stencil!`)
    #####
    function second_order_op_field_apply_stencil!(c, f)
        ∇f = Operators.DivergenceF2C()
        wvec = Geometry.WVector
        @. c.x = ∇f(wvec(f.y))
        return nothing
    end

    #####
    ##### Field implementation (with `apply_stencil!` and boundary window)
    #####
    function second_order_op_field_bcs!(c, f)
        wvec = Geometry.WVector
        FT = Spaces.undertype(axes(c))
        ∇f = Operators.DivergenceC2F(;
            bottom = Operators.SetValue(wvec(FT(0))),
            top = Operators.SetValue(wvec(FT(0))),
        )
        @. f.x = ∇f(wvec(c.y))
        return nothing
    end

    println("\n############################ Arrays loop")
    trial = @benchmark second_order_op_raw_arrays_loop!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Arrays bc")
    trial = @benchmark second_order_op_raw_arrays_bc!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Base copyto!")
    trial = @benchmark second_order_op_field_only_Basecopyto!($(cfield.x), $(cfield.y), $(cfield.D), $(cfield.U))
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field copyto!")
    trial = @benchmark second_order_op_field_only_copyto!($(cfield.x), $(cfield.y), $(cfield.D), $(cfield.U))
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil!")
    trial = @benchmark second_order_op_field_apply_stencil!($cfield, $ffield)
    show(stdout, MIME("text/plain"), trial)

    println("\n############################ Field apply_stencil! with BC")
    trial = @benchmark second_order_op_field_bcs!($cfield, $ffield)
    show(stdout, MIME("text/plain"), trial)
end

second_order_op_raw_arrays_loop!(xarr, yarr, D, U)
second_order_op_raw_arrays_bc!(xarr, yarr, D, U)
second_order_op_field_only_Basecopyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
second_order_op_field_only_copyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
second_order_op_field_apply_stencil!(cfield, ffield)
second_order_op_field_bcs!(cfield, ffield)

@code_lowered second_order_op_raw_arrays_loop!(xarr, yarr, D, U)
@code_lowered second_order_op_raw_arrays_bc!(xarr, yarr, D, U)
@code_lowered second_order_op_field_only_Basecopyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_lowered second_order_op_field_only_copyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_lowered second_order_op_field_apply_stencil!(cfield, ffield)
@code_lowered second_order_op_field_bcs!(cfield, ffield)

@code_typed second_order_op_raw_arrays_loop!(xarr, yarr, D, U)
@code_typed second_order_op_raw_arrays_bc!(xarr, yarr, D, U)
@code_typed second_order_op_field_only_Basecopyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_typed second_order_op_field_only_copyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_typed second_order_op_field_apply_stencil!(cfield, ffield)
@code_typed second_order_op_field_bcs!(cfield, ffield)

@code_native second_order_op_raw_arrays_loop!(xarr, yarr, D, U)
@code_native second_order_op_raw_arrays_bc!(xarr, yarr, D, U)
@code_native second_order_op_field_only_Basecopyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_native second_order_op_field_only_copyto!((cfield.x), (cfield.y), (cfield.D), (cfield.U))
@code_native second_order_op_field_apply_stencil!(cfield, ffield)
@code_native second_order_op_field_bcs!(cfield, ffield)
