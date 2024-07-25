import ClimaCore
using ClimaComms
ClimaComms.@import_required_backends
using BenchmarkTools
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

import ClimaCore.Operators: half, PlusHalf
const using_cuda = ClimaComms.device() isa ClimaComms.CUDADevice

const n_tuples = 3

a_bcs(::Type{FT}, i::Int) where {FT} =
    (; bottom = Operators.SetValue(FT(0)), top = Operators.Extrapolate())

function alloc_test_f2c_interp(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    Ic = Operators.InterpolateF2C()
    # Compile first
    #! format: off
    @. cfield.cz = cfield.cx * cfield.cy * Ic(ffield.fy) * Ic(ffield.fx) * cfield.cϕ * cfield.cψ
    p = @allocated begin
        @. cfield.cz = cfield.cx * cfield.cy * Ic(ffield.fy) * Ic(ffield.fx) * cfield.cϕ * cfield.cψ
    end
    #! format: off
    @test p == 0 broken = using_cuda
    @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
    p = @allocated begin
        @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
    closure() = @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
    closure()
    p = @allocated begin
        closure()
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_c2f_interp(cfield, ffield, If)
    (;fx,fy,fz,fϕ,fψ) = ffield
    (;cx,cy,cz,cϕ,cψ) = cfield
    wvec = Geometry.WVector
    # Compile first
    #! format: off
    @. ffield.fz = ffield.fx * ffield.fy * If(cfield.cy) * If(cfield.cx) * ffield.fϕ * ffield.fψ
    p = @allocated begin
        @. ffield.fz = ffield.fx * ffield.fy * If(cfield.cy) * If(cfield.cx) * ffield.fϕ * ffield.fψ
    end
    #! format: on
    @test p == 0 broken = using_cuda
    @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
    p = @allocated begin
        @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
    end
    @test p == 0 broken = using_cuda
    fclosure() = @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
    fclosure()
    p = @allocated begin
        fclosure()
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_derivative(cfield, ffield, ∇c, ∇f)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    ##### F2C
    wvec = Geometry.WVector
    # Compile first
    #! format: off
    @. cfield.cz =
        cfield.cx * cfield.cy * ∇c(wvec(ffield.fy)) * ∇c(wvec(ffield.fx)) * cfield.cϕ * cfield.cψ
    p = @allocated begin
        @. cfield.cz = cfield.cx * cfield.cy * ∇c(wvec(ffield.fy)) * ∇c(wvec(ffield.fx)) * cfield.cϕ * cfield.cψ
    end
    #! format: on
    @test p == 0 broken = using_cuda
    @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
    c∇closure() = @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    c∇closure()
    p = @allocated begin
        c∇closure()
    end
    @test p == 0 broken = using_cuda

    ##### C2F
    # wvec = Geometry.WVector # cannot re-define, otherwise many allocations

    # Compile first
    @. fz = fx * fy * ∇f(wvec(cy)) * ∇f(wvec(cx)) * fϕ * fψ
    p = @allocated begin
        @. fz = fx * fy * ∇f(wvec(cy)) * ∇f(wvec(cx)) * fϕ * fψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_redefined_operators(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    ∇c = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    # Compile first
    @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    end
    @test_broken p == 0
    c∇closure1() = @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    c∇closure1()
    p = @allocated begin
        c∇closure1()
    end
    @test_broken p == 0

    # Now simply repeat above:
    ∇c = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    # Compile first
    @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    end
    @test_broken p == 0
    c∇closure2() = @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
    c∇closure2()
    p = @allocated begin
        c∇closure2()
    end
    @test_broken p == 0
end

function alloc_test_operators_in_loops(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    for i in 1:3
        wvec = Geometry.WVector
        bcval = i * 2
        bcs = (;
            bottom = Operators.SetValue(wvec(bcval)),
            top = Operators.SetValue(wvec(bcval)),
        )
        ∇c = Operators.DivergenceF2C(; bcs...)
        # Compile first
        @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
        p = @allocated begin
            @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
        end
        @test p == 0 broken = using_cuda
        c∇closure() = @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
        c∇closure()
        p = @allocated begin
            c∇closure()
        end
        @test p == 0 broken = using_cuda
    end
end
function alloc_test_nested_expressions_1(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    ∇c = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    LB = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(1))
    @. cz = cx * cy * ∇c(wvec(LB(cy))) * ∇c(wvec(LB(cx))) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(LB(cy))) * ∇c(wvec(LB(cx))) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_2(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    ∇c = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    RB = Operators.RightBiasedC2F(; top = Operators.SetValue(1))
    @. cz = cx * cy * ∇c(wvec(RB(cy))) * ∇c(wvec(RB(cx))) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(RB(cy))) * ∇c(wvec(RB(cx))) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_3(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    Ic = Operators.InterpolateF2C()
    ∇c = Operators.DivergenceF2C()
    wvec = Geometry.WVector
    LB = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(1))
    #! format: off
    @. cz = cx * cy * ∇c(wvec(LB(Ic(fy) * cx))) * ∇c(wvec(LB(Ic(fy) * cx))) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(LB(Ic(fy) * cx))) * ∇c(wvec(LB(Ic(fy) * cx))) * cϕ * cψ
    end
    #! format: on
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_4(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    wvec = Geometry.WVector
    If = Operators.InterpolateC2F(;
        bottom = Operators.SetValue(0),
        top = Operators.SetValue(0),
    )
    ∇f = Operators.DivergenceC2F(;
        bottom = Operators.SetValue(wvec(0)),
        top = Operators.SetValue(wvec(0)),
    )
    LB = Operators.LeftBiasedF2C(; bottom = Operators.SetValue(1))
    #! format: off
    @. fz = fx * fy * ∇f(wvec(LB(If(cy) * fx))) * ∇f(wvec(LB(If(cy) * fx))) * fϕ * fψ # Compile first
    p = @allocated begin
        @. fz = fx * fy * ∇f(wvec(LB(If(cy) * fx))) * ∇f(wvec(LB(If(cy) * fx))) * fϕ * fψ
    end
    #! format: on
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_5(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    wvec = Geometry.WVector
    If = Operators.InterpolateC2F(;
        bottom = Operators.SetValue(0),
        top = Operators.SetValue(0),
    )
    ∇c = Operators.DivergenceF2C()
    #! format: off
    @. cz = cx * cy * ∇c(wvec(If(cy) * fx)) * ∇c(wvec(If(cy) * fx)) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * ∇c(wvec(If(cy) * fx)) * ∇c(wvec(If(cy) * fx)) * cϕ * cψ
    end
    #! format: off
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_6(cfield, ffield)
    (;fx,fy,fz,fϕ,fψ) = ffield
    (;cx,cy,cz,cϕ,cψ) = cfield
    wvec = Geometry.WVector
    Ic = Operators.InterpolateF2C()
    ∇f = Operators.DivergenceC2F(;
        bottom = Operators.SetValue(wvec(0)),
        top = Operators.SetValue(wvec(0)),
    )
    #! format: off
    @. fz = fx * fy * ∇f(wvec(Ic(fy) * cx)) * ∇f(wvec(Ic(fy) * cx)) * fϕ * fψ # Compile first
    p = @allocated begin
        @. fz = fx * fy * ∇f(wvec(Ic(fy) * cx)) * ∇f(wvec(Ic(fy) * cx)) * fϕ * fψ
    end
    #! format: on
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_7(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    # similar to alloc_test_nested_expressions_8
    Ic = Operators.InterpolateF2C()
    @. cz = cx * cy * Ic(fy) * Ic(fy) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * Ic(fy) * Ic(fy) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_8(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    wvec = Geometry.WVector
    Ic = Operators.InterpolateF2C()
    @. cz = cx * cy * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = cx * cy * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_9(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    wvec = Geometry.WVector
    Ic = Operators.InterpolateF2C()
    @. cz = Int(cx < cy) * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ # Compile first
    p = @allocated begin
        @. cz = Int(cx < cy) * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_10(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    Ic = Operators.InterpolateF2C()
    @. cz = ifelse(cx < cy, abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ, 0) # Compile first
    p = @allocated begin
        @. cz = ifelse(cx < cy, abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ, 0)
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_11(cfield, ffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield
    If = Operators.InterpolateC2F(;
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )
    @. fz = fx * fy * abs(If(cy * cx)) * abs(If(cy * cx)) * fϕ * fψ # Compile first
    p = @allocated begin
        @. fz = fx * fy * abs(If(cy * cx)) * abs(If(cy * cx)) * fϕ * fψ
    end
    @test p == 0 broken = using_cuda
end

function alloc_test_nested_expressions_12(cfield, ffield, ntcfield, ntffield)
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield

    Ic = Operators.InterpolateF2C()
    cnt = ntcfield.nt
    fnt = ntffield.nt

    # Compile first
    @inbounds for i in 1:n_tuples
        cnt_i = cnt.:($i)
        fnt_i = fnt.:($i)
        cxnt = cnt_i.cx
        fxnt = fnt_i.fx
        cynt = cnt_i.cy
        fynt = fnt_i.fy
        cznt = cnt_i.cz
        fznt = fnt_i.fz
        cϕnt = cnt_i.cϕ
        fϕnt = fnt_i.fϕ
        cψnt = cnt_i.cψ
        fψnt = fnt_i.fψ
        @. cznt = cxnt * cynt * Ic(fynt) * Ic(fynt) * cϕnt * cψnt
    end

    @inbounds for i in 1:n_tuples
        p_i = @allocated begin
            cnt_i = cnt.:($i)
            fnt_i = fnt.:($i)
        end
        @test_broken p_i == 0
        cxnt = cnt_i.cx
        fxnt = fnt_i.fx
        cynt = cnt_i.cy
        fynt = fnt_i.fy
        cznt = cnt_i.cz
        fznt = fnt_i.fz
        cϕnt = cnt_i.cϕ
        fϕnt = fnt_i.fϕ
        cψnt = cnt_i.cψ
        fψnt = fnt_i.fψ
        p = @allocated begin
            @. cznt = cxnt * cynt * Ic(fynt) * Ic(fynt) * cϕnt * cψnt
        end
        @test_broken p == 0
    end
end

function alloc_test_nested_expressions_13(
    cfield,
    ffield,
    ntcfield,
    ntffield,
    ::Type{FT},
) where {FT}
    (; fx, fy, fz, fϕ, fψ) = ffield
    (; cx, cy, cz, cϕ, cψ) = cfield

    Ic = Operators.InterpolateF2C()
    cnt = ntcfield.nt
    fnt = ntffield.nt
    wvec = Geometry.WVector

    adv_bcs = (;
        bottom = Operators.SetValue(wvec(FT(0))),
        top = Operators.SetValue(wvec(FT(0))),
    )
    LBC = Operators.LeftBiasedF2C(; bottom = Operators.SetValue(FT(0)))
    zero_bcs =
        (; bottom = Operators.SetValue(FT(0)), top = Operators.SetValue(FT(0)))
    I0f = Operators.InterpolateC2F(; zero_bcs...)
    ∇f = Operators.DivergenceC2F(; adv_bcs...)

    # Compile first
    @inbounds for i in 1:n_tuples
        cnt_i = cnt.:($i)
        fnt_i = fnt.:($i)
        cxnt = cnt_i.cx
        fxnt = fnt_i.fx
        fynt = fnt_i.fy
        a_up_bcs = a_bcs(FT, i)
        Iaf1 = Operators.InterpolateC2F(; a_up_bcs...)
        @. fynt =
            -(∇f(wvec(LBC(Iaf1(cxnt) * fx * fxnt * fxnt)))) +
            (fx * Iaf1(cxnt) * fxnt * (I0f(cz) * fy - I0f(cy) * fxnt)) +
            (fx * Iaf1(cxnt) * I0f(cϕ)) +
            fψ
    end

    @inbounds for i in 1:n_tuples
        cnt_i = cnt.:($i)
        fnt_i = fnt.:($i)
        cxnt = cnt_i.cx
        fxnt = fnt_i.fx
        fynt = fnt_i.fy
        #! format: off
        p_i = @allocated begin
            a_up_bcs = a_bcs(FT, i)
            Iaf2 = Operators.InterpolateC2F(; a_up_bcs...)
            # add extra parentheses so that we call +(+(a,b,c),d), as +(a,b,c,d) triggers allocations
            @. fynt =
                (-(∇f(wvec(LBC(Iaf2(cxnt) * fx * fxnt * fxnt)))) +
                (fx * Iaf2(cxnt) * fxnt * (I0f(cz) * fy - I0f(cy) * fxnt)) +
                (fx * Iaf2(cxnt) * I0f(cϕ))) +
                fψ
        end
        #! format: on
        @test_broken p_i == 0
    end
end

@testset "FD operator allocation tests" begin
    FT = Float64
    n_elems = 1000
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = n_elems)
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)
    cfield_vars() =
        (; cx = FT(0), cy = FT(0), cz = FT(0), cϕ = FT(0), cψ = FT(0))
    ffield_vars() =
        (; fx = FT(0), fy = FT(0), fz = FT(0), fϕ = FT(0), fψ = FT(0))
    cntfield_vars() = (; nt = ntuple(i -> cfield_vars(), n_tuples))
    fntfield_vars() = (; nt = ntuple(i -> ffield_vars(), n_tuples))
    cfield = fill(cfield_vars(), cs)
    ffield = fill(ffield_vars(), fs)
    ntcfield = fill(cntfield_vars(), cs)
    ntffield = fill(fntfield_vars(), fs)
    wvec_glob = Geometry.WVector

    alloc_test_f2c_interp(cfield, ffield)

    alloc_test_c2f_interp(
        cfield,
        ffield,
        Operators.InterpolateC2F(;
            bottom = Operators.SetValue(0),
            top = Operators.SetValue(0),
        ),
    )
    alloc_test_c2f_interp(
        cfield,
        ffield,
        Operators.InterpolateC2F(;
            bottom = Operators.SetGradient(wvec_glob(0)),
            top = Operators.SetGradient(wvec_glob(0)),
        ),
    )
    alloc_test_c2f_interp(
        cfield,
        ffield,
        Operators.InterpolateC2F(;
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        ),
    )
    alloc_test_c2f_interp(
        cfield,
        ffield,
        Operators.LeftBiasedC2F(; bottom = Operators.SetValue(0)),
    )
    alloc_test_c2f_interp(
        cfield,
        ffield,
        Operators.RightBiasedC2F(; top = Operators.SetValue(0)),
    )

    alloc_test_derivative(
        cfield,
        ffield,
        Operators.DivergenceF2C(),
        Operators.DivergenceC2F(;
            bottom = Operators.SetValue(wvec_glob(0)),
            top = Operators.SetValue(wvec_glob(0)),
        ),
    )
    alloc_test_derivative(
        cfield,
        ffield,
        Operators.DivergenceF2C(;
            bottom = Operators.SetValue(wvec_glob(0)),
            top = Operators.SetValue(wvec_glob(0)),
        ),
        Operators.DivergenceC2F(;
            bottom = Operators.SetValue(wvec_glob(0)),
            top = Operators.SetValue(wvec_glob(0)),
        ),
    )
    alloc_test_derivative(
        cfield,
        ffield,
        Operators.DivergenceF2C(;
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        ),
        Operators.DivergenceC2F(;
            bottom = Operators.SetDivergence(0),
            top = Operators.SetDivergence(0),
        ),
    )

    alloc_test_redefined_operators(cfield, ffield)
    alloc_test_operators_in_loops(cfield, ffield)
    alloc_test_nested_expressions_1(cfield, ffield)
    alloc_test_nested_expressions_2(cfield, ffield)
    alloc_test_nested_expressions_3(cfield, ffield)
    alloc_test_nested_expressions_4(cfield, ffield)
    alloc_test_nested_expressions_5(cfield, ffield)
    alloc_test_nested_expressions_6(cfield, ffield)
    alloc_test_nested_expressions_7(cfield, ffield)
    alloc_test_nested_expressions_8(cfield, ffield)
    alloc_test_nested_expressions_9(cfield, ffield)
    alloc_test_nested_expressions_10(cfield, ffield)
    alloc_test_nested_expressions_11(cfield, ffield)
    alloc_test_nested_expressions_12(cfield, ffield, ntcfield, ntffield)
    alloc_test_nested_expressions_13(cfield, ffield, ntcfield, ntffield, FT)
end


# https://github.com/CliMA/ClimaCore.jl/issues/1602
const CT3 = Geometry.Contravariant3Vector
const C12 = ClimaCore.Geometry.Covariant12Vector
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
function set_ᶠuₕ³!(ᶜx, ᶠx)
    ᶜJ = Fields.local_geometry_field(ᶜx).J
    @. ᶠx.ᶠuₕ³ = ᶠwinterp(ᶜx.ρ * ᶜJ, CT3(ᶜx.uₕ))
    return nothing
end
@testset "Inference/allocations when broadcasting types" begin
    FT = Float64
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 25, helem = 10)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    device = ClimaComms.device(cspace)
    @info "device = $device"
    ᶜx = fill((; uₕ = zero(C12{FT}), ρ = FT(0)), cspace)
    ᶠx = fill((; ᶠuₕ³ = zero(CT3{FT})), fspace)
    set_ᶠuₕ³!(ᶜx, ᶠx) # compile
    p_allocated = @allocated set_ᶠuₕ³!(ᶜx, ᶠx)
    @show p_allocated

    trial = @benchmark ClimaComms.@cuda_sync $device set_ᶠuₕ³!($ ᶜx, $ᶠx)
    show(stdout, MIME("text/plain"), trial)
end
