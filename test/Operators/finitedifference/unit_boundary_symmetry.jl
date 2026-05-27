using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Geometry,
    Operators

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

@testset "Finite Difference Boundary Symmetry Property Tests" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        L = FT(2)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(-L / 2),
            Geometry.ZPoint(L / 2);
            boundary_names = (:bottom, :top),
        )
        nelems = 20
        mesh = Meshes.IntervalMesh(domain; nelems)
        topology = Topologies.IntervalTopology(context, mesh)
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        zc = Fields.coordinate_field(center_space).z
        zf = Fields.coordinate_field(face_space).z

        # 1. Symmetric even input f(z) = cos(π z / L) => f(-z) == f(z)
        fc = @. cos(FT(π) * zc / L)
        ff = @. cos(FT(π) * zf / L)

        # 2. Anti-symmetric odd input g(z) = sin(π z / L) => g(-z) == -g(z)
        gc = @. sin(FT(π) * zc / L)
        gf = @. sin(FT(π) * zf / L)

        # A. Interpolation C2F and F2C: symmetric input -> symmetric output
        interp_c2f = Operators.InterpolateC2F(
            bottom = Operators.SetValue(FT(0)),
            top = Operators.SetValue(FT(0)),
        )
        interp_f2c = Operators.InterpolateF2C()

        res_ff = interp_c2f.(fc)
        arr_ff = parent(res_ff)
        # Check symmetry: arr_ff[k] == arr_ff[end - k + 1]
        @test arr_ff ≈ reverse(arr_ff)

        res_fc = interp_f2c.(ff)
        arr_fc = parent(res_fc)
        @test arr_fc ≈ reverse(arr_fc)

        # B. Gradient C2F: symmetric input with symmetric boundary values
        # produces anti-symmetric gradient: grad(f)(-z) == -grad(f)(z).
        # The boundary value x₀ = 0 is imposed through `SetGradient`, since the
        # covariant gradient on the bottom face is 2 (x[1] - x₀) and on the top
        # face is 2 (x₀ - x[end]).
        f_bottom = Fields.level(fc, 1)
        f_top = Fields.level(fc, Fields.nlevels(fc))
        grad_c2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(
                Geometry.Covariant3Vector.(2 .* f_bottom),
            ),
            top = Operators.SetGradient(
                Geometry.Covariant3Vector.(-2 .* f_top),
            ),
        )
        res_grad_c2f = Geometry.WVector.(grad_c2f.(fc))
        arr_grad_c2f = parent(res_grad_c2f)
        # Anti-symmetric: arr_grad_c2f[k] == -arr_grad_c2f[end - k + 1]
        @test arr_grad_c2f ≈ -reverse(arr_grad_c2f)

        # C. Gradient F2C: anti-symmetric input produces symmetric gradient
        grad_f2c = Operators.GradientF2C()
        res_grad_f2c = Geometry.WVector.(grad_f2c.(gf))
        arr_grad_f2c = parent(res_grad_f2c)
        @test arr_grad_f2c ≈ reverse(arr_grad_f2c)

        # D. Divergence C2F and F2C symmetry
        div_f2c = Operators.DivergenceF2C()
        # Symmetric vector field w(z) = cos(π z / L) produces anti-symmetric divergence
        vec_ff = Geometry.WVector.(ff)
        res_div_f2c = div_f2c.(vec_ff)
        arr_div = parent(res_div_f2c)
        @test arr_div ≈ -reverse(arr_div)

        # E. SetGradient symmetric BCs
        grad_c2f_sym = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
            top = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        res_grad_sym = Geometry.WVector.(grad_c2f_sym.(fc))
        arr_grad_sym = parent(res_grad_sym)
        @test arr_grad_sym ≈ -reverse(arr_grad_sym)

        # F. Zero allocation sentinels
        target_f = similar(res_ff)
        @noinline function _eval_interp!(tgt, op, src)
            @. tgt = op(src)
            return nothing
        end
        _eval_interp!(target_f, interp_c2f, fc)
        allocs = @allocated _eval_interp!(target_f, interp_c2f, fc)
        # On GPU the kernel launch itself allocates host memory, so the
        # zero-allocation sentinel only holds on CPU.
        ClimaComms.device() isa ClimaComms.CUDADevice ||
            !TU.allocation_checks_meaningful() ||
            @test allocs == 0
    end
end
