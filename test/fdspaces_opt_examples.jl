using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

import ClimaCore.Operators: half, PlusHalf

if VERSION >= v"1.7.0"
    @testset "FD operator allocation tests" begin
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
        field_vars() = (; x = FT(0), y = FT(0), z = FT(0), ϕ = FT(0), ψ = FT(0))
        cfield = field_wrapper(cs, field_vars())
        ffield = field_wrapper(fs, field_vars())
        wvec_glob = Geometry.WVector

        cx = cfield.x
        fx = ffield.x
        cy = cfield.y
        fy = ffield.y
        cz = cfield.z
        fz = ffield.z
        cϕ = cfield.ϕ
        fϕ = ffield.ϕ
        cψ = cfield.ψ
        fψ = ffield.ψ

        function alloc_test_f2c_interp()
            Ic = Operators.InterpolateF2C()
            # Compile first
            #! format: off
            @. cfield.z = cfield.x * cfield.y * Ic(ffield.y) * Ic(ffield.x) * cfield.ϕ * cfield.ψ
            p = @allocated begin
                @. cfield.z = cfield.x * cfield.y * Ic(ffield.y) * Ic(ffield.x) * cfield.ϕ * cfield.ψ
            end
            #! format: off
            @test_broken p == 0
            @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
            p = @allocated begin
                @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
            end
            @test p == 0
            closure() = @. cz = cx * cy * Ic(fy) * Ic(fx) * cϕ * cψ
            closure()
            p = @allocated begin
                closure()
            end
            @test p == 0
        end
        alloc_test_f2c_interp()

        function alloc_test_c2f_interp(If)
            wvec = Geometry.WVector
            # Compile first
            #! format: off
            @. ffield.z = ffield.x * ffield.y * If(cfield.y) * If(cfield.x) * ffield.ϕ * ffield.ψ
            p = @allocated begin
                @. ffield.z = ffield.x * ffield.y * If(cfield.y) * If(cfield.x) * ffield.ϕ * ffield.ψ
            end
            #! format: on
            @test_broken p == 0
            @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
            p = @allocated begin
                @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
            end
            @test p == 0
            fclosure() = @. fz = fx * fy * If(cy) * If(cx) * fϕ * fψ
            fclosure()
            p = @allocated begin
                fclosure()
            end
            @test p == 0
        end

        alloc_test_c2f_interp(
            Operators.InterpolateC2F(;
                bottom = Operators.SetValue(0),
                top = Operators.SetValue(0),
            ),
        )
        alloc_test_c2f_interp(
            Operators.InterpolateC2F(;
                bottom = Operators.SetGradient(wvec_glob(0)),
                top = Operators.SetGradient(wvec_glob(0)),
            ),
        )
        alloc_test_c2f_interp(
            Operators.InterpolateC2F(;
                bottom = Operators.Extrapolate(),
                top = Operators.Extrapolate(),
            ),
        )
        alloc_test_c2f_interp(
            Operators.LeftBiasedC2F(; bottom = Operators.SetValue(0)),
        )
        alloc_test_c2f_interp(
            Operators.RightBiasedC2F(; top = Operators.SetValue(0)),
        )

        function alloc_test_derivative(∇c, ∇f)
            ##### F2C
            wvec = Geometry.WVector
            # Compile first
            #! format: off
            @. cfield.z =
                cfield.x * cfield.y * ∇c(wvec(ffield.y)) * ∇c(wvec(ffield.x)) * cfield.ϕ * cfield.ψ
            p = @allocated begin
                @. cfield.z = cfield.x * cfield.y * ∇c(wvec(ffield.y)) * ∇c(wvec(ffield.x)) * cfield.ϕ * cfield.ψ
            end
            #! format: on
            @test_broken p == 0
            @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            p = @allocated begin
                @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            end
            @test p == 0
            c∇closure() =
                @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            c∇closure()
            p = @allocated begin
                c∇closure()
            end
            @test p == 0

            ##### C2F
            # wvec = Geometry.WVector # cannot re-define, otherwise many allocations

            # Compile first
            @. fz = fx * fy * ∇f(wvec(cy)) * ∇f(wvec(cx)) * fϕ * fψ
            p = @allocated begin
                @. fz = fx * fy * ∇f(wvec(cy)) * ∇f(wvec(cx)) * fϕ * fψ
            end
            @test p == 0
        end

        alloc_test_derivative(
            Operators.DivergenceF2C(),
            Operators.DivergenceC2F(;
                bottom = Operators.SetValue(wvec_glob(0)),
                top = Operators.SetValue(wvec_glob(0)),
            ),
        )
        alloc_test_derivative(
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
            Operators.DivergenceF2C(;
                bottom = Operators.Extrapolate(),
                top = Operators.Extrapolate(),
            ),
            Operators.DivergenceC2F(;
                bottom = Operators.SetDivergence(0),
                top = Operators.SetDivergence(0),
            ),
        )

        function alloc_test_redefined_operators()
            ∇c = Operators.DivergenceF2C()
            wvec = Geometry.WVector
            # Compile first
            @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            p = @allocated begin
                @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            end
            @test_broken p == 0
            c∇closure1() =
                @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
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
            c∇closure2() =
                @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
            c∇closure2()
            p = @allocated begin
                c∇closure2()
            end
            @test_broken p == 0
        end
        alloc_test_redefined_operators()

        function alloc_test_operators_in_loops()
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
                @test p == 0
                c∇closure() =
                    @. cz = cx * cy * ∇c(wvec(fy)) * ∇c(wvec(fx)) * cϕ * cψ
                c∇closure()
                p = @allocated begin
                    c∇closure()
                end
                @test p == 0
            end
        end
        alloc_test_operators_in_loops()

        function alloc_test_nested_expressions_1()

            ∇c = Operators.DivergenceF2C()
            wvec = Geometry.WVector
            LB = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(1))
            @. cz = cx * cy * ∇c(wvec(LB(cy))) * ∇c(wvec(LB(cx))) * cϕ * cψ # Compile first
            p = @allocated begin
                @. cz = cx * cy * ∇c(wvec(LB(cy))) * ∇c(wvec(LB(cx))) * cϕ * cψ
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_2()
            ∇c = Operators.DivergenceF2C()
            wvec = Geometry.WVector
            RB = Operators.RightBiasedC2F(; top = Operators.SetValue(1))
            @. cz = cx * cy * ∇c(wvec(RB(cy))) * ∇c(wvec(RB(cx))) * cϕ * cψ # Compile first
            p = @allocated begin
                @. cz = cx * cy * ∇c(wvec(RB(cy))) * ∇c(wvec(RB(cx))) * cϕ * cψ
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_3()
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
            @test p == 0
        end

        function alloc_test_nested_expressions_4()
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
            @test p == 0
        end

        function alloc_test_nested_expressions_5()
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
            @test p == 0
        end

        function alloc_test_nested_expressions_6()
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
            @test p == 0
        end

        function alloc_test_nested_expressions_7()
            # similar to alloc_test_nested_expressions_8
            Ic = Operators.InterpolateF2C()
            @. cz = cx * cy * Ic(fy) * Ic(fy) * cϕ * cψ # Compile first
            p = @allocated begin
                @. cz = cx * cy * Ic(fy) * Ic(fy) * cϕ * cψ
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_8()
            wvec = Geometry.WVector
            Ic = Operators.InterpolateF2C()
            @. cz = cx * cy * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ # Compile first
            p = @allocated begin
                @. cz = cx * cy * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_9()
            wvec = Geometry.WVector
            Ic = Operators.InterpolateF2C()
            @. cz = Int(cx < cy) * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ # Compile first
            p = @allocated begin
                @. cz = Int(cx < cy) * abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_10()
            Ic = Operators.InterpolateF2C()
            @. cz = ifelse(cx < cy, abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ, 0) # Compile first
            p = @allocated begin
                @. cz = ifelse(cx < cy, abs(Ic(fy)) * abs(Ic(fy)) * cϕ * cψ, 0)
            end
            @test p == 0
        end

        function alloc_test_nested_expressions_11()
            If = Operators.InterpolateC2F(;
                bottom = Operators.SetValue(0.0),
                top = Operators.SetValue(0.0),
            )
            @. fz = fx * fy * abs(If(cy * cx)) * abs(If(cy * cx)) * fϕ * fψ # Compile first
            p = @allocated begin
                @. fz = fx * fy * abs(If(cy * cx)) * abs(If(cy * cx)) * fϕ * fψ
            end
            @test p == 0
        end

        alloc_test_nested_expressions_1()
        alloc_test_nested_expressions_2()
        alloc_test_nested_expressions_3()
        alloc_test_nested_expressions_4()
        alloc_test_nested_expressions_5()
        alloc_test_nested_expressions_6()
        alloc_test_nested_expressions_7()
        alloc_test_nested_expressions_8()
        alloc_test_nested_expressions_9()
        alloc_test_nested_expressions_10()
        alloc_test_nested_expressions_11()
    end
end
