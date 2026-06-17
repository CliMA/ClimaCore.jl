include("utils_2d.jl")

@testset "1D SE, 1D FD Extruded Domain vertical advection operator" begin

    function advection(c, f, hv_center_space)
        adv = zeros(eltype(f), hv_center_space)
        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
            top = Operators.SetGradient(Geometry.WVector(FT(1))),
        )
        interpf2c = Operators.InterpolateF2C()
        return @. adv =
            interpf2c(LinearAlgebra.dot(Geometry.Contravariant3Vector(c), gradc2f(f)))
    end

    n_elems_seq = 2 .^ (5, 6, 7, 8)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        # Advection Operator
        # c ∂_z f
        # for this test, we use f(z) = sin(z) and c = 1, a WVector
        # => c ∂_z f = cos(z)
        hv_center_space, hv_face_space = hvspace_2D(helem = n, velem = n)

        Δh[k] = 1.0 / n

        # advective velocity
        c = Geometry.WVector.(ones(Float64, hv_face_space),)
        # scalar-valued field to be advected
        f = sin.(Fields.coordinate_field(hv_center_space).z)

        # Call the advection operator
        adv = advection(c, f, hv_center_space)

        err[k] = norm(adv .- cos.(Fields.coordinate_field(hv_center_space).z))
    end
    # AdvectionC2C convergence rate
    conv_adv_c2c = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_adv_c2c[1] ≈ 2 atol = 0.1
    @test conv_adv_c2c[2] ≈ 2 atol = 0.1
    @test conv_adv_c2c[3] ≈ 2 atol = 0.1
end


@testset "1D SE, 1D FD Extruded Domain horz & vert divergence operator, with Extrapolate BCs" begin

    # Divergence operator in 2D Cartesian domain with
    # Cₕ = (cₕ, 0), Cᵥ = (0, cᵥ)
    # ∇ₕ⋅(Cₕ * f) + ∇ᵥ⋅(Cᵥ * f)
    # here cₕ == cᵥ == 1

    # NOTE: the equation setup is only correct for Cartesian domains!

    function divergence(h, bottom_flux)

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c =
            Operators.DivergenceF2C(bottom = Operators.SetValue(bottom_flux))
        # only upward component of divergence
        dh = @. divf2c(Ic2f(h))

        # only horizontal component of divergence
        hdiv = Operators.Divergence()
        @. dh += hdiv(h) # add the two components for full divergence +=
        Spaces.weighted_dss!(dh)

        return dh
    end


    # A horizontal Gaussian blob, centered at (x_0, z_0),
    # with `a` the height of the blob and σ the standard deviation
    function h_blob(x, z, x_0, z_0, a = 1.0, σ = 0.2)
        a * exp(-((x - x_0)^2 + (z - z_0)^2) / (2 * σ^2))
    end
    # Spatial gradient of a horizontal Gaussian blob, centered at (x_0, z_0),
    # with `a` the height of the blob and σ the standard deviation
    ∇h_blob(x, z, x_0, z_0, a = 1.0, σ = 0.2) =
        -a * (exp(-((x - x_0)^2 + (z - z_0)^2) / (2 * σ^2))) / (σ^2) *
        Geometry.UWVector(x - x_0, z - z_0)


    n_elems_seq = 2 .^ (6, 7, 8, 9)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        hv_center_space, hv_face_space =
            hvspace_2D(xlim = (-1, 1), zlim = (-1, 1), helem = n, velem = n)
        ccoords = Fields.coordinate_field(hv_center_space)
        bcoords =
            Fields.coordinate_field(Spaces.horizontal_space(hv_center_space))

        Δh[k] = 1.0 / n

        h =
            h_blob.(ccoords.x, ccoords.z, -0.5, -0.5) .*
            Ref(Geometry.UWVector(1.0, 1.0))
        bottom_flux =
            h_blob.(bcoords.x, -1.0, -0.5, -0.5) .*
            Ref(Geometry.UWVector(1.0, 1.0))
        exact_divh =
            Ref(Geometry.UWVector(1.0, 1.0)') .*
            ∇h_blob.(ccoords.x, ccoords.z, -0.5, -0.5)

        divh = divergence(h, bottom_flux)

        err[k] = norm(exact_divh .- divh)
    end
    # Divergence convergence rate
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.01
    conv_comp_div = convergence_rate(err, Δh)
    #= not sure these are right?
    @test conv_comp_div[1] ≈ 0.5 atol = 0.1
    @test conv_comp_div[2] ≈ 0.5 atol = 0.1
    @test conv_comp_div[3] ≈ 0.5 atol = 0.1
    =#
end

@testset "1D SE, 1D FD Extruded Domain horz & vert divergence operator, with Dirichelet BCs" begin

    # Divergence operator in 2D Cartesian domain with
    # Cₕ = (cₕ, 0), Cᵥ = (0, cᵥ)
    # ∇ₕ⋅(Cₕ * f) + ∇ᵥ⋅(Cᵥ * f)
    # here cₕ == cᵥ == 1

    # NOTE: the equation setup is only correct for Cartesian domains!

    function divergence(f, hv_center_space)

        divuf = Fields.FieldVector(h = zeros(eltype(f), hv_center_space))
        h = f.h
        dh = divuf.h

        # vertical advection no inflow at bottom
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(
            top = Operators.SetValue(0.0),
            bottom = Operators.SetValue(0.0),
        )
        divf2c = Operators.DivergenceF2C()
        # only upward component of divergence
        @. dh = divf2c(Ic2f(h) * Geometry.WVector(1.0))

        # only horizontal component of divergence
        hdiv = Operators.Divergence()
        @. dh += hdiv(h * Geometry.UVector(1.0)) # add the two components for full divergence +=
        Spaces.weighted_dss!(dh)

        return divuf
    end

    # A horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function h_blob(coords, x_0, z_0, a = 1.0, σ = 0.2)
        f = map(coords) do coord
            a * exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))
        end

        return f
    end

    # Spatial derivative of a horizontal Gaussian blob, centered at (-x_0, -z_0),
    # with `a` the height of the blob and σ the standard deviation
    function div_h_blob(coords, x_0, z_0, a = 1.0, σ = 0.2)
        div_uf = map(coords) do coord
            -a * (exp(-((coord.x + x_0)^2 + (coord.z + z_0)^2) / (2 * σ^2))) /
            (σ^2) * ((coord.x + x_0) + (coord.z + z_0))
        end

        return div_uf
    end

    n_elems_seq = 2 .^ (6, 7, 8, 9)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        hv_center_space, hv_face_space =
            hvspace_2D(xlim = (-1, 1), zlim = (-1, 1), helem = n, velem = n)

        Δh[k] = 1.0 / n

        coords = Fields.coordinate_field(hv_center_space)
        f = Fields.FieldVector(h = h_blob(coords, 0.0, 0.0))

        divuf = divergence(f, hv_center_space)
        exact_divuf = div_h_blob(coords, 0.0, 0.0)

        err[k] = norm(exact_divuf .- divuf.h)
    end
    # Divergence convergence rate
    conv_comp_div = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_comp_div[1] ≈ 2 atol = 0.1
    @test conv_comp_div[2] ≈ 2 atol = 0.1
    @test conv_comp_div[3] ≈ 2 atol = 0.1
end
