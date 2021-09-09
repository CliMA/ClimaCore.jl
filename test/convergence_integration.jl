using Test
using StaticArrays, IntervalSets, LinearAlgebra, Statistics

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
import ClimaCore.Domains.Geometry: Cartesian2DPoint

function hvspace_2D(;
    xlim = (-π, π),
    zlim = (0, 4π),
    helem::I = 10,
    velem::I = 64,
    npoly::I = 7,
    stretch_fcn = Meshes.ExponentialStretching(0.75),
) where {I <: Int}
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        FT(zlim[1]),
        FT(zlim[2]);
        x3boundary = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch_fcn, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vert_center_space)

    horzdomain = Domains.RectangleDomain(
        xlim[1]..xlim[2],
        -0..0,
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, helem, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

metric(y_pred, y_truth) = norm(y_pred .- y_truth) / norm(y_truth)

n_elems_seq = 2 .^ [3, 4, 5, 6, 7, 8]
Δh = zeros(length(n_elems_seq))
err = zeros(length(n_elems_seq))
u_true, u_pred, u_init = [], [], []

# setup
np = 3
nh = 10
#nv = 64
cₕ = 0.0
cᵥ = 1.0

# convergence test loop
for (k, nv) in enumerate(n_elems_seq)
    hv_center_space, hv_face_space = hvspace_2D(
        xlim = (-2, 2), 
        zlim = (-2, 2), 
        helem = nh, 
        velem = nv, 
        npoly = np
    )

    function rhs!(dudt, u, _, t)
        h = u.h
        dh = dudt.h

        # vertical advection no inflow at bottom 
        # and outflow at top
        Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0, 0.0)),
        )
        # only upward advection
        @. dh = -divf2c(Ic2f(h) * Geometry.Cartesian13Vector(0.0, cᵥ))

        # only horizontal advection
        hdiv = Operators.Divergence()
        @. dh += -hdiv(h * Geometry.Cartesian1Vector(cₕ))
        Spaces.weighted_dss!(dh)

        return dudt
    end

    # initial conditions
    h_init(x_init, z_init) = begin
        coords = Fields.coordinate_field(hv_center_space)
        h = map(coords) do coord
            exp(-((coord.x + x_init)^2 + (coord.z + z_init)^2) / (2 * 0.2^2))
        end

        return h
    end

    using OrdinaryDiffEq
    U = Fields.FieldVector(h = h_init(0.5, 0.5))
    push!(u_init, copy(U.h))

    Δt = 0.001
    t_end = 1.0
    prob = ODEProblem(rhs!, U, (0.0, t_end))
    sol = solve(prob, SSPRK33(), dt = Δt)

    h_end = h_init(0.5 - cₕ * t_end, 0.5 - cᵥ * t_end)
    push!(u_true, copy(h_end))
    push!(u_pred, copy(sol.u[end].h))

    # calculate error metrics
    #Δh[k] = 1 / nv
    Δh[k] = ClimaCore.column(hv_center_space.face_local_geometry.J,1,1,1)[1]
    err[k] = metric(sol.u[end].h, h_end)
end

# post-processing
using Plots
summary = Plots.plot(
    log10.(Δh), 
    log10.(err),
    xlabel = "log10 of minimum grid spacing",
    ylabel = "log10 of absolute error l2-norm",
    marker = true,
    label = "data",
    legend = :topleft
)

# plot
plot!(summary, log10.(Δh), log10.(err[end-1].*(Δh./Δh[end-1]).^(2)), label = "2nd-order", 
    title = "Gaussian vertical advection convergence test,\n vertical refinement, n_h = 10, n_p = 3,\n exp-stretched(0.75)"
)
Plots.png(summary, "err_vadvect_stretched.png")

# final
Plots.png(plot(u_pred[end]), "final.png")
Plots.png(plot(u_pred[1]), "inital.png")
