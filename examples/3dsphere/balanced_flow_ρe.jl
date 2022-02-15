using Test
using ClimaCorePlots, Plots
using DiffEqCallbacks

include("baroclinic_wave_utilities.jl")

function cb(Y, t, integrator)
    saved_Ys = integrator.p.saved_Ys

    N_saved = length(saved_Ys)
    for i in 1:(N_saved - 1)
        saved_Ys[i] .= saved_Ys[i + 1]
    end
    saved_Ys[N_saved] .= Y
end

driver_values(FT) = (;
    zmax = FT(30.0e3),
    velem = 10,
    helem = 4,
    npoly = 4,
    tmax = FT(60 * 60 * 24 * 870),
    dt = FT(500.0), #FT(5.0),
    ode_algorithm = OrdinaryDiffEq.Rosenbrock23, #OrdinaryDiffEq.SSPRK33,
    jacobian_flags = (; ∂𝔼ₜ∂𝕄_mode = :constant_P, ∂𝕄ₜ∂ρ_mode = :exact),
    max_newton_iters = 2,
    save_every_n_steps = 1728,
    additional_solver_kwargs = (; callback = FunctionCallingCallback(cb)), # e.g., reltol, abstol, etc.
)

initial_condition(local_geometry) =
    initial_condition_ρe(local_geometry; is_balanced_flow = true)
initial_condition_velocity(local_geometry) =
    initial_condition_velocity(local_geometry; is_balanced_flow = true)

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = false),
    (; saved_Ys = [copy(Y) for _ in 1:100]),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρe_remaining_tendency!(dY, Y, p, t; κ₄ = 2.0e17)
    final_adjustments!(
        dY,
        Y,
        p,
        t;
        use_flux_correction = false,
        use_rayleigh_sponge = false,
    )
    return dY
end

# function postprocessing(sol, path)
#     @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
#     @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

#     anim = Plots.@animate for Y in sol.u
#         v = Geometry.UVVector.(Y.uₕ).components.data.:2
#         Plots.plot(v, level = 3, clim = (-3, 3))
#     end
#     Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)

#     u_end = Geometry.UVVector.(sol.u[end].uₕ).components.data.:1
#     Plots.png(Plots.plot(u_end, level = 3), joinpath(path, "u_end.png"))

#     w_end = Geometry.WVector.(sol.u[end].w).components.data.:1
#     Plots.png(
#         Plots.plot(w_end, level = 3 + half, clim = (-4, 4)),
#         joinpath(path, "w_end.png"),
#     )

#     Δu_end = Geometry.UVVector.(sol.u[end].uₕ .- sol.u[1].uₕ).components.data.:1
#     Plots.png(
#         Plots.plot(Δu_end, level = 3, clim = (-1, 1)),
#         joinpath(path, "Δu_end.png"),
#     )

#     @test sol.u[end].Yc.ρ ≈ sol.u[1].Yc.ρ rtol = 5e-2
#     @test sol.u[end].Yc.ρe ≈ sol.u[1].Yc.ρe rtol = 5e-2
#     @test sol.u[end].uₕ ≈ sol.u[1].uₕ rtol = 5e-2
# end

function postprocessing(sol, path)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:2
        Plots.plot(v, level = 3, clim = (-3, 3))
    end
    Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:1
        Plots.plot(v, level = 3, clim = (-25, 25))
    end
    Plots.mp4(anim, joinpath(path, "u.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        cuₕ = Y.uₕ
        fw = Y.w
        cw = If2c.(fw)
        cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)
        normuvw = norm(cuvw)
        ρ = Y.Yc.ρ
        e_tot = @. Y.Yc.ρe / Y.Yc.ρ
        Φ = p.Φ
        I = @. e_tot - Φ - normuvw^2 / 2
        T = @. I / cv_d + T_tri
        Plots.plot(T, level = 3, clim = (225, 255))
    end
    Plots.mp4(anim, joinpath(path, "T.mp4"), fps = 5)
end

function debug_nc(saved_Ys, nlat, nlon, path, c_local_geometry, f_local_geometry, Nq)
    ### First step: save the data on cg nodal points to ncfile
    # create the debug nc file
    datafile_cc = joinpath(path,"debug-rho_etot.nc")
    nc = NCDataset(datafile_cc, "c")

    # get spaces from local geometries
    cspace = Fields.axes(c_local_geometry)
    fspace = Fields.axes(f_local_geometry)

    # define space coords in nc file
    def_space_coord(nc, cspace)

    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)

    # vars
    nc_rho = defVar(nc, "rho", Float64, cspace, ("time",))
    nc_etot = defVar(nc, "etot", Float64, cspace, ("time",))
    nc_u = defVar(nc, "u", Float64, cspace, ("time",))
    nc_v = defVar(nc, "v", Float64, cspace, ("time",))

    # loop over time
    for i = 1:length(saved_Ys)
        nc_time[i] = i

        # scalar fields
        Yc = saved_Ys[i].Yc
        # covariant vector -> UVVector
        uv = Geometry.UVVector.(saved_Ys[i].uₕ)

        # save data to nc
        nc_rho[:,i] = Yc.ρ
        nc_etot[:,i] = Yc.ρe ./ Yc.ρ
        nc_u[:,i] = uv.components.data.:1
        nc_v[:,i] = uv.components.data.:2
    end
    close(nc)

    ### Second step: use Tempest to remap onto regular lon/lat grid
    # get horizontal space from center space
    hspace = cspace.horizontal_space

    # write out our cubed sphere mesh
    meshfile_cc = joinpath(path,"mesh_cubedsphere.g")
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = joinpath(path,"mesh_rll.g")
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = joinpath(path, "mesh_overlap.g")
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = joinpath(path,"remap_weights.nc")
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    datafile_rll = joinpath(path, "debug-rho_etot_rll.nc")
    apply_remap(datafile_rll, datafile_cc, weightfile, ["rho", "etot", "u", "v"])
end