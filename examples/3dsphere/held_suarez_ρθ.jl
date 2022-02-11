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
    tmax = FT(60 * 60 * 24 * 1200),
    dt = FT(400.0),
    ode_algorithm = OrdinaryDiffEq.Rosenbrock23,
    jacobian_flags = (; ∂𝔼ₜ∂𝕄_mode = :exact, ∂𝕄ₜ∂ρ_mode = :exact),
    max_newton_iters = 2,
    save_every_n_steps = 2160, #10,
    additional_solver_kwargs = (; callback = FunctionCallingCallback(cb)), # e.g., reltol, abstol, etc.
)

initial_condition(local_geometry) = initial_condition_ρθ(local_geometry)
initial_condition_velocity(local_geometry) =
    initial_condition_velocity(local_geometry; is_balanced_flow = false)

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    held_suarez_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = false),
    (; saved_Ys = [copy(Y) for _ in 1:1000]),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρθ_remaining_tendency!(dY, Y, p, t; κ₄ = 2.0e17)
    held_suarez_forcing!(dY, Y, p, t)
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

function postprocessing(sol, path)
    @info "L₂ norm of ρθ at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρθ))"
    @info "L₂ norm of ρθ at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρθ))"

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:2
        Plots.plot(v, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)
end

function debug_nc(saved_Ys, nlat, nlon, path, c_local_geometry, f_local_geometry, Nq)
    ### First step: save the data on cg nodal points to ncfile
    # create the debug nc file
    datafile_cc = joinpath(path,"debug-rho_theta.nc")
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
    nc_theta = defVar(nc, "theta", Float64, cspace, ("time",))
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
        nc_theta[:,i] = Yc.ρθ ./ Yc.ρ
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

    datafile_rll = joinpath(path, "debug-rho_theta_rll.nc")
    apply_remap(datafile_rll, datafile_cc, weightfile, ["rho", "theta", "u", "v"])
end