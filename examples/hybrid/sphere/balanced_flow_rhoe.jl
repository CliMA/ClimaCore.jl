using Test
using ClimaCorePlots, Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
helems, zelems, npoly = 4, 10, 4

horzdomain = Domains.SphereDomain(R)
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(FT(0)),
    Geometry.ZPoint{FT}(FT(30e3));
    boundary_tags = (:bottom, :top),
)
horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helems)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelems)
quad = Spaces.Quadratures.GLL{npoly + 1}()

Nv = Meshes.nelements(vertmesh)
Nf_center, Nf_face = 4, 1
vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

if usempi
    horztopology = Topologies.DistributedTopology2D(horzmesh, Context)
    comms_ctx =
        Spaces.setup_comms(Context, horztopology, quad, Nv + 1, Nf_center)
    global_topology = Topologies.Topology2D(horzmesh)
    global_horz_space = Spaces.SpectralElementSpace2D(global_topology, quad)
    global_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
        global_horz_space,
        vert_center_space,
    )
    global_face_space =
        Spaces.FaceExtrudedFiniteDifferenceSpace(global_center_space)

else
    horztopology = Topologies.Topology2D(horzmesh)
    comms_ctx = nothing
end

horzspace = Spaces.SpectralElementSpace2D(horztopology, quad, comms_ctx)

hv_center_space =
    Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

t_end = FT(60 * 60)
dt = FT(5)
dt_save_to_sol = FT(50)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.SSPRK33

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_tendency!(Yₜ, Y, p, t, comms_ctx = nothing)
    hyperdiffusion_tendency!(Yₜ, Y, p, t, comms_ctx)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρe); is_balanced_flow = true)

function postprocessing(sol, p, output_dir, usempi = false)
    sol_global = []
    if usempi
        for sol_step in sol.u
            sol_step_values_center_global =
                DataLayouts.gather(comms_ctx, Fields.field_values(sol_step.c))
            sol_step_values_face_global =
                DataLayouts.gather(comms_ctx, Fields.field_values(sol_step.f))
            if ClimaComms.iamroot(Context)
                sol_step_global = Fields.FieldVector(
                    c = Fields.Field(
                        sol_step_values_center_global,
                        global_center_space,
                    ),
                    f = Fields.Field(
                        sol_step_values_face_global,
                        global_face_space,
                    ),
                )
                push!(sol_global, sol_step_global)
            end
        end
        if ClimaComms.iamroot(Context)
        end
    else
        sol_global = sol.u
    end

    if !usempi || (usempi && ClimaComms.iamroot(Context))
        @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol_global[1].c.ρe))"
        @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol_global[end].c.ρe))"

        ᶜu_end = Geometry.UVVector.(sol_global[end].c.uₕ).components.data.:1
        Plots.png(
            Plots.plot(ᶜu_end, level = 3),
            joinpath(output_dir, "u_end.png"),
        )

        ᶜw_end = Geometry.WVector.(sol_global[end].f.w).components.data.:1
        Plots.png(
            Plots.plot(ᶜw_end, level = 3 + half, clim = (-4, 4)),
            joinpath(output_dir, "w_end.png"),
        )

        ᶜu_start = Geometry.UVVector.(sol_global[1].c.uₕ).components.data.:1
        Plots.png(
            Plots.plot(ᶜu_end .- ᶜu_start, level = 3, clim = (-1, 1)),
            joinpath(output_dir, "Δu_end.png"),
        )

        @test sol_global[end].c.ρ ≈ sol_global[1].c.ρ rtol = 5e-2
        @test sol_global[end].c.ρe ≈ sol_global[1].c.ρe rtol = 5e-2
        @test sol_global[end].c.uₕ ≈ sol_global[1].c.uₕ rtol = 5e-2
    end
end
