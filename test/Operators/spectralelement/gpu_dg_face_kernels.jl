#=
Single-process CPU-vs-GPU equivalence for the discontinuous-Galerkin (DG) face
kernels. For each supported space this builds the same DG space twice — once on
a CPU context and once on the active device (CUDA under CLIMACOMMS_DEVICE=CUDA)
— from the same mesh, feeds byte-identical inputs, applies each face operator on
both, and asserts the device result matches the CPU result to round-off.

This exercises the CUDA face kernels in `ext/cuda/operators_dg.jl`
(`dg_face_flux_kernel!`, `dg_face_gather_kernel!`, `dg_boundary_face_flux_kernel!`,
`dg_fddg_volume_kernel!`) against their CPU counterparts in
`src/Operators/numericalflux.jl`, closing the "CPU-vs-GPU equivalence for every
face kernel" gap. The cross-rank ghost kernel is covered separately by the
distributed tests (`distributed/ddg_setup.jl`); the extruded-1D column face
operators are CPU-only (no CUDA kernel) and are not exercised here.

Marked `:gpu_only` in the harness, so on CPU-only runs it is skipped.
=#
using Test
using LinearAlgebra
using IntervalSets
import Random
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Quadratures, Spaces, Topologies

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

include("utils_dg.jl")  # dg_sphere_space, dg_central_flux, dg_jump_penalty

# Inf-norm relative comparison, matching `test/Operators/gpu_edge_cases.jl`. The
# DG gather sums a node's face contributions in a device-dependent order, so
# results agree to round-off rather than bitwise.
device_matches_cpu(dev, cpu; rtol) = isapprox(
    Array(parent(dev)),
    Array(parent(cpu));
    rtol = rtol,
    norm = x -> norm(x, Inf),
)

# --- context-parametrized DG space builders ------------------------------------
# Each takes a `context`; passing a CPU context and a device context that wrap
# the same mesh yields the identical partition/geometry on each device.

function dg_extruded_sphere_space(
    ::Type{FT};
    context,
    radius = FT(6.371e6),
    zmax = FT(30e3),
    helem = 4,
    zelem = 4,
    Nq = 4,
) where {FT}
    device = ClimaComms.device(context)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain; nelems = zelem)
    vspace =
        Spaces.CenterFiniteDifferenceSpace(Topologies.IntervalTopology(device, vmesh))
    hspace = dg_sphere_space(FT; radius, helem, Nq, context)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
end

function dg_channel_space(
    ::Type{FT};
    context,
    Lx = FT(2π),
    Ly = FT(2),
    nelem = 4,
    Nq = 4,
    x1periodic = false,
) where {FT}
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(zero(Lx)) .. Geometry.XPoint{FT}(Lx),
        Geometry.YPoint{FT}(-Ly / 2) .. Geometry.YPoint{FT}(Ly / 2);
        x1periodic,
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discretization = Spaces.DG(),
    )
end

function dg_extruded_channel_space(
    ::Type{FT};
    context,
    zmax = FT(3),
    zelem = 5,
    kwargs...,
) where {FT}
    device = ClimaComms.device(context)
    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    vspace = Spaces.CenterFiniteDifferenceSpace(
        Topologies.IntervalTopology(device, Meshes.IntervalMesh(zdomain; nelems = zelem)),
    )
    hspace = dg_channel_space(FT; context, kwargs...)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
end

# Overwrite `field`'s data with a deterministic analytic-plus-perturbation host
# array, and return that array so a partner field on another device can be
# seeded with the identical bytes. Perturbing makes the interface state genuinely
# discontinuous so the flux kernels do nontrivial work.
function perturbed_host_array(field, FT; seed)
    h = Array(parent(field))
    Random.seed!(seed)
    scale = maximum(abs, h)
    scale = scale == 0 ? one(FT) : scale
    h .+= FT(0.1) .* scale .* (rand(FT, size(h)) .- FT(0.5))
    copyto!(parent(field), h)
    return h
end

# Build the same `Field` on the CPU and device spaces, seeded with identical
# perturbed bytes.
function paired_field(build, cpu_space, dev_space, FT; seed)
    f_cpu = build(cpu_space)
    f_dev = build(dev_space)
    h = perturbed_host_array(f_cpu, FT; seed)
    copyto!(parent(f_dev), h)
    return f_cpu, f_dev
end

state(space, cf) =
    let coords = Fields.coordinate_field(space)
        (q, uv) = cf(coords)
        map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
    end

# Test-local two-point volume fluxes (see `unit_extruded_sphere_dg.jl`).
central_2pt(nvec_a, nvec_b, a, b) =
    ((a.q * a.uv)' * nvec_a + (b.q * b.uv)' * nvec_b) / 2
kg_2pt(nvec_a, nvec_b, a, b) =
    ((a.q + b.q) / 2) * ((a.uv' * nvec_a + b.uv' * nvec_b) / 2)

# Flux tensor `T = v ⊗ m` on a sphere space, with `m` a constant global-Cartesian
# momentum expressed in the local frame (so its momentum/second axis is the full
# 3D `UVWAxis`, as `cartesian_tensor_divergence` requires). The typed `mcart`
# local keeps the `LocalVector` broadcast inferrable; the singleton
# `global_geometry` broadcasts as a scalar on both devices.
function sphere_tensor_flux(space, ::Type{FT}) where {FT}
    coords = Fields.coordinate_field(space)
    gg = Spaces.global_geometry(space)
    mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
    local_momentum(geom, coord) = Geometry.LocalVector(mcart, geom, coord)
    mloc = local_momentum.(gg, coords)
    v = @. Geometry.UVVector(
        sind(coords.long) * cosd(coords.lat),
        cosd(coords.long),
    )
    return @. v ⊗ mloc
end

@testset "DG face kernels: CPU vs GPU equivalence" begin
    cpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    dev_context = ClimaComms.SingletonCommsContext(ClimaComms.device())
    on_gpu = ClimaComms.device() isa ClimaComms.CUDADevice
    if !on_gpu
        @info "gpu_dg_face_kernels: no CUDA device; comparing CPU against CPU (smoke test)"
    end

    TU.@test_precisions FT begin
        rtol = FT == Float32 ? FT(1e-4) : 10 * eps(FT)

        # Interface flux on the cubed sphere (pure 2D) and hydrostatic extruded
        # sphere (extruded 2D): the two space families the CUDA face kernel
        # supports.
        sphere_cf(coords) = (
            (@. sind(coords.long) * cosd(coords.lat)^2),
            (@. Geometry.UVVector(
                cosd(coords.long),
                -sind(coords.long) * sind(coords.lat),
            )),
        )

        space_pairs = (
            (
                "pure-2D sphere",
                dg_sphere_space(FT; context = cpu_context),
                dg_sphere_space(FT; context = dev_context),
                sphere_cf,
            ),
            (
                "extruded-2D sphere",
                dg_extruded_sphere_space(FT; context = cpu_context),
                dg_extruded_sphere_space(FT; context = dev_context),
                sphere_cf,
            ),
        )

        for (name, cpu_space, dev_space, cf) in space_pairs
            @testset "interior numerical flux [$name, $FT]" begin
                y_cpu = state(cpu_space, cf)
                y_dev = state(dev_space, cf)
                copyto!(parent(y_dev), perturbed_host_array(y_cpu, FT; seed = 11))

                for fn in (dg_central_flux, dg_jump_penalty)
                    r_cpu = similar(y_cpu.q)
                    r_dev = similar(y_dev.q)
                    fill!(parent(r_cpu), 0)
                    fill!(parent(r_dev), 0)
                    # dg_jump_penalty takes the scalar q; dg_central_flux the state.
                    arg_cpu = fn === dg_jump_penalty ? y_cpu.q : y_cpu
                    arg_dev = fn === dg_jump_penalty ? y_dev.q : y_dev
                    Operators.add_numerical_flux_interior!(fn, r_cpu, arg_cpu)
                    Operators.add_numerical_flux_interior!(fn, r_dev, arg_dev)
                    @test device_matches_cpu(r_dev, r_cpu; rtol)
                end
            end

            @testset "lifting flux [$name, $FT]" begin
                q_cpu, q_dev = paired_field(
                    s -> cf(Fields.coordinate_field(s))[1],
                    cpu_space,
                    dev_space,
                    FT;
                    seed = 12,
                )
                r_cpu = similar(q_cpu, Geometry.UVVector{FT})
                r_dev = similar(q_dev, Geometry.UVVector{FT})
                fill!(parent(r_cpu), 0)
                fill!(parent(r_dev), 0)
                Operators.add_lifting_flux_interior!(
                    Operators.central_gradient_lift,
                    r_cpu,
                    q_cpu,
                )
                Operators.add_lifting_flux_interior!(
                    Operators.central_gradient_lift,
                    r_dev,
                    q_dev,
                )
                @test device_matches_cpu(r_dev, r_cpu; rtol)
            end

            @testset "flux-differencing volume [$name, $FT]" begin
                y_cpu = state(cpu_space, cf)
                y_dev = state(dev_space, cf)
                copyto!(parent(y_dev), perturbed_host_array(y_cpu, FT; seed = 13))
                for fn2pt in (central_2pt, kg_2pt)
                    r_cpu = similar(y_cpu.q)
                    r_dev = similar(y_dev.q)
                    fill!(parent(r_cpu), 0)
                    fill!(parent(r_dev), 0)
                    Operators.add_flux_differencing_divergence!(fn2pt, r_cpu, y_cpu)
                    Operators.add_flux_differencing_divergence!(fn2pt, r_dev, y_dev)
                    @test device_matches_cpu(r_dev, r_cpu; rtol)
                end
            end
        end

        # Boundary numerical flux needs a bounded space (the sphere has no
        # boundary faces): a fully bounded box (2D) and its extrusion.
        bflux(normal, (q⁻,)) = q⁻ * (Geometry.UVVector(FT(0), FT(1))' * normal)
        boundary_pairs = (
            (
                "bounded box",
                dg_channel_space(FT; context = cpu_context),
                dg_channel_space(FT; context = dev_context),
            ),
            (
                "extruded bounded box",
                dg_extruded_channel_space(FT; context = cpu_context),
                dg_extruded_channel_space(FT; context = dev_context),
            ),
        )
        for (name, cpu_space, dev_space) in boundary_pairs
            @testset "boundary numerical flux [$name, $FT]" begin
                q_cpu, q_dev = paired_field(
                    s -> let coords = Fields.coordinate_field(s)
                        @. sin(coords.x) + FT(2)
                    end,
                    cpu_space,
                    dev_space,
                    FT;
                    seed = 14,
                )
                r_cpu = similar(q_cpu)
                r_dev = similar(q_dev)
                fill!(parent(r_cpu), 0)
                fill!(parent(r_dev), 0)
                Operators.add_numerical_flux_boundary!(bflux, r_cpu, q_cpu)
                Operators.add_numerical_flux_boundary!(bflux, r_dev, q_dev)
                @test device_matches_cpu(r_dev, r_cpu; rtol)
            end
        end

        # Full weak-form Cartesian tensor divergence on the cubed sphere:
        # momentum-axis rotation → weak `Divergence` → interior numerical flux
        # → inverse rotation, the operator behind flux-form momentum on a curved
        # space. Chains the Geometry rotation broadcasts with the DG volume and
        # face kernels, so it covers the whole device path in one comparison.
        @testset "cartesian tensor divergence [pure-2D sphere, $FT]" begin
            cpu_space = dg_sphere_space(FT; context = cpu_context)
            dev_space = dg_sphere_space(FT; context = dev_context)
            T_cpu = sphere_tensor_flux(cpu_space, FT)
            T_dev = sphere_tensor_flux(dev_space, FT)
            # identical (perturbed) bytes on both devices
            copyto!(parent(T_dev), perturbed_host_array(T_cpu, FT; seed = 15))
            central = Operators.CentralNumericalFlux(identity)
            d_cpu = Operators.cartesian_tensor_divergence(T_cpu, central)
            d_dev = Operators.cartesian_tensor_divergence(T_dev, central)
            @test device_matches_cpu(d_dev, d_cpu; rtol)
        end
    end
end
