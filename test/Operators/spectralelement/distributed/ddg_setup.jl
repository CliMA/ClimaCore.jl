#=
Shared setup for the distributed (MPI) discontinuous-Galerkin operator tests,
launched at a fixed rank count (see ddg2.jl / ddg3.jl):

    CLIMACOMMS_CONTEXT=MPI mpiexec -n 2 julia --project=... ddg2.jl

On a distributed cubed sphere, faces that straddle a rank boundary live in
`Topologies.ghost_faces` and are completed by the ghost-face exchange in
`add_numerical_flux_interior!` / `add_lifting_flux_interior!`. The checks are
per-local-node comparisons against element-local reference operators, so a
wrong (or skipped) ghost exchange shows up as a mismatch at the
partition-boundary elements without any cross-rank gather.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
include("../utils_dg.jl")

const context = ClimaComms.context()
const pid, nprocs = ClimaComms.init(context)

# Global (cross-rank) sum of a scalar field's parent array.
allsum(field) = ClimaComms.allreduce(context, sum(parent(field)), +)

function run_ddg_tests(::Type{FT}) where {FT}
    tol = FT == Float32 ? FT(1e-4) : FT(1e-10)
    space = dg_sphere_space(FT; context)
    topology = Spaces.topology(space)

    # This test is only meaningful when the partition actually cuts faces.
    nghost = length(Topologies.ghost_faces(topology))
    nghost_global = ClimaComms.allreduce(context, nghost, +)
    if ClimaComms.iamroot(context)
        @info "distributed DG" nprocs FT local_ghost_faces = nghost total = nghost_global
    end
    # the ghost-face path is exercised only when the partition cuts faces
    nprocs > 1 && @test nghost_global > 0

    coords = Fields.coordinate_field(space)
    lgeom = Fields.local_geometry_field(space)

    uv = @. Geometry.UVVector(
        cosd(coords.long),
        -sind(coords.long) * sind(coords.lat),
    )

    @testset "weak divergence + central flux == strong divergence [$FT, $nprocs ranks]" begin
        # `hdiv` (strong spectral divergence) is element-local and therefore
        # correct on every rank, including partition-boundary elements. The
        # weak form plus the central numerical flux reconstructs it only if the
        # flux couples the true neighbour across the rank boundary, so this
        # comparison at every local node tests the ghost exchange directly.
        hwdiv = Operators.Divergence{Operators.WeakForm}()
        hdiv = Operators.Divergence()
        F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), uv)
        q = ones(space)
        y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

        dy_mw = @. hwdiv(F) * (-(lgeom.WJ))
        Operators.add_numerical_flux_interior!(dg_central_flux, dy_mw, y)
        dy = @. dy_mw / lgeom.WJ

        dy_strong = @. -hdiv(F)
        err = maximum(abs, parent(dy) .- parent(dy_strong))
        scale = maximum(abs, parent(dy_strong))
        err_global = ClimaComms.allreduce(context, err, max)
        scale_global = ClimaComms.allreduce(context, scale, max)
        @test err_global < tol * scale_global
    end

    @testset "central numerical flux conserves globally [$FT, $nprocs ranks]" begin
        # Each interior face adds ∓sWJ·flux to its two nodes; across ranks the
        # ghost-face contributions are the missing halves, so the global node
        # sum of the residual is zero to round-off only if they are included.
        q = @. sind(coords.long) * cosd(coords.lat)^2
        y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
        r = similar(q)
        r .= 0
        Operators.add_numerical_flux_interior!(dg_central_flux, r, y)
        total = allsum(r)
        scale = ClimaComms.allreduce(context, sum(abs, parent(r)), +)
        cons_tol = FT == Float32 ? FT(1e-5) : FT(1e-10)
        @test abs(total) < cons_tol * scale
    end

    @testset "penalty numerical flux vanishes for continuous fields [$FT, $nprocs ranks]" begin
        # A single-valued field has no jump, so the jump-penalty numerical flux
        # must vanish at every node — including partition-boundary nodes, where
        # the jump is computed against the received ghost value.
        q = @. sind(coords.long) * cosd(coords.lat)^2
        r = similar(q)
        r .= 0
        Operators.add_numerical_flux_interior!(dg_jump_penalty, r, q)
        rn = @. r / lgeom.WJ
        err = maximum(abs, parent(rn))
        err_global = ClimaComms.allreduce(context, err, max)
        qmax = ClimaComms.allreduce(context, maximum(abs, parent(q)), max)
        @test err_global < tol * qmax
    end

    @testset "two-argument jump penalty vanishes [$FT, $nprocs ranks]" begin
        # Two same-typed Field arguments get separate ghost buffers (keyed by
        # argument position); if they aliased, the second fill would overwrite
        # the first field's halo and the jump would not vanish.
        two_field_jump(normal, (q⁻, s⁻), (q⁺, s⁺)) =
            ((q⁻ + s⁻) - (q⁺ + s⁺)) / 2
        q = @. sind(coords.long) * cosd(coords.lat)^2
        s = @. cosd(coords.long) * cosd(coords.lat)
        r = similar(q)
        r .= 0
        Operators.add_numerical_flux_interior!(two_field_jump, r, q, s)
        rn = @. r / lgeom.WJ
        err = maximum(abs, parent(rn))
        err_global = ClimaComms.allreduce(context, err, max)
        qmax = ClimaComms.allreduce(context, maximum(abs, parent(q)), max)
        @test err_global < tol * qmax
    end

    @testset "shared ghost exchange across face operators [$FT, $nprocs ranks]" begin
        # One exchange feeds the numerical flux and the lifting; both results
        # must be bitwise identical to each operator exchanging on its own.
        lift_q(normal, (y⁻,), (y⁺,)) = ((y⁺.q - y⁻.q) / 2) * normal
        q = @. sind(coords.long) * cosd(coords.lat)^2
        y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

        r_ref = similar(q)
        r_ref .= 0
        Operators.add_numerical_flux_interior!(dg_central_flux, r_ref, y)
        l_ref = similar(q, Geometry.UVVector{FT})
        fill!(parent(l_ref), 0)
        Operators.add_lifting_flux_interior!(lift_q, l_ref, y)

        ex = Operators.start_dg_ghost_exchange(y)
        r = similar(q)
        r .= 0
        Operators.add_numerical_flux_interior!(ex, dg_central_flux, r, y)
        l = similar(q, Geometry.UVVector{FT})
        fill!(parent(l), 0)
        Operators.add_lifting_flux_interior!(ex, lift_q, l, y)
        @test parent(r) == parent(r_ref)
        @test parent(l) == parent(l_ref)

        # A handle started on different arguments is rejected (detectable
        # only where ghost strips exist, i.e. on distributed ranks with
        # ghost faces).
        if nghost > 0
            ex2 = Operators.start_dg_ghost_exchange(y)
            r2 = similar(q)
            r2 .= 0
            @test_throws ErrorException Operators.add_numerical_flux_interior!(
                ex2,
                dg_jump_penalty,
                r2,
                q,
            )
            # Consume the started exchange so the next round on `y` finds
            # its buffers idle.
            Operators.add_numerical_flux_interior!(
                ex2,
                dg_central_flux,
                r2,
                y,
            )

            # Distinct fields of the same type share exchange buffers, so
            # starting a second round while one is in flight throws instead
            # of overwriting the in-flight send strips; after the first
            # round is consumed, a round on the other field works.
            q2 = @. cosd(coords.long) * cosd(coords.lat)
            ex3 = Operators.start_dg_ghost_exchange(q)
            @test_throws ErrorException Operators.start_dg_ghost_exchange(q2)
            r3 = similar(q)
            r3 .= 0
            Operators.add_numerical_flux_interior!(ex3, dg_jump_penalty, r3, q)
            ex4 = Operators.start_dg_ghost_exchange(q2)
            Operators.add_numerical_flux_interior!(ex4, dg_jump_penalty, r3, q2)
        end
    end

    @testset "ghost connectivity gather matches CPU ghost path [$FT, $nprocs ranks]" begin
        # Host emulation of the GPU ghost algorithm (exchange, stage
        # sWJ·fn(n̂⁻, ·⁻, ·⁺) per ghost face node, gather with the minus-side
        # map), compared against the CPU ghost-face loop. Validates
        # `Operators.dg_ghost_connectivity` and the staging/gather semantics
        # of the CUDA ghost kernels without a GPU.
        # Skipped on GPU devices: the emulation indexes host-style, and the
        # CPU ghost loop it compares against is host-only. The GPU runs of
        # this file exercise the CUDA ghost kernels through the operator
        # testsets instead.
        gconn = Operators.dg_ghost_connectivity(space)
        if nprocs == 1
            @test isnothing(gconn)
        elseif ClimaComms.device(space) isa ClimaComms.AbstractCPUDevice
            q = @. sind(coords.long) * cosd(coords.lat)^2
            y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
            r_ghost = similar(q)
            fill!(parent(r_ghost), 0)
            Operators._add_dg_ghost_faces!(
                Val(:numflux),
                dg_central_flux,
                r_ghost,
                (y,),
            )

            y_data = Fields.field_values(y)
            ex = Operators._dg_face_exchange(space, y_data, 1)
            Topologies.fill_face_send_buffer!(y_data, ex)
            ClimaComms.start(ex.graph_context)
            ClimaComms.finish(ex.graph_context)

            Nq = Quadratures.degrees_of_freedom(
                Spaces.quadrature_style(space),
            )
            staging = Array{FT}(undef, Nq, gconn.nfaces)
            for f in 1:gconn.nfaces, qq in 1:Nq
                elem⁻ = Int(gconn.faces[1, f])
                face⁻ = Int(gconn.faces[2, f])
                slot = Int(gconn.faces[3, f])
                reversed = gconn.faces[5, f] == 1
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, qq, false)
                q′ = reversed ? Nq - qq + 1 : qq
                sg = gconn.sgeom[1, qq, f]
                y⁻ = y_data[CartesianIndex(1, i⁻, j⁻, elem⁻)]
                y⁺ = ex.recv_data[CartesianIndex(1, q′, 1, slot)]
                staging[qq, f] =
                    sg.sWJ * dg_central_flux(sg.normal, (y⁻,), (y⁺,))
            end

            @test all(==(1), Array(gconn.contrib_side))
            r_em = similar(q)
            fill!(parent(r_em), 0)
            r_data = Fields.field_values(r_em)
            for n in 1:gconn.nbnodes
                I = CartesianIndex(
                    1,
                    Int(gconn.node_i[n]),
                    Int(gconn.node_j[n]),
                    Int(gconn.node_elem[n]),
                )
                acc = r_data[I]
                for c in
                    Int(gconn.node_offset[n]):(Int(gconn.node_offset[n + 1]) - 1)

                    acc -= staging[Int(gconn.contrib_q[c]), Int(gconn.contrib_face[c])]
                end
                r_data[I] = acc
            end

            err = maximum(abs, parent(r_em) .- parent(r_ghost))
            scale = maximum(abs, parent(r_ghost))
            err_global = ClimaComms.allreduce(context, err, max)
            scale_global = ClimaComms.allreduce(context, scale, max)
            @test err_global < tol * scale_global
        end
    end

    @testset "central lifting vanishes for continuous fields [$FT, $nprocs ranks]" begin
        # Exercises the symmetric-lifting ghost path (the `+` accumulation):
        # the central gradient lift is a jump, so it vanishes on a single-valued
        # field only if the partition-boundary jump uses the received ghost
        # value.
        q = @. sind(coords.long) * cosd(coords.lat)^2
        r = similar(q, Geometry.UVVector{FT})
        fill!(parent(r), 0)
        Operators.add_lifting_flux_interior!(
            Operators.central_gradient_lift,
            r,
            q,
        )
        rn = @. r / lgeom.WJ
        err = maximum(abs, parent(rn))
        err_global = ClimaComms.allreduce(context, err, max)
        qmax = ClimaComms.allreduce(context, maximum(abs, parent(q)), max)
        @test err_global < tol * qmax
    end
end
