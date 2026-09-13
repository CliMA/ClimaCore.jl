# Precompile workload for basic grids, spaces, operators, and field
# broadcasts. `Base.generating_output` does not exist on Julia 1.10, so the
# underlying C call is used directly (as PrecompileTools does). Developers
# iterating on ClimaCore can skip the workload with
# CLIMA_SKIP_PRECOMPILE_WORKLOAD=true.
#
# A data layout carries its vertical extent `Nv` and quadrature order in its
# type, and each distinct broadcast expression compiles a separate kernel, so
# the workload only removes first-use latency for code whose type parameters
# match a production run. It therefore uses a 63-level column (centers carry
# Nv = 63, faces Nv = 64) at GLL{4}, an AMIP resolution, with `NamedTuple`-
# eltype center and face fields standing in for a prognostic state: a scalar
# field and a scalar subfield view into a `NamedTuple` field compile to
# different kernels, so both are exercised. Adding resolutions or broadcast
# expressions widens the coverage and lengthens package load time.
if ccall(:jl_generating_output, Cint, ()) == 1 &&
   get(ENV, "CLIMA_SKIP_PRECOMPILE_WORKLOAD", "false") != "true"
    let
        # Centers carry Nv = 63 and faces Nv = 64, an AMIP resolution.
        nelems = 63
        for FT in (Float64, Float32)
            # The device is pinned rather than taken from `ClimaComms.device()`,
            # which reads `CLIMACOMMS_DEVICE` and throws on `CUDA` unless the
            # CUDA extension is loaded. This workload runs inside ClimaCore's
            # own precompilation, where CUDA.jl is a weak dependency and so is
            # never loaded, so a run script exporting `CLIMACOMMS_DEVICE=CUDA`
            # before `using ClimaCore` would otherwise fail to precompile.
            context =
                ClimaComms.SingletonCommsContext(ClimaComms.CPUMultiThreaded())
            vdomain = Domains.IntervalDomain(
                Geometry.ZPoint(FT(0)),
                Geometry.ZPoint(FT(1));
                boundary_names = (:bottom, :top),
            )
            vmesh = Meshes.IntervalMesh(vdomain, nelems = nelems)
            vtopology = Topologies.IntervalTopology(context, vmesh)
            vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

            hdomain = Domains.SphereDomain(FT(10))
            hmesh = Meshes.EquiangularCubedSphere(hdomain, 1)
            htopology = Topologies.Topology2D(context, hmesh)
            quad = Quadratures.GLL{4}()
            hspace = Spaces.SpectralElementSpace2D(htopology, quad)

            cspace = Spaces.ExtrudedFiniteDifferenceSpace(
                hspace,
                vspace,
                Hypsography.Flat(),
            )
            fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)

            z = Fields.coordinate_field(cspace).z
            ff = zeros(fspace)

            # A prognostic-like state: a `NamedTuple` field on centers (density,
            # energy, and horizontal covariant momentum, the momentum eltype
            # whose broadcasts and spectral operators dominate downstream
            # first-use latency) and one on faces (vertical covariant velocity).
            # The subfield views below are what production tendencies broadcast
            # over.
            cstate = @. (;
                ρ = one(z),
                ρe = 2 * one(z),
                uₕ = Geometry.Covariant12Vector(
                    Geometry.UVVector(one(z), 2 * one(z)),
                ),
            )
            fstate = @. (; w = Geometry.Covariant3Vector(zero(ff)))
            ρ = cstate.ρ
            ρe = cstate.ρe
            uₕ = cstate.uₕ
            @. ρe = ρ * ρe + 1

            # A standalone scalar field alongside the `ρ` subfield view: the two
            # drive different broadcast kernels.
            fc = copy(ρ)

            # Horizontal spectral operators: scalar and vector Laplacians (the
            # hyperdiffusion atoms), plus the weighted second-pass form, over
            # the state's subfield views.
            χ = Base.Broadcast.materialize(Operators.scalar_laplacian(ρ))
            χu = Base.Broadcast.materialize(Operators.vector_laplacian(uₕ))
            χw = Base.Broadcast.materialize(
                Operators.scalar_laplacian(χ; weight = ρ),
            )

            # Continuous DSS
            Spaces.weighted_dss!(ρ)

            # Vertical interpolation operators, without and with boundary
            # conditions
            I_c2f = Operators.InterpolateC2F()
            I_f2c = Operators.InterpolateF2C()
            @. ff = I_c2f(fc)
            @. fc = I_f2c(ff)
            I_c2f_extrap = Operators.InterpolateC2F(
                bottom = Operators.Extrapolate(),
                top = Operators.Extrapolate(),
            )
            @. ff = I_c2f_extrap(fc)

            # Common vertical stencil operators: gradient, flux divergence,
            # upwind advection, and mass-weighted interpolation
            G = Operators.GradientC2F(
                bottom = Operators.SetValue(FT(0)),
                top = Operators.SetValue(FT(0)),
            )
            gf = @. G(fc)
            wvec = Geometry.WVector
            D = Operators.DivergenceF2C(
                bottom = Operators.SetValue(wvec(FT(0))),
                top = Operators.SetValue(wvec(FT(0))),
            )
            w3 = @. wvec(one(ff))
            dc = @. D(w3)
            U = Operators.UpwindBiasedProductC2F()
            up = @. U(w3, fc)
            WI = Operators.WeightedInterpolateF2C()
            Jf = ones(fspace)
            wi = @. WI(Jf, ff)

            # FieldVector broadcasts over the `NamedTuple`-eltype state,
            # including an aliased in-place update in the vector-safe form a
            # Runge-Kutta stage uses.
            X = Fields.FieldVector(c = cstate, f = fstate)
            Y = similar(X)
            @. Y = X + FT(0.5) * X
            Y .-= X
        end
        # The topology and grid constructors memoize into the global object
        # cache; empty it so the workload's objects are not serialized into
        # the package image (the compiled code is cached either way).
        Utilities.Cache.clean_cache!()
    end
end
