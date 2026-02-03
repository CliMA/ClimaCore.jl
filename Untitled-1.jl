#=
VarManager Example for ClimaAtmos

This file demonstrates how to use ClimaCore's VarManager to define
the dependency graph for computing tendencies from prognostic variables.

The example implements a simplified atmospheric model with:
  - Prognostic variables: c.ρ, c.ρe_tot, c.ρq_tot, f.u₃
  - Computed variables: c.p, c.T, f.ρ, f.w
  - Tendencies: Yₜ.c.ρ, Yₜ.c.ρe_tot, Yₜ.c.ρq_tot, Yₜ.f.u₃

Run with:
    julia --project -e 'include("src/varmanager_example.jl")'
=#

using ClimaCore
using ClimaCore.VarManager
using ClimaCore.MatrixFields: @name, @Name, FieldName
using ClimaCore: Fields, Spaces, Domains, Meshes, Topologies, Geometry, Operators
import LazyBroadcast: @lazy
using ClimaComms
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C()
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector

struct AtmosModel end

# Face mass flux ρw (for vertical advection)
VarManager.var_dependencies(::@Name(f.ρw), model) =
    (@name(c.ρ), @name(f.u₃))

VarManager.tendency_dependencies(::@Name(c.ρ), model) =
    (@name(f.ρw),)

function VarManager.compute_var(::@Name(f.ρw), model, vars, t)
    return @lazy @. ᶠinterp(vars.c.ρ) * vars.f.u₃
end

function VarManager.compute_tendency(::@Name(c.ρ), model, vars, t)
    return @lazy @. -(ᶜdivᵥ(vars.f.ρw))
end

function run_example()
    println("VarManager Example for ClimaAtmos")
    
    # Create a simple 1D column space
    FT = Float64
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1000);  # 1km column
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 10)
    topology = Topologies.IntervalTopology(context, mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(topology)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
    
    # Create model
    model = AtmosModel()
    
    # Define prognostic and tendency names
    prog_names = (@name(c.ρ), @name(c.uₕ), @name(f.u₃))
    tend_names = (@name(c.ρ),)
    
    println("Building dependency graph...")
    graph = build_dependency_graph(tend_names, prog_names, model)
    @show graph
    print_graph(graph)
    
    Y = Fields.FieldVector(;
        c = (;
            ρ = Fields.Field(FT, center_space),      # kg/m³ (sea level density)
            uₕ = Fields.Field(C12{FT}, center_space),      # m/s (horizontal velocity)
        ),
        f = (;
            u₃ = Fields.Field(C3{FT}, face_space),         # m/s (vertical velocity)
        ),
    )

    # Non-uniform ρ or w so ∂(ρw)/∂z ≠ 0 (uniform fields → zero divergence)
    z_face = Fields.coordinate_field(face_space)
    Y.c.ρ .= 10
    @. Y.c.uₕ = C12(10.0, 0.0)
    @. Y.f.u₃ = C3(0.1 + 0.0001 * z_face.z)  # w increases with height → non-zero ∂(ρw)/∂z

    # Evaluate the graph
    println("\nEvaluating dependency graph...")
    t = 0.0
    Yₜ = evaluate_graph(Y, graph, model, t)
    
    return graph, Y, Yₜ
end

run_example()
