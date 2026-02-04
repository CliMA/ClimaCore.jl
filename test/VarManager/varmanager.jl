#=
julia --check-bounds=yes --project
using Revise; include(joinpath("test", "VarManager", "varmanager.jl"))
=#
using Test
using ClimaComms
using ClimaCore
using ClimaCore: Domains, Meshes, Topologies, Spaces, Fields, Geometry
using ClimaCore.MatrixFields: @name, @Name,FieldName
using ClimaCore.VarManager

# Test utilities to create simple spaces and fields
function create_test_space(FT = Float64)
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 10)
    topology = Topologies.IntervalTopology(context, mesh)
    space = Spaces.CenterFiniteDifferenceSpace(topology)
    return space
end

# Simple test model
struct TestModel
    multiplier::Float64
end

@testset "Caching Strategies" begin
    # Test EagerGlobalCaching
    eager = EagerGlobalCaching()
    @test VarManager.add_to_global_cache(eager, Base.broadcasted(+, 1, 2)) == true
    @test VarManager.add_to_global_cache(eager, 1.0) == false
    @test VarManager.add_to_global_cache(eager, [1, 2, 3]) == false
    
    # Test NoCaching
    nocache = NoCaching()
    @test VarManager.add_to_global_cache(nocache, Base.broadcasted(+, 1, 2)) == false
    @test VarManager.add_to_global_cache(nocache, 1.0) == false
end

model = TestModel(2.0)
prog_names = (@name(c.ρ), @name(f.u₃))
tend_names = (@name(c.ρ), @name(f.u₃))

VarManager.var_dependencies(::@Name(c.ρ_scaled), _) = (@name(c.ρ),)
VarManager.tendency_dependencies(::@Name(c.ρ), _) = (@name(c.ρ_scaled),)
VarManager.tendency_dependencies(::@Name(f.u₃), _) = (@name(f.u₃),)

graph = build_dependency_graph(tend_names, prog_names, model)

@testset "DependencyGraph" begin
    @test haskey(graph.edges, @name(c.ρ))
    @test haskey(graph.edges, @name(f.u₃))
    @test haskey(graph.edges, @name(c.ρ_scaled))
    
    @test graph.node_types[@name(c.ρ)] == VarManager.PROGNOSTIC_VAR
    @test graph.node_types[@name(f.u₃)] == VarManager.PROGNOSTIC_VAR
    @test graph.node_types[@name(c.ρ_scaled)] == VarManager.COMPUTED_VAR

    sorted = topological_sort(graph)
    @test sorted == FieldName[@name(f.u₃), @name(c.ρ), @name(c.ρ_scaled)]

    eval_order = VarManager.get_evaluation_order(graph)
    @test eval_order == FieldName[@name(c.ρ_scaled)]
end

@testset "_get_field_from_fieldvector" begin
    FT = Float64
    space = create_test_space(FT)
    
    # Create a simple FieldVector-like structure
    c_ρ = Fields.ones(space)
    f_space = Spaces.FaceFiniteDifferenceSpace(space)
    f_u₃ = Fields.ones(f_space)
    
    Y = Fields.FieldVector(; c = (; ρ = c_ρ), f = (; u₃ = f_u₃))
    
    # Test extracting fields from FieldVector using FieldName
    extracted = VarManager._get_field_from_fieldvector(Y, @name(c.ρ))
    @test extracted === Y.c.ρ
    
    extracted_f = VarManager._get_field_from_fieldvector(Y, @name(f.u₃))
    @test extracted_f === Y.f.u₃
end

VarManager.compute_var(::@Name(c.ρ_scaled), model, vars, t) =
    @. model.multiplier * vars.c.ρ
VarManager.compute_tendency(::@Name(c.ρ), model, vars, t) =
    @. vars.c.ρ_scaled
VarManager.compute_tendency(::@Name(f.u₃), model, vars, t) =
    @. vars.f.u₃

@testset "Full evaluate_graph integration test" begin
    FT = Float64
    space = create_test_space(FT)
    
    # Create state vector Y with initial values
    c_ρ = Fields.ones(space) .* 2.0   # ρ = 2.0
    f_u₃ = Fields.ones(space) .* 3.0

    Y = Fields.FieldVector(; c = (; ρ = c_ρ,), f = (; u₃ = f_u₃))
    
    # Evaluate the graph
    t = 0.0

    for strategy in [EagerGlobalCaching(), NoCaching()]
        Yₜ = evaluate_graph(Y, graph, model, t, strategy)
        @test all(parent(Yₜ.c.ρ) .== 4.0)
        @test all(parent(Yₜ.f.u₃) .== 3.0)
    end
end

@testset "Test cycle detection" begin
    VarManager.var_dependencies(::@Name(c.ρ2), _) = (@name(c.ρ3), @name(c.ρ))
    VarManager.var_dependencies(::@Name(c.ρ3), _) = (@name(c.ρ2),)
    VarManager.tendency_dependencies(::@Name(c.ρ4), _) = (@name(c.ρ3),)
    @test_throws ErrorException build_dependency_graph((@name(c.ρ4),), (@name(c.ρ),), model)
end

@testset "Cache reuse - no re-allocation of cache_fields" begin
    FT = Float64
    space = create_test_space(FT)
    c_ρ = Fields.ones(space) .* 2.0
    f_u₃ = Fields.ones(space) .* 3.0
    Y = Fields.FieldVector(; c = (; ρ = c_ρ), f = (; u₃ = f_u₃))

    cache = VarCache()
    evaluate_graph(Y, graph, model, 0.0, EagerGlobalCaching(); cache)
    field_ref = cache.cache_fields[@name(c.ρ_scaled)]

    evaluate_graph(Y, graph, model, 0.0, EagerGlobalCaching(); cache)
    @test cache.cache_fields[@name(c.ρ_scaled)] === field_ref
end

@testset "VarCacheView - property access matches index access" begin
    FT = Float64
    space = create_test_space(FT)
    c_ρ = Fields.ones(space) .* 2.0
    f_u₃ = Fields.ones(space) .* 3.0
    Y = Fields.FieldVector(; c = (; ρ = c_ρ), f = (; u₃ = f_u₃))

    cache = VarCache()
    evaluate_graph(Y, graph, model, 0.0, EagerGlobalCaching(); cache)

    @test cache.c.ρ === cache[@name(c.ρ)]
    @test cache.f.u₃ === cache[@name(f.u₃)]
end
