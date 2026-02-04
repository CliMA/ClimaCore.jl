"""
    compute_var(name::FieldName, model, vars, t)

Computes the value for the variable `name` using the already-computed
variables in `vars`.

This function should be overloaded for each computed variable in the model.

Note: `name` is the first argument to enable dispatching on variable.

# Arguments
- `name`: The `FieldName` of the variable to compute
- `model`: The model object
- `vars`: A cache containing already-computed variables (supports `vars[@name(...)]`)
- `t`: Current simulation time

# Returns
Either a `Field` or a lazy broadcast expression (`Base.AbstractBroadcasted`).

# Example

```julia
compute_var(::FieldName{(:c, :J)}, ::MyModel, vars, t) =
    Fields.local_geometry_field(vars[@name(c.uₕ)]).J

compute_var(::FieldName{(:f, :u³)}, ::MyModel, vars, t) =
    @. lazy(vars[@name(f.uₕ³)] + CT3(vars[@name(f.u₃)]))
```
"""
function compute_var(name::FieldName{chain}, model, vars, t) where {chain}
    error("""
        No `compute_var` method defined for $name.
        
        You must implement:
            VarManager.compute_var(::FieldName{$(repr(chain))}, ::$(typeof(model)), vars, t) = ...
        
        The function should return either a Field or a lazy broadcast expression.
        """)
end

"""
    compute_tendency(name::FieldName, model, vars, t)

Computes the tendency for the prognostic variable `name` using the
already-computed variables in `vars`.

This function should be overloaded for each tendency in the model.

Note: `name` is the first argument to enable dispatching on variable.

# Arguments
- `name`: The `FieldName` of the prognostic variable whose tendency to compute
- `model`: The model object
- `vars`: A cache containing all computed variables (supports `vars[@name(...)]`)
- `t`: Current simulation time

# Returns
Either a `Field` or a lazy broadcast expression (`Base.AbstractBroadcasted`).

# Example

```julia
compute_tendency(::@Name(c.ρ), ::MyModel, vars, t) =
    @. lazy(-(ᶜdivᵥ(vars.f.ρ) * vars.f.u³))
```
"""
function compute_tendency(name::FieldName{chain}, model, vars, t) where {chain}
    error("""
        No `compute_tendency` method defined for $name.
        
        You must implement:
            VarManager.compute_tendency(::FieldName{$(repr(chain))}, ::$(typeof(model)), vars, t) = ...
        
        The function should return either a Field or a lazy broadcast expression.
        """)
end

#####
##### FieldVector access helpers
#####

"""
    initialize_from_prognostic!(cache::VarCache, Y::FieldVector, prognostic_names)

Initializes the cache with prognostic variables from the state vector Y.
"""
function initialize_from_prognostic!(cache::VarCache, Y::FieldVector, prognostic_names)
    for name in prognostic_names
        field = _get_field_from_fieldvector(Y, name)
        cache[name] = field
    end
    return cache
end

"""
    _get_field_from_fieldvector(Y::FieldVector, name::FieldName)

Extracts a field from a FieldVector using a FieldName.
Handles nested property access like @name(c.ρ) -> Y.c.ρ
"""
function _get_field_from_fieldvector(Y::FieldVector, ::FieldName{name_chain}) where {name_chain}
    result = Y
    for prop in name_chain
        result = getproperty(result, prop)
    end
    return result
end

"""
    _store_tendency!(Yₜ, name, result)

Stores a computed tendency into the tendency vector.
"""
function _store_tendency!(Yₜ::FieldVector, name::FieldName, result)
    dest = _get_field_from_fieldvector(Yₜ, name)
    if result isa Base.AbstractBroadcasted
        Base.Broadcast.materialize!(dest, result)
    elseif result isa LazyBroadcast.LazyBroadcasted
        Base.Broadcast.materialize!(dest, result.value)
    else
        dest .= result
    end
end

#####
##### Graph evaluation
#####

"""
    evaluate_graph(
        Y::FieldVector,
        graph::DependencyGraph,
        model,
        t,
        caching_strategy::VarCachingStrategy = EagerGlobalCaching();
        cache::VarCache = VarCache()
    )

Evaluates the dependency graph and returns a newly allocated tendency FieldVector.

# Arguments
- `Y`: The current state vector (prognostic variables)
- `graph`: The dependency graph
- `model`: The model object
- `t`: Current simulation time
- `caching_strategy`: Strategy for caching intermediate results
- `cache`: Optional pre-existing cache to reuse

# Returns
A new FieldVector containing the computed tendencies.
"""
function evaluate_graph(
    Y::FieldVector,
    graph::DependencyGraph,
    model,
    t,
    caching_strategy::VarCachingStrategy = EagerGlobalCaching();
    cache = VarCache()
)
    name_tree = FieldNameTree(Y)
    keys = FieldVectorKeys(tuple(graph.tendency_names...), name_tree)
    entries = map(name -> zero(_get_field_from_fieldvector(Y, name)), keys)
    Yₜ = concrete_field_vector(FieldNameDict(keys, entries))
    return evaluate_graph!(Yₜ, Y, graph, model, t, caching_strategy; cache)
end

"""
    evaluate_graph!(
        Yₜ::FieldVector,
        Y::FieldVector,
        graph::DependencyGraph,
        model,
        t,
        caching_strategy::VarCachingStrategy = EagerGlobalCaching();
        cache::VarCache = VarCache()
    )

Evaluates the dependency graph to compute all tendencies.

# Arguments
- `Yₜ`: The tendency vector to populate (modified in place)
- `Y`: The current state vector (prognostic variables)
- `graph`: The dependency graph
- `model`: The model object
- `t`: Current simulation time
- `caching_strategy`: Strategy for caching intermediate results
- `cache`: Optional pre-existing cache to reuse
"""
function evaluate_graph!(
    Yₜ::FieldVector,
    Y::FieldVector,
    graph::DependencyGraph,
    model,
    t,
    caching_strategy::VarCachingStrategy = EagerGlobalCaching();
    cache::VarCache = VarCache()
)
    # Get evaluation order for computed vars (excludes prognostic variables)
    eval_order = get_evaluation_order(graph)
    
    # Initialize cache with prognostic variables from Y
    initialize_from_prognostic!(cache, Y, prognostic_names(graph))
    
    # First, evaluate all computed variables in dependency order
    for name in eval_order
        result = compute_var(name, model, cache, t)
        _store_result!(cache, name, result, caching_strategy)
    end
    
    # Then, compute all tendencies (they depend on computed vars and prognostics)
    for name in graph.tendency_names
        deps = graph.tendency_edges[name]
        result = _safe_compute_tendency(name, model, cache, t, deps)
        _store_tendency!(Yₜ, name, result)
    end
    
    return Yₜ
end
