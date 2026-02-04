"""
    var_dependencies(name::FieldName, model)

Returns a tuple of `FieldName`s that the computed variable `name` depends on.

This function should be overloaded for each computed variable in the model.
The dependencies form the edges of the dependency graph.

Note: `name` is the first argument to enable dispatching on variable.

# Example

```julia
var_dependencies(::FieldName{(:c, :J)}, ::MyModel) = (@name(c.uₕ),)
var_dependencies(::FieldName{(:f, :u³)}, ::MyModel) = (@name(f.uₕ³), @name(f.u₃))
```
"""
function var_dependencies(name::FieldName{chain}, model) where {chain}
    error("""
        No `var_dependencies` method defined for $name.
        
        You must implement:
            VarManager.var_dependencies(::FieldName{$(repr(chain))}, ::$(typeof(model))) = (dependencies...)
        
        where `dependencies` is a tuple of FieldNames that $name depends on.
        """)
end

"""
    tendency_dependencies(name::FieldName, model)

Returns a tuple of `FieldName`s of **computed variables** that the tendency for `name` depends on.

This function should be overloaded for each tendency in the model.
The dependencies form the edges of the dependency graph.

**Important:** Only list computed variables here. Prognostic variables are always
available in the cache during tendency computation and don't need to be listed.
A warning will be issued if you list a prognostic variable unnecessarily.

Note: `name` is the first argument to enable dispatching on variable.

# Example

```julia
# Only list computed vars (f.ρ, f.u³), not prognostics
tendency_dependencies(::FieldName{(:c, :ρ)}, ::MyModel) = (@name(f.ρ), @name(f.u³))
```
"""
function tendency_dependencies(name::FieldName{chain}, model) where {chain}
    error("""
        No `tendency_dependencies` method defined for $name.
        
        You must implement:
            VarManager.tendency_dependencies(::FieldName{$(repr(chain))}, ::$(typeof(model))) = (dependencies...)
        
        where `dependencies` is a tuple of FieldNames that the tendency for $name depends on.
        """)
end

"""
    NodeType

Enum indicating the type of a node in the dependency graph.
Tendencies are not stored in `node_types` - they are tracked separately
via `tendency_names` and `tendency_edges`.
"""
@enum NodeType begin
    PROGNOSTIC_VAR  # Leaf node: comes from Y
    COMPUTED_VAR    # Intermediate node: computed via compute_var
end

"""
    DependencyGraph

A directed acyclic graph (DAG) representing the dependencies between
computed variables and tendencies.

The graph is constructed by recursively following dependencies from
tendency nodes back to prognostic variable nodes (leaves).

# Fields
- `edges::Dict{FieldName, Vector{FieldName}}`: Maps computed vars to their dependencies
- `node_types::Dict{FieldName, NodeType}`: Maps each node to its type (PROGNOSTIC_VAR or COMPUTED_VAR)
- `tendency_edges::Dict{FieldName, Vector{FieldName}}`: Maps tendency names to their dependencies
- `tendency_names::Vector{FieldName}`: Ordered list of tendency names (deterministic iteration order)
"""
struct DependencyGraph
    edges::Dict{FieldName, Vector{FieldName}}
    node_types::Dict{FieldName, NodeType}
    tendency_edges::Dict{FieldName, Vector{FieldName}}
    tendency_names::Vector{FieldName}
end

"""
    prognostic_names(graph::DependencyGraph)

Returns an iterator over the prognostic variable names in the graph.
"""
function prognostic_names(graph::DependencyGraph)
    filter(keys(graph.node_types)) do name
        graph.node_types[name] == PROGNOSTIC_VAR
    end
end

"""
    build_dependency_graph(tendency_names, prognostic_names, model)

Builds a dependency graph starting from the given tendency names and
recursively discovering all dependencies down to the prognostic variables.

# Arguments
- `tendency_names`: Collection of `FieldName`s for which tendencies are computed
- `prognostic_names`: Collection of `FieldName`s that are prognostic (leaf nodes)
- `model`: The model object passed to `var_dependencies` and `tendency_dependencies`

# Returns
A `DependencyGraph` containing all nodes and edges.
"""
function build_dependency_graph(tendency_names, prognostic_names, model)
    # Build up data structures locally
    edges = Dict{FieldName, Vector{FieldName}}()
    node_types = Dict{FieldName, NodeType}()
    tendency_edges = Dict{FieldName, Vector{FieldName}}()
    
    # Convert to set for efficient lookup during graph building
    prog_set = Set{FieldName}(prognostic_names)
    tend_vec = collect(FieldName, tendency_names)  # Vector for deterministic order
    
    # Mark prognostic variables as leaves (no dependencies)
    for name in prog_set
        node_types[name] = PROGNOSTIC_VAR
        edges[name] = FieldName[]
    end
    
    # Build graph by traversing from tendencies to leaves
    visited = Set{FieldName}()
    
    for tend_name in tend_vec
        _add_tendency_node!(edges, node_types, tendency_edges, tend_name, model, visited, prog_set)
    end
    
    # Construct immutable graph at the end (prog_set not stored, derive from node_types)
    graph = DependencyGraph(edges, node_types, tendency_edges, tend_vec)
    
    # Validate the graph before returning
    validate_dependencies(graph)
    
    return graph
end

"""
    _add_tendency_node!(edges, node_types, tendency_edges, name, model, visited, prognostic_names)

Adds a tendency node and recursively adds its dependencies.

Tendency dependencies are stored separately in `tendency_edges` because:
1. A tendency can depend on its same-named prognostic (Y.c.ρ -> Yₜ.c.ρ)
2. Tendencies are outputs, not intermediate nodes in the DAG
3. Prognostic variables remain leaves with no dependencies in `edges`
"""
function _add_tendency_node!(edges, node_types, tendency_edges, name, model, visited, prognostic_names)
    # Get and store dependencies for this tendency (separate from var edges)
    deps = tendency_dependencies(name, model)
    
    tendency_edges[name] = collect(deps)
    
    # Recurse on dependencies to build the var graph
    for dep in deps
        if !(dep in visited)
            _add_var_node!(edges, node_types, dep, model, visited, prognostic_names)
        end
    end
end

"""
    _add_var_node!(edges, node_types, name, model, visited, prognostic_names)

Adds a computed variable node and recursively adds its dependencies.
If the node is a prognostic variable, it's marked as a leaf.
"""
function _add_var_node!(edges, node_types, name, model, visited, prognostic_names)
    name in visited && return
    push!(visited, name)
    
    if name in prognostic_names
        node_types[name] = PROGNOSTIC_VAR
        edges[name] = FieldName[]
    else
        node_types[name] = COMPUTED_VAR
        deps = var_dependencies(name, model)
        edges[name] = collect(deps)
        
        for dep in deps
            _add_var_node!(edges, node_types, dep, model, visited, prognostic_names)
        end
    end
end

"""
    topological_sort(graph::DependencyGraph)

Performs a topological sort of the dependency graph using Kahn's algorithm.

Returns a vector of `FieldName`s in evaluation order (leaves first, then
computed variables, then tendencies).

# Throws
Throws an error if the graph contains a cycle.
"""
function topological_sort(graph::DependencyGraph)
    # Compute in-degrees
    in_degree = Dict{FieldName, Int}()
    for name in keys(graph.edges)
        in_degree[name] = 0
    end
    
    # Build dependents and count in-degrees
    dependents = Dict{FieldName, Vector{FieldName}}()
    for name in keys(graph.edges)
        dependents[name] = FieldName[]
    end
    
    for (name, deps) in graph.edges
        for dep in deps
            haskey(dependents, dep) && push!(dependents[dep], name)
            in_degree[name] += 1
        end
    end
    
    # Initialize queue with nodes that have no dependencies (in_degree == 0)
    queue = FieldName[]
    for (name, degree) in in_degree
        if degree == 0
            push!(queue, name)
        end
    end
    
    # Kahn's algorithm
    sorted = FieldName[]
    while !isempty(queue)
        node = popfirst!(queue)
        push!(sorted, node)
        
        for dependent in dependents[node]
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0
                push!(queue, dependent)
            end
        end
    end
    
    # Check for cycles
    if length(sorted) != length(graph.edges)
        remaining = setdiff(keys(graph.edges), sorted)
        error("Dependency graph contains a cycle involving: $(collect(remaining))")
    end
    
    return sorted
end

"""
    get_evaluation_order(graph::DependencyGraph)

Returns the evaluation order for the graph, excluding prognostic variables
(which are already available as inputs).

The returned vector contains `FieldName`s in the order they should be
computed: computed variables first (in dependency order), then tendencies.
"""
function get_evaluation_order(graph::DependencyGraph)
    sorted = topological_sort(graph)
    
    # Filter out prognostic variables (they don't need to be computed)
    # This gives us computed vars in dependency order
    computed_vars = filter(name -> graph.node_types[name] == COMPUTED_VAR, sorted)
    
    return computed_vars
end

"""
    is_tendency(graph::DependencyGraph, name::FieldName)

Returns true if `name` is a tendency (has an entry in tendency_edges).
"""
is_tendency(graph::DependencyGraph, name::FieldName) = haskey(graph.tendency_edges, name)

"""
    validate_dependencies(graph::DependencyGraph)

Validates that the dependency graph is well-formed:
- All var dependencies reference nodes that exist in the graph
- All tendency dependencies reference nodes that exist in the graph
- No self-references in computed vars
- No circular dependencies between computed variables

Returns `nothing` if valid, throws an error otherwise.
"""
function validate_dependencies(graph::DependencyGraph)
    # Detect cycles via topological sort (throws if cycle exists)
    topological_sort(graph)
    all_nodes = Set(keys(graph.edges))
    
    # Validate computed var edges
    for (name, deps) in graph.edges
        # Check for self-reference (only for computed vars, not prognostics)
        if graph.node_types[name] == COMPUTED_VAR && name in deps
            error("Self-reference detected: $name depends on itself")
        end
        
        # Check all dependencies exist
        for dep in deps
            !(dep in all_nodes) && error("Missing dependency: $name depends on $dep, but $dep is not in the graph")
        end
    end
    
    # Validate tendency edges
    for (name, deps) in graph.tendency_edges
        for dep in deps
            if !(dep in all_nodes)
                error("Missing tendency dependency: $name depends on $dep, but $dep is not in the graph")
            end
        end
        
        isempty(deps) && @warn "Tendency $name has no dependencies - is this intentional?"
    end
end
