"""
    ComputationError

An error type that provides user-friendly error messages when variable
or tendency computation fails.

# Fields
- `name::FieldName`: The variable or tendency that failed to compute
- `is_tendency::Bool`: Whether this was a tendency (true) or variable (false)
- `original_error::Exception`: The underlying error that was caught
- `dependency_chain::Vector{FieldName}`: The chain of dependencies leading to this computation
"""
struct ComputationError <: Exception
    name::FieldName
    is_tendency::Bool
    original_error::Exception
    dependency_chain::Vector{FieldName}
end

function Base.showerror(io::IO, e::ComputationError)
    type_str = e.is_tendency ? "tendency" : "variable"
    println(io, "ComputationError: Failed to compute $type_str $(e.name)")
    println(io)
    
    if !isempty(e.dependency_chain)
        println(io, "Dependency chain:")
        for (i, dep) in enumerate(e.dependency_chain)
            indent = "  " ^ i
            println(io, "$(indent)└─ $dep")
        end
        println(io)
    end
    
    println(io, "Original error:")
    # Filter out kernel launch internals from the error message
    _show_filtered_error(io, e.original_error)
end

"""
    _show_filtered_error(io, error)

Shows the original error, filtering out low-level kernel launch details
that are not useful for debugging user code.
"""
function _show_filtered_error(io::IO, error::Exception)
    # Capture the error message
    error_str = sprint(showerror, error)
    
    # Filter patterns that indicate internal kernel machinery
    filter_patterns = [
        r"at .*broadcast\.jl:\d+",
        r"at .*cuda/.*\.jl:\d+",
        r"at .*CUDA/.*\.jl:\d+",
        r"Stacktrace:.*$"s,  # Remove full stacktrace
    ]
    
    filtered = error_str
    for pattern in filter_patterns
        filtered = replace(filtered, pattern => "")
    end
    
    # Clean up multiple newlines
    filtered = replace(filtered, r"\n{3,}" => "\n\n")
    
    print(io, "  ", strip(filtered))
end

"""
    _safe_compute_var(name, model, cache, t, dependencies)

Safely computes a variable, wrapping any errors in a ComputationError
that includes the dependency chain for debugging.
"""
function _safe_compute_var(name::FieldName, model, cache::VarCache, t, dependencies::Vector{FieldName})
    try
        return compute_var(name, model, cache, t)
    catch e
        if e isa ComputationError
            # Propagate with extended chain
            throw(ComputationError(name, false, e.original_error, 
                                   vcat([e.name], e.dependency_chain)))
        else
            throw(ComputationError(name, false, e, collect(dependencies)))
        end
    end
end

"""
    _safe_compute_tendency(name, model, cache, t, dependencies)

Safely computes a tendency, wrapping any errors in a ComputationError
that includes the dependency chain for debugging.
"""
function _safe_compute_tendency(name::FieldName, model, cache::VarCache, t, dependencies::Vector{FieldName})
    try
        return compute_tendency(name, model, cache, t)
    catch e
        if e isa ComputationError
            # Propagate with extended chain
            throw(ComputationError(name, true, e.original_error,
                                   vcat([e.name], e.dependency_chain)))
        else
            throw(ComputationError(name, true, e, collect(dependencies)))
        end
    end
end

"""
    print_graph(io::IO, graph::DependencyGraph)
    print_graph(graph::DependencyGraph)

Prints an ASCII visualization of the dependency graph showing the flow
from prognostic variables through computed variables to tendencies.
"""
function print_graph(io::IO, graph::DependencyGraph)
    println(io, "Dependency Graph:")
    println(io)
    
    # For each tendency, print the tree of dependencies
    for tend_name in graph.tendency_names
        println(io, "Yₜ.$tend_name")
        
        # Get tendency dependencies
        tend_deps = graph.tendency_edges[tend_name]
        _print_deps(io, graph, tend_deps, "")
    end
end

function _print_deps(io::IO, graph::DependencyGraph, deps::Vector{FieldName}, prefix::String)
    for (j, dep) in enumerate(deps)
        is_last = (j == length(deps))
        connector = is_last ? "└── " : "├── "
        child_prefix = prefix * (is_last ? "    " : "│   ")
        
        node_type = get(graph.node_types, dep, nothing)
        
        if node_type == PROGNOSTIC_VAR
            # Leaf node - show as input from Y
            println(io, "$(prefix)$(connector)Y.$dep")
        elseif node_type == COMPUTED_VAR
            # Intermediate node - recurse
            println(io, "$(prefix)$(connector)$dep")
            subdeps = graph.edges[dep]
            _print_deps(io, graph, subdeps, child_prefix)
        else
            # Unknown (shouldn't happen)
            println(io, "$(prefix)$(connector)$dep [?]")
        end
    end
end

print_graph(graph::DependencyGraph) = print_graph(stdout, graph)

"""
    Base.show(io::IO, ::MIME"text/plain", graph::DependencyGraph)

Pretty-prints the dependency graph showing all nodes and their relationships.
"""
function Base.show(io::IO, ::MIME"text/plain", graph::DependencyGraph)
    n_prog = count(_ -> true, prognostic_names(graph))
    n_computed = count(t -> t == COMPUTED_VAR, values(graph.node_types))
    n_tend = length(graph.tendency_names)
    
    println(io, "DependencyGraph:")
    println(io, "  $(n_prog) prognostic variables")
    println(io, "  $(n_computed) computed variables")
    println(io, "  $(n_tend) tendencies")
    println(io)
    
    # Prognostic variables (leaves)
    println(io, "Prognostic variables (inputs from Y):")
    for name in prognostic_names(graph)
        println(io, "  • $name")
    end
    println(io)
    
    # Computed variables with dependencies
    if n_computed > 0
        println(io, "Computed variables:")
        for (name, node_type) in graph.node_types
            if node_type == COMPUTED_VAR
                deps = graph.edges[name]
                deps_str = join(string.(deps), ", ")
                println(io, "  • $name ← $deps_str")
            end
        end
        println(io)
    end
    
    # Tendencies with dependencies
    println(io, "Tendencies (outputs to Yₜ):")
    for name in graph.tendency_names
        deps = graph.tendency_edges[name]
        deps_str = isempty(deps) ? "(no deps)" : join(string.(deps), ", ")
        println(io, "  • Yₜ.$name ← $deps_str")
    end
end

"""
    print_evaluation_order(io::IO, graph::DependencyGraph)
    print_evaluation_order(graph::DependencyGraph)

Prints the numbered evaluation order for computed variables and tendencies.
"""
function print_evaluation_order(io::IO, graph::DependencyGraph)
    order = get_evaluation_order(graph)
    
    println(io, "Evaluation order:")
    println(io, "=" ^ 50)
    
    step = 1
    # First: computed variables
    for name in order
        deps = graph.edges[name]
        deps_str = isempty(deps) ? "(no deps)" : "← " * join(string.(deps), ", ")
        println(io, "  $step. [VAR] $name $deps_str")
        step += 1
    end
    
    # Then: tendencies
    println(io, "  --- Tendencies ---")
    for name in graph.tendency_names
        deps = graph.tendency_edges[name]
        deps_str = isempty(deps) ? "(no deps)" : "← " * join(string.(deps), ", ")
        println(io, "  $step. [TENDENCY] Yₜ.$name $deps_str")
        step += 1
    end
    
    println(io, "=" ^ 50)
    println(io, "Total: $(length(order) + length(graph.tendency_names)) computations")
end

print_evaluation_order(graph::DependencyGraph) = print_evaluation_order(stdout, graph)
