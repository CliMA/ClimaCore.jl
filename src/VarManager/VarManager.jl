"""
    VarManager

A module for managing the computation of derived variables and tendencies from
prognostic state variables. It provides infrastructure for:

- Defining dependencies between computed variables
- Building dependency graphs from prognostic variables to tendencies
- Topologically sorting the dependency graph for correct evaluation order
- Evaluating the graph with configurable caching strategies
- User-friendly debugging of computation errors

## Usage

ClimaAtmos (or other packages) should implement:
- `var_dependencies(name::FieldName, model)` - returns tuple of FieldNames that `name` depends on
- `tendency_dependencies(name::FieldName, model)` - returns tuple of FieldNames that tendency `name` depends on
- `compute_var(name::FieldName, model, vars, t)` - computes the value for `name`
- `compute_tendency(name::FieldName, model, vars, t)` - computes the tendency for `name`

Then ClimaCore's VarManager handles graph construction, sorting, and evaluation.
"""
module VarManager

import ..MatrixFields: FieldName, FieldNameSet, FieldVectorKeys, @name, @Name
import ..MatrixFields: FieldNameDict, FieldNameTree, concrete_field_vector
import ..Fields
import ..Fields: Field, FieldVector
import ..Spaces
import LazyBroadcast

using UnrolledUtilities

export VarCachingStrategy, EagerGlobalCaching, NoCaching
export DependencyGraph, build_dependency_graph, topological_sort, prognostic_names, validate_dependencies
export var_dependencies, tendency_dependencies
export compute_var, compute_tendency
export evaluate_graph, evaluate_graph!, evaluate_tendency, evaluate_tendency!, ComputationError
export VarCache, VarCacheValue, get_cached
export print_graph, print_evaluation_order

include("caching.jl")
include("dependency_graph.jl")
include("evaluation.jl")
include("debugging.jl")

end # module
