module ClimaCoreCatalyst

import SciMLBase

import ClimaCore: ClimaCore, Fields, Geometry, Operators, Spaces, Topologies
import ParaviewCatalyst: ParaviewCatalyst, Conduit, ConduitNode

export CatalystCallback

# Adapted from DiffEqCallbacks PeriodicCallback to use only SciMLBase primitives
struct CatalystCallbackAffect{dT,Ref1,Ref2}
    Δt::dT
    t0::Ref1
    index::Ref2
end

function (S::CatalystCallbackAffect)(integrator)
    (; Δt, t0, index) = S

    ClimaCoreCatalyst.execute(integrator.u, time = integrator.t, cycle = index[])

    tstops = integrator.opts.tstops
    tnew = t0[] + (index[] + 1) * Δt
    tstops = integrator.opts.tstops
    tdir_tnew = integrator.tdir * tnew
    for i in length(tstops) : -1 : 1
        if tdir_tnew < tstops.valtree[i]
            index[] += 1
            SciMLBase.add_tstop!(integrator, tnew)
            break
        end
    end
end

function CatalystCallback(Δt::Number;
                          initial_affect = true, # call before any time integration
                          initialize = (cb,u,t,integrator) -> SciMLBase.u_modified!(integrator, initial_affect),
                          kwargs...)

    # Value of `t` at which `f` should be called next:
    t0 = Ref(typemax(Δt))
    index = Ref(0)
    condition = function (u, t, integrator)
        t == (t0[] + index[] * Δt)
    end
    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = CatalystCallbackAffect(Δt, t0, index)
    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        t0[] = t
        if initial_affect
            index[] = 0
            affect!(integrator)
        else
            index[] = 1
            SciMLBase.add_tstop!(integrator, t0[] + Δt)
        end
    end
    return SciMLBase.DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

function initialize(;libpath=nothing)
    if libpath === nothing 
        @warn "if you are on anything other than Jakes personal lapatop need to supply a catalyst libpath"
    end
    # TODO: check initialization state and no-op if already initialized (easier for REPL)
    ParaviewCatalyst.catalyst_initialize(;libpath)
end

# stub
viz(fields) = execute(fields)

function execute(fields; time = 0.0, timestep = 0, cycle = 0)
    ConduitNode() do node
        node["catalyst/state/timestep"] = timestep
        node["catalyst/state/time"] = time
        node["catalyst/state/cycle"] = cycle
        node["catalyst/channels/input/type"] = "mesh"
        ConduitNode() do mesh_node
            node["catalyst/channels/input/data"] = conduit_mesh(mesh_node, fields)
        end
        ParaviewCatalyst.catalyst_execute(node)
    end
end

function conduit_mesh(node::ConduitNode, fields)
    # add mesh coordinate and topology info given element face vertices
    fspace = element_face_space(fields)
    add_mesh_coordsets!(node, fspace)
    add_mesh_topologies!(node, fspace)
    # interpolate data to element centers and write out field data
    cspace = element_center_space(fspace)
    add_mesh_fields!(node, fields, cspace)
    return node
end

function add_mesh_coordsets!(node::ConduitNode, facespace)
    coords = Geometry.Cartesian123Point.(
        Fields.coordinate_field(facespace),
        Ref(facespace.global_geometry),
    )
    node["coordsets/coords/type"] = "explicit"
    node["coordsets/coords/values/x"] = Array(parent(coords.x1))
    node["coordsets/coords/values/y"] = Array(parent(coords.x2))
    node["coordsets/coords/values/z"] = Array(parent(coords.x3))
    return node
end

function add_mesh_topologies!(node::ConduitNode, facespace)
    shape, connection = element_vertex_connectivity(facespace)
    node["topologies/mesh/type"] = "unstructured"
    node["topologies/mesh/coordset"] = "coords"
    node["topologies/mesh/elements/shape"] = shape
    node["topologies/mesh/elements/connectivity"] = connection
    return node
end

add_mesh_fields!(node::ConduitNode, fields, centerspace) =
    add_mesh_fields!(node, "", fields, centerspace)

function add_mesh_fields!(node::ConduitNode, prefix, fields, centerspace)
    for (key, val) in pairs(fields)
        name = string(key)
        if !isempty(prefix)
            name = prefix * "." * name
        end
        add_mesh_fields!(node, name, val, centerspace)
    end
end
add_mesh_fields!(node::ConduitNode, prefix, field::Fields.FieldVector, centerspace) =
    add_mesh_fields!(node, prefix, Fields._values(field), centerspace)

add_mesh_fields!(node::ConduitNode, prefix, field::Fields.Field, centerspace) =
    add_mesh_fields!(node, eltype(field), prefix, field, centerspace)

function add_mesh_fields!(node::ConduitNode, ::Type{T}, prefix, fields, centerspace) where {T}
    for i in 1:fieldcount(T)
        name = string(fieldname(T, i))
        if !isempty(prefix)
            name = prefix * "." * name
        end
        add_mesh_fields!(node, name, getproperty(fields, i), centerspace)
    end
end

# element center scalar field
function add_mesh_fields!(node::ConduitNode, ::Type{<:Real}, name, field, centerspace)
    interp = Operators.Interpolate(centerspace)
    ifield = interp.(field)
    name = isempty(name) ? "data" : name
    node["fields/$name/topology"] = "mesh"
    node["fields/$name/association"] = "element"
    node["fields/$name/type"] = "scalar"
    # TODO: zero copy field views set_external!(node, path, pointer...)
    node["fields/$name/values"] = Array(parent(ifield))
    return node
end

# element center vector field
function add_mesh_fields!(node::ConduitNode, ::Type{<:Geometry.AxisVector}, name, field, centerspace)
    interp = Operators.Interpolate(centerspace)
    ifield = interp.(Geometry.Cartesian123Vector.(field))
    name = isempty(name) ? "data" : name
    node["fields/$name/topology"] = "mesh"
    node["fields/$name/association"] = "element"
    node["fields/$name/type"] = "vector"
    # TODO: zero copy reshaped arrays set_external!(node, path, pointer...)
    node["fields/$name/values/u"] = Array(parent(ifield.components.data.:1))
    node["fields/$name/values/v"] = Array(parent(ifield.components.data.:2))
    node["fields/$name/values/w"] = Array(parent(ifield.components.data.:3))
    return node
end

function element_vertex_connectivity(space::Spaces.SpectralElementSpace2D)
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    Nh = Topologies.nlocalelems(space)
    ind = LinearIndices((1:Nq, 1:Nq, 1:Nh))
    connection = Vector{Int}(undef, Nh * (Nq - 1) * (Nq - 1) * 4)
    # mesh blueprint assumes VTK vertex winding convention
    vidx = 0
    for e in 1:Nh, j in 1:(Nq - 1), i in 1:(Nq - 1)
        connection[vidx + 1] = ind[i, j, e]
        connection[vidx + 2] = ind[i + 1, j, e]
        connection[vidx + 3] = ind[i + 1, j + 1, e]
        connection[vidx + 4] = ind[i, j + 1, e]
        vidx += 4
    end
    connection .-= 1
    return ("quad", connection)
end

#TODO: add the hexs

function element_vertex_connectivity(space::Spaces.FaceExtrudedFiniteDifferenceSpace)
    horizontal_space = space.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        horizontal_space.quadrature_style,
    )
    Nh = Topologies.nlocalelems(horizontal_space)
    Nv = Spaces.nlevels(space)
    ind = LinearIndices((1:Nv, 1:Nq, 1:Nh))
    # TODO: we probably don't have to materialize the vector, we can add an iterator if
    # the conduit api was extended to support a lightweight ref to the data memoryspace?
    connection = Vector{Int}(undef, Nh * (Nq - 1) * (Nv - 1) * 4)
    # mesh blueprint assumes VTK vertex winding convention
    for e in 1:Nh, i in 1:(Nq - 1), v in 1:(Nv - 1)
        connection[vidx + 1] = ind[v, i, e]
        connection[vidx + 2] = ind[v + 1, i, e]
        connection[vidx + 3] = ind[v + 1, i + 1, e]
        connection[vidx + 4] = ind[v, i + 1, e]
        vidx += 4
    end
    # TODO: dont do this
    connection .-= 1
    return ("quad", connection)
end

function element_face_space(space::Spaces.SpectralElementSpace1D)
    if space.quadrature_style isa Spaces.Quadratures.ClosedUniform
        return space
    end
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    lagrange_quad = Spaces.Quadratures.ClosedUniform{Nq}()
    return Spaces.SpectralElementSpace1D(space.topology, lagrange_quad)
end
function element_face_space(space::Spaces.SpectralElementSpace2D)
    if space.quadrature_style isa Spaces.Quadratures.ClosedUniform
        return space
    end
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    lagrange_quad = Spaces.Quadratures.ClosedUniform{Nq}()
    return Spaces.SpectralElementSpace2D(space.topology, lagrange_quad)
end
function element_face_space(space::Spaces.FaceExtrudedFiniteDifferenceSpace)
    horizontal_space = element_face_space(space.horizontal_space)
    vertical_space = Spaces.FaceFiniteDifferenceSpace(space.vertical_topology)
    return Spaces.ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
    )
end
element_face_space(space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    element_face_space(Spaces.FaceExtrudedFiniteDifferenceSpace(space))
element_face_space(field::Fields.Field) = element_face_space(axes(field))
element_face_space(fields::NamedTuple) = element_face_space(first(fields))
element_face_space(fieldvec::Fields.FieldVector) =
    element_face_space(Fields._values(fieldvec))

function element_center_space(facespace::Spaces.SpectralElementSpace1D)
    @assert facespace.quadrature_style isa Spaces.Quadratures.ClosedUniform
    Nq = Spaces.Quadratures.degrees_of_freedom(facespace.quadrature_style)
    quad = Spaces.Quadratures.Uniform{Nq - 1}()
    return Spaces.SpectralElementSpace1D(facespace.topology, quad)
end

function element_center_space(facespace::Spaces.SpectralElementSpace2D)
    @assert facespace.quadrature_style isa Spaces.Quadratures.ClosedUniform
    Nq = Spaces.Quadratures.degrees_of_freedom(facespace.quadrature_style)
    quad = Spaces.Quadratures.Uniform{Nq - 1}()
    return Spaces.SpectralElementSpace2D(facespace.topology, quad)
end

function element_center_space(facespace::Spaces.FaceExtrudedFiniteDifferenceSpace)
    horizontal_space = element_center_space(facespace.horizontal_space)
    vertical_space =
        Spaces.CenterFiniteDifferenceSpace(facespace.vertical_topology)
    return Spaces.ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
    )
end

end # module
