using Test
using ClimaCore
using StaticArrays
using OrdinaryDiffEq
using UnicodePlots
using Plots

#=
@testset "test dss_1d!" begin
    FT = Float64
    quad = ClimaCore.Spaces.Quadratures.GLL{4}()
    domain = ClimaCore.Domains.IntervalDomain(FT(0.0), FT(10.0), (:left, :right))
    mesh = ClimaCore.Meshes.EquispacedLineMesh(domain, 2)
    topo = ClimaCore.Topologies.GridTopology1D(mesh)
    space = ClimaCore.Spaces.SpectralElementSpace1D(topo, quad)

    ones_field = ones(FT, space)
    ClimaCore.Spaces.dss_1d!(ones_field)
    ones_field .= one(FT) ./ ones_field
    @test parent(space.dss_weights) == parent(ClimaCore.Fields.field_values(ones_field))
end
=#

const FT = Float64
const parameters = (
    c = 1, # constant advection velocity
)

quad = ClimaCore.Spaces.Quadratures.GLL{5}()
quado = ClimaCore.Spaces.Quadratures.GLL{8}()

domain = ClimaCore.Domains.IntervalDomain(FT(-π), FT(π), (:left, :right))
mesh = ClimaCore.Meshes.EquispacedLineMesh(domain, 20)
topo = ClimaCore.Topologies.GridTopology1D(mesh)
space = ClimaCore.Spaces.SpectralElementSpace1D(topo, quad)
Ispace = ClimaCore.Spaces.SpectralElementSpace1D(topo, quado)

function slab_gradient!(∇data, data, space)
    # all derivatives calculated in the reference local geometry FT precision
    FT = ClimaCore.Spaces.undertype(space)
    D = ClimaCore.Spaces.Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    # for each element in the element stack
    Nh = length(data)
    for h in 1:Nh
        data_slab = ClimaCore.slab(data, h)
        ∇data_slab = ClimaCore.slab(∇data, h)
        ∂f∂ξ₁ = zeros(StaticArrays.MVector{Nq, FT})
        for i in 1:Nq
            # compute covariant derivatives
            ∂f∂ξ₁ .+= D[:,i] * parent(data_slab)[i] 
        end
        # convert to desired basis
        for i in 1:Nq
            parent(∇data_slab)[i] = ∂f∂ξ₁[i]
        end
    end
    return ∇data
end

function tensor_product!(
    out_slab::ClimaCore.DataLayouts.DataSlab1D{S, Ni_out},
    in_slab::ClimaCore.DataLayouts.DataSlab1D{S, Ni_in},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Ni_out, Ni_in}
    for i in 1:Ni_out
        out_slab[i] = ClimaCore.RecursiveApply.rmatmul(M, in_slab, i) 
    end
    return out_slab
end

function tensor_product!(
    out::ClimaCore.DataLayouts.Data1D{S, Ni_out},
    in::ClimaCore.DataLayouts.Data1D{S, Ni_in},
    Imat::SMatrix{Ni_out, Ni_in},
) where {S, Ni_out, Ni_in}
    Nh = length(in)
    @assert Nh == length(out)
    for h in 1:Nh
        in_slab = ClimaCore.slab(in, h)
        out_slab = ClimaCore.slab(out, h)
        tensor_product!(out_slab, in_slab, Imat)
    end
    return out
end

function interpolate!(field_to::ClimaCore.Fields.SpectralElementField1D, 
                      field_from::ClimaCore.Fields.SpectralElementField1D)
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology
                     
    Imat = ClimaCore.Spaces.Quadratures.interpolation_matrix(
        Float64,
        space_to.quadrature_style,
        space_from.quadrature_style,
    )
    tensor_product!(
        ClimaCore.Fields.field_values(field_to),
        ClimaCore.Fields.field_values(field_from),
        Imat,
    )
    return field_to
end


function restrict!(field_to::ClimaCore.Fields.SpectralElementField1D,  # solution space
                   field_from::ClimaCore.Fields.SpectralElementField1D) # OI grid 
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology
    
    Imat = ClimaCore.Spaces.Quadratures.interpolation_matrix(
        Float64,
        space_from.quadrature_style,
        space_to.quadrature_style,
    )
    tensor_product!(
        ClimaCore.Fields.field_values(field_to),
        ClimaCore.Fields.field_values(field_from),
        Imat',
    )
    return field_to
end


#=
result_field = zeros(FT, space)
sin_field = map(x -> sin(x.x3), ClimaCore.Fields.coordinate_field(space))
Nq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)

for i in 1:ClimaCore.Topologies.nlocalelems(topo)
    input_slab = ClimaCore.slab(sin_field, i)
    result_slab = ClimaCore.slab(result_field, i)
    local_geometry_slab = ClimaCore.slab(space.local_geometry, i)
    slab_gradient!(result_slab, input_slab, axes(result_slab))
    for q in 1:Nq
        val = parent(result_slab)[q]
        parent(result_slab)[q] = val * parent(local_geometry_slab.∂ξ∂x)[q]
    end
end

ClimaCore.Spaces.dss_1d!(result_field)
result_field
=#

#U = map(x -> sin(4*x.x3*π/2π), ClimaCore.Fields.coordinate_field(space))

U = map(x -> FT(2) .+ sin(x.x3), ClimaCore.Fields.coordinate_field(space))
ITemp = zeros(FT, Ispace)
IU = zeros(FT, Ispace)
x = ClimaCore.Fields.coordinate_field(space)
function rhs!(dudt, u, _, t)
    interpolate!(IU, u)
    uspace = axes(IU)
    # assert uspace == axes(dudt)
    INq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(uspace.quadrature_style)
    for i in 1:ClimaCore.Topologies.nlocalelems(topo)
        input_slab = ClimaCore.slab(IU, i)
        result_slab = ClimaCore.slab(ITemp, i)
        local_geometry_slab = ClimaCore.slab(uspace.local_geometry, i)
        slab_gradient!(result_slab, input_slab, uspace)
        for q in 1:INq # for OI: apply there on OI quadrature.
            #val = parameters.c * parent(result_slab)[q] * parent(local_geometry_slab.WJ)[q]
            val = parent(result_slab)[q] * parent(local_geometry_slab.WJ)[q]
            parent(result_slab)[q] = val * parent(local_geometry_slab.∂ξ∂x)[q]
        end
        # for OI: then multily this by transpose(basis_func).
    end
    restrict!(dudt, ITemp) 
    ClimaCore.Spaces.dss_1d!(dudt)
    uspace = axes(dudt)
    Nq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(uspace.quadrature_style)
    for i in 1:ClimaCore.Topologies.nlocalelems(topo)
        result_slab = ClimaCore.slab(dudt, i)
        inverse_mass_matrix_slab = ClimaCore.slab(uspace.inverse_mass_matrix, i)
        for q in 1:Nq
            parent(result_slab)[q] *= parent(inverse_mass_matrix_slab)[q] # Minv * dudt
        end
    end
    dudt .*= (-parameters.c)
end   

# Solve the ODE operator
Δt = 0.01
#t_int = 100 * 2π #2π
t_int = 2π #2π
n_steps = cld(t_int, Δt)
@show n_steps
dudt = zeros(FT, space)
rhs!(dudt, U, nothing, 0.0)
#prob = ODEProblem(rhs!, U, (0.0, 1000 * Δt))
prob = ODEProblem(rhs!, U, (0.0, n_steps * Δt))
sol = solve(
    prob,
    AB3(),
    dt = Δt,
    saveat = 10 * Δt,
);
#=
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10 * Δt,
);
=#

xc = parent(ClimaCore.Fields.field_values(x))[:, 1, :][:]
init = parent(ClimaCore.Fields.field_values(U))[:, 1, :][:]
half = cld(length(sol.u), 2)
val_half = parent(ClimaCore.Fields.field_values(sol.u[half]))[:, 1, :][:]
val = parent(ClimaCore.Fields.field_values(sol.u[end]))[:, 1, :][:]

Δx = 0.05424687455240074
CFL = parameters.c * Δt / Δx
@show CFL
#UnicodePlots.lineplot(init, ylim=(-1,1))
#UnicodePlots.lineplot(val, ylim=(-1,1))
#=
using Plots
anim = Plots.@animate for (i, u) in enumerate(sol.u)
    title = "Timepoint: $(i * Δt)"
    data = parent(ClimaCore.Fields.field_values(u))[:, 1, :][:]
    Plots.plot(data, title=title, clim = (-1, 1))
end
#Plots.mp4(anim, "advect.mp4", fps = 3)
Plots.gif(anim, "advect.gif", fps = 3)
=#

val = parent(ClimaCore.Fields.field_values(sol.u[end]))[:, 1, :][:]

init = parent(ClimaCore.Fields.field_values(U))[:, 1, :][:]
#initI = parent(ClimaCore.Fields.field_values(IU))[:, 1, :][:]
#initR = parent(ClimaCore.Fields.field_values(RU))[:, 1, :][:]

println("\n-------init-------------------------")
display(lineplot(xc, init,     ylim = [0 3]))
println("\n--------val_half------------------------")
display(lineplot(xc, val_half, ylim = [0 3]))
println("\n--------val------------------------")
display(lineplot(xc, val,      ylim = [0 3]))
println("\n--------------------------------")

