using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: IJFH
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra

using DifferentialEquations

FT = Float64

domain = Domains.RectangleDomain(
    x1min = FT(-π),
    x1max = FT(π),
    x2min = FT(-π),
    x2max = FT(π),
    x1periodic = true,
    x2periodic = true,
)
discretiation = Domains.EquispacedRectangleDiscretization(domain, 5, 5)
grid_topology = Topologies.GridTopology(discretiation)

Nq = 6
quad = Meshes.Quadratures.GLL{Nq}()
points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)
mesh = Meshes.Mesh2D(grid_topology, quad)

f0(x) = exp(-x.x1^2 / 2)
y0 = f0.(Fields.coordinate_field(mesh))

# https://github.com/sandreza/NodalDiscontinuousGalerkin/blob/b971af24d1f6cc63016a76e2063547de5022e867/DG1D/solveHeat.jl

function rhs!(rawdydt,rawdata,mesh,t)
    data = IJFH{Float64,Nq}(rawdata)
    dydt = IJFH{Float64,Nq}(rawdydt)

    ∇data = Operators.volume_gradient!(
        similar(data, Geometry.Cartesian2DVector{Float64}),
        data,
        mesh,
    )
    Operators.volume_weak_divergence!(
        dydt,
        ∇data,
        mesh,
    )

    WJ = copy(mesh.local_geometry.WJ)
    Operators.horizontal_dss!(WJ, mesh)
    Operators.horizontal_dss!(dydt .* mesh.local_geometry.WJ, mesh)
    dydt .= dydt ./ WJ
    dydt
    #K = stiffness matrix
#and the mass matrix M
#. ∇φi(x)∇φj(x)dx
    #M du/dt = Kuˆ+ ...
end

prob = ODEProblem(rhs!, parent(Fields.field_values(y0)),(0.0,1.0),mesh)


sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)


