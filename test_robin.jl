import ClimaCore as CC
using ClimaComms
using ClimaCorePlots
using Plots

function compare_dirichlet(value, data)
    p = plot()
    bottom_bc = CC.Operators.SetValue(value)
    top_bc = CC.Operators.SetValue(value)
    gradc2f = CC.Operators.GradientC2F(bottom=bottom_bc, top=top_bc)
    ∇data = gradc2f.(data)
    plot!(
        map(x -> x.w, CC.Geometry.WVector.(∇data)),
        color=:black,
        label="Dirichlet",
    )

    bottom_bc = CC.Operators.SetRobin([value, 0.0, 1.0])
    top_bc = CC.Operators.SetRobin([value, 0.0, 1.0])
    gradc2f = CC.Operators.GradientC2F(bottom=bottom_bc, top=top_bc)
    ∇data = gradc2f.(data)
    plot!(
        map(x -> x.w, CC.Geometry.WVector.(∇data)),
        color=:red,
        label="Robin",
        linestyle=:dot,
    )
    plot!(
        title="g=$value",
        legend=:true,
    )
    return p
end


function compare_neumann(value, data)
    p = plot()
    bottom_bc = CC.Operators.SetGradient(CC.Geometry.WVector(value))
    top_bc = CC.Operators.SetGradient(CC.Geometry.WVector(value))
    gradc2f = CC.Operators.GradientC2F(bottom=bottom_bc, top=top_bc)
    ∇data = gradc2f.(data)
    plot!(
        map(x -> x.w, CC.Geometry.WVector.(∇data)),
        color=:black,
        label="Neumann",
    )

    bottom_bc = CC.Operators.SetRobin([value, 1.0, 0.0])
    top_bc = CC.Operators.SetRobin([value, 1.0, 0.0])
    gradc2f = CC.Operators.GradientC2F(bottom=bottom_bc, top=top_bc)
    ∇data = gradc2f.(data)
    plot!(
        map(x -> x.w, CC.Geometry.WVector.(∇data)),
        color=:red,
        label="Robin",
        linestyle=:dot,
    )
    plot!(
        title="g=$value",
        legend=:true,
    )
    return p
end

function get_coord(lower_boundary::Float64, upper_boundary::Float64, nelems::Int)
    device = ClimaComms.device()
    domain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint(lower_boundary),
        CC.Geometry.ZPoint(upper_boundary);
        boundary_names=(:bottom, :top),
    )
    mesh = CC.Meshes.IntervalMesh(domain, nelems=nelems)
    cspace = CC.Spaces.CenterFiniteDifferenceSpace(device, mesh)
    coord = CC.Fields.coordinate_field(cspace)
    return coord
end

coord = get_coord(0.0, 10.0, 20)
data = cos.(coord.z)

l = @layout [a b c; d e f]

pd1 = compare_dirichlet(1.0, data)
pd2 = compare_dirichlet(0.0, data)
pd3 = compare_dirichlet(-1.0, data)
pn1 = compare_neumann(1.0, data)
pn2 = compare_neumann(0.0, data)
pn3 = compare_neumann(-1.0, data)
plot(pd1, pd2, pd3, pn1, pn2, pn3; layout = l)
plot!(;
    titlefontsize=12,
    size=(800,600),
    link=:all,
    xlabel="∇cos(z)",
    ylabel="z",
    legend_position=:right,
)
