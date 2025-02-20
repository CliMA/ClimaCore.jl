ENV["CLIMACOMMS_DEVICE"] = "CUDA"; # requires cuda
using LazyBroadcast: lazy
using ClimaCore.Utilities: half
using Test, ClimaComms
ClimaComms.@import_required_backends;
using ClimaCore: Geometry, Spaces, Fields, Operators, ClimaCore;
using ClimaCore.CommonSpaces;

get_space_extruded(dev, FT) = ExtrudedCubedSphereSpace(
    FT;
    device = dev,
    z_elem = 63,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 30,
    n_quad_points = 4,
    staggering = CellCenter(),
);

get_space_column(dev, FT) = ColumnSpace(
    FT;
    device = dev,
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    staggering = CellCenter(),
);

function kernels!(fields)
    (; f, ρ, ϕ) = fields
    (; ᶜout1, ᶜout2, ᶜout3, ᶜout4, ᶜout5, ᶜout6, ᶜout7, ᶜout8) = fields
    FT = Spaces.undertype(axes(ϕ))
    div_bcs = Operators.DivergenceF2C(;
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(100))),
        top = Operators.SetValue(Geometry.Covariant3Vector(FT(10000))),
    )
    div = Operators.DivergenceF2C()
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. ᶜout1 = div(Geometry.WVector(f))
    @. ᶜout2 = 0
    @. ᶜout2 += div(Geometry.WVector(f) * 2)
    @. ᶜout3 = div(Geometry.WVector(ᶠwinterp(ϕ, ρ)))

    @. ᶜout4 = div_bcs(Geometry.WVector(f))
    @. ᶜout5 = 0
    @. ᶜout5 += div_bcs(Geometry.WVector(f) * 2)
    @. ᶜout6 = div_bcs(Geometry.WVector(ᶠwinterp(ϕ, ρ)))

    # from the wild
    Ic2f = Operators.InterpolateC2F(; top = Operators.Extrapolate())
    divf2c = Operators.DivergenceF2C(;
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(100000000))),
    )
    # only upward component of divergence
    @. ᶜout7 = divf2c(Geometry.WVector(Ic2f(ϕ))) # works
    @. ᶜout8 = divf2c(Ic2f(Geometry.WVector(ϕ))) # breaks
    return nothing
end;

function get_fields(space::Operators.AllFaceFiniteDifferenceSpace)
    FT = Spaces.undertype(space)
    (; z) = Fields.coordinate_field(space)
    nt = (; f = Fields.Field(FT, space))
    @. nt.f = sin(z)
    return nt
end

function get_fields(space::Operators.AllCenterFiniteDifferenceSpace)
    FT = Spaces.undertype(space)
    K = (ntuple(i -> Symbol("ᶜout$i"), 8)..., :ρ, :ϕ)
    V = ntuple(i -> Fields.zeros(space), length(K))
    (; z) = Fields.coordinate_field(space)
    nt = (; zip(K, V)...)
    @. nt.ρ = sin(z)
    @. nt.ϕ = sin(z)
    return nt
end

function compare_cpu_gpu(cpu, gpu; print_diff = true, C_best = 10)
    # there are some odd errors that build up when run without debug / bounds checks:
    space = axes(cpu)
    are_boundschecks_forced = Base.JLOptions().check_bounds == 1
    absΔ = abs.(parent(cpu) .- Array(parent(gpu)))
    B =
        are_boundschecks_forced ? maximum(absΔ) <= 1000 * eps() :
        maximum(absΔ) <= 10000000 * eps()
    C =
        are_boundschecks_forced ? count(x -> x <= 1000 * eps(), absΔ) :
        count(x -> x <= 10000000 * eps(), absΔ)
    if !B && print_diff
        if space isa Spaces.FiniteDifferenceSpace
            @show parent(cpu)[1:3]
            @show parent(gpu)[1:3]
            @show parent(cpu)[(end - 3):end]
            @show parent(gpu)[(end - 3):end]
        else
            @show parent(cpu)[1:3, 1, 1, 1, end]
            @show parent(gpu)[1:3, 1, 1, 1, end]
            @show parent(cpu)[(end - 3):end, 1, 1, 1, end]
            @show parent(gpu)[(end - 3):end, 1, 1, 1, end]
        end
    end
    @test B
    return B
end

# This function is useful for debugging new cases.
function compare_cpu_gpu_incremental(cpu, gpu; print_diff = true, C_best = 10)
    # there are some odd errors that build up when run without debug / bounds checks:
    space = axes(cpu)
    are_boundschecks_forced = Base.JLOptions().check_bounds == 1
    absΔ = abs.(parent(cpu) .- Array(parent(gpu)))
    B =
        are_boundschecks_forced ? maximum(absΔ) <= 1000 * eps() :
        maximum(absΔ) <= 10000000 * eps()
    C =
        are_boundschecks_forced ? count(x -> x <= 1000 * eps(), absΔ) :
        count(x -> x <= 10000000 * eps(), absΔ)
    @test C ≥ C_best
    if !(C_best == 10)
        C > C_best && @show C_best
        @test_broken C > C_best
    end
    if !B && print_diff
        if space isa Spaces.FiniteDifferenceSpace
            @show parent(cpu)[1:3]
            @show parent(gpu)[1:3]
            @show parent(cpu)[(end - 3):end]
            @show parent(gpu)[(end - 3):end]
        else
            @show parent(cpu)[1:3, 1, 1, 1, end]
            @show parent(gpu)[1:3, 1, 1, 1, end]
            @show parent(cpu)[(end - 3):end, 1, 1, 1, end]
            @show parent(gpu)[(end - 3):end, 1, 1, 1, end]
        end
    end
    return true
end

is_trivial(x) = length(parent(x)) == count(iszero, parent(x)) # Make sure we don't have a trivial solution

nothing
