using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields
import ClimaCore.Fields: is_flat_compatible, flatten_bc_arg

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

# Stand-in for a GPU array: a DenseArray wrapper that the FieldVector
# flattening gate treats as a GPU type, so the gate and the transform can be
# unit-tested without a GPU.
struct FlatTestArray{T, N} <: DenseArray{T, N}
    data::Array{T, N}
end
Base.size(a::FlatTestArray) = size(a.data)
Base.getindex(a::FlatTestArray, i::Int) = a.data[i]
Base.setindex!(a::FlatTestArray, v, i::Int) = (a.data[i] = v)
Base.IndexStyle(::Type{<:FlatTestArray}) = IndexLinear()
Base.strides(a::FlatTestArray) = strides(a.data)
Fields.is_gpu_array_type(::Type{<:FlatTestArray}) = true

@testset "FieldVector flattening gate" begin
    FT = Float64
    dest = FlatTestArray(zeros(FT, 3, 4, 5))
    same = FlatTestArray(rand(FT, 3, 4, 5))
    smaller = FlatTestArray(rand(FT, 3, 4, 1))
    cpu = rand(FT, 3, 4, 5)

    @test is_flat_compatible(dest, same)
    @test is_flat_compatible(dest, FT(2))
    # Broadcasts that expand a smaller argument keep Cartesian indexing.
    @test !is_flat_compatible(dest, smaller)
    # Non-GPU arrays and destinations keep the standard path.
    @test !is_flat_compatible(dest, cpu)
    @test !is_flat_compatible(cpu, same)

    whole = FlatTestArray(rand(FT, 3, 4, 10))
    @test is_flat_compatible(dest, view(whole, :, :, 1:5))
    wide = FlatTestArray(rand(FT, 3, 8, 5))
    @test !is_flat_compatible(dest, view(wide, :, 1:4, :))

    bc = Base.Broadcast.broadcasted(
        +,
        Base.Broadcast.broadcasted(*, same, FT(2)),
        same,
    )
    @test is_flat_compatible(dest, bc)
    bc_mixed = Base.Broadcast.broadcasted(+, same, cpu)
    @test !is_flat_compatible(dest, bc_mixed)

    # The gate and the transform must accept exactly the same argument types:
    # anything the gate rejects has no flatten_bc_arg method, so a divergence
    # fails loudly instead of flattening incorrectly.
    @test !hasmethod(flatten_bc_arg, Tuple{Array{FT, 3}, Vector{FT}, String})
end

@testset "FieldVector flattening transform equivalence" begin
    FT = Float64
    a = FlatTestArray(rand(FT, 3, 4, 5))
    b = FlatTestArray(rand(FT, 3, 4, 5))
    dest = FlatTestArray(zeros(FT, 3, 4, 5))
    bc = Base.Broadcast.broadcasted(
        +,
        Base.Broadcast.broadcasted(*, a, FT(2)),
        b,
    )
    flat_dest = vec(dest)
    copyto!(
        flat_dest,
        Base.Broadcast.instantiate(flatten_bc_arg(dest, flat_dest, bc)),
    )
    @test dest.data ≈ @. 2 * a.data + b.data

    # The destination's own appearance in the broadcast must flatten to the
    # same object handed to copyto!, or Broadcast's aliasing check makes a
    # defensive copy of the destination on every in-place update.
    bc_aliased = Base.Broadcast.broadcasted(+, dest, b)
    fbc = flatten_bc_arg(dest, flat_dest, bc_aliased)
    @test fbc.args[1] === flat_dest
    @test fbc.args[2] !== flat_dest
end

# On GPU runs this exercises the flattened path end-to-end; on CPU runs it
# checks that the gate leaves the standard path unchanged.
@testset "FieldVector broadcast end-to-end" begin
    FT = Float64
    device = ClimaComms.device()
    space = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 10,
        helem = 4,
        Nq = 4,
        context = ClimaComms.context(device),
    )
    z = Fields.coordinate_field(space).z
    X = Fields.FieldVector(x1 = (@. sin(z)), x2 = (@. cos(z)))
    Y = Fields.FieldVector(x1 = (@. z + 1), x2 = (@. z^2 + 1))
    Z = similar(X)
    @. Z = 2 * X + X * Y - 3
    for name in (:x1, :x2)
        x = Array(parent(getproperty(X, name)))
        y = Array(parent(getproperty(Y, name)))
        expected = @. 2 * x + x * y - 3
        @test Array(parent(getproperty(Z, name))) ≈ expected
    end

    # FieldVector assignment takes the array-to-array fast path.
    W = similar(X)
    W .= X
    for name in (:x1, :x2)
        @test Array(parent(getproperty(W, name))) ==
              Array(parent(getproperty(X, name)))
    end

    # In-place update where the destination appears in its own broadcast: on
    # GPU this exercises the identity-preserving flatten path, which must not
    # fall back to a defensive aliasing copy or change results.
    z_old = Dict(
        name => Array(parent(getproperty(Z, name))) for name in (:x1, :x2)
    )
    @. Z = 2 * Z + Y - 1
    for name in (:x1, :x2)
        y = Array(parent(getproperty(Y, name)))
        expected = @. 2 * z_old[name] + y - 1
        @test Array(parent(getproperty(Z, name))) ≈ expected
    end
end
