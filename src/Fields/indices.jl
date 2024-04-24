
Base.@propagate_inbounds Base.getindex(field::Field, colidx::ColumnIndex) =
    column(field, colidx)
Base.@propagate_inbounds function Base.getindex(
    fv::FieldVector{T},
    colidx::ColumnIndex,
) where {T}
    values = map(x -> x[colidx], _values(fv))
    return FieldVector{T, typeof(values)}(values)
end
Base.@propagate_inbounds function column(
    field::SpectralElementField1D,
    colidx::ColumnIndex{1},
)
    column(field, colidx.ij[1], colidx.h)
end
Base.@propagate_inbounds function column(
    field::ExtrudedFiniteDifferenceField,
    colidx::ColumnIndex{1},
)
    column(field, colidx.ij[1], colidx.h)
end
Base.@propagate_inbounds function column(
    field::SpectralElementField2D,
    colidx::ColumnIndex{2},
)
    column(field, colidx.ij[1], colidx.ij[2], colidx.h)
end
Base.@propagate_inbounds function column(
    field::ExtrudedFiniteDifferenceField,
    colidx::ColumnIndex{2},
)
    column(field, colidx.ij[1], colidx.ij[2], colidx.h)
end

"""
    Fields.bycolumn(fn, space)

Call `fn(colidx)` to every [`ColumnIndex`](@ref) `colidx` of `space`. This can
be used to apply multiple column-wise operations in a single pass, making use of
multiple threads.

!!! note

    On GPUs this will simply evaluate `f` once with `colidx=:` (i.e. it doesn't
    perform evaluation by columns). This may change in future.

# Example

```julia
∇ = GradientF2C()
div = DivergenceC2F()

bycolumn(axes(f)) do colidx
    @. ∇f[colidx] = ∇(f[colidx])
    @. df[colidx] = div(∇f[colidx])
end
```
"""
function bycolumn(fn, space::Spaces.AbstractSpace)
    bycolumn(fn, space, ClimaComms.device(space))
end

function bycolumn(
    fn,
    space::Spaces.SpectralElementSpace1D,
    ::ClimaComms.CPUSingleThreaded,
)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        for h in 1:Nh
            for i in 1:Nq
                fn(ColumnIndex((i,), h))
            end
        end
    end
    return nothing
end
function bycolumn(
    fn,
    space::Spaces.SpectralElementSpace1D,
    ::ClimaComms.CPUMultiThreaded,
)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for i in 1:Nq
                fn(ColumnIndex((i,), h))
            end
        end
    end
    return nothing
end
function bycolumn(
    fn,
    space::Spaces.SpectralElementSpace2D,
    ::ClimaComms.CPUSingleThreaded,
)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        for h in 1:Nh
            for j in 1:Nq, i in 1:Nq
                fn(ColumnIndex((i, j), h))
            end
        end
    end
    return nothing
end
function bycolumn(
    fn,
    space::Spaces.SpectralElementSpace2D,
    ::ClimaComms.CPUMultiThreaded,
)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for j in 1:Nq, i in 1:Nq
                fn(ColumnIndex((i, j), h))
            end
        end
    end
    return nothing
end
bycolumn(
    fn,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    device::ClimaComms.AbstractCPUDevice,
) = bycolumn(fn, Spaces.horizontal_space(space), device)



"""
    ncolumns(::Field)
    ncolumns(::Space)

Number of columns in a given space.
"""
ncolumns(field::Field) = ncolumns(axes(field))

ncolumns(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    ncolumns(Spaces.horizontal_space(space))

function ncolumns(space::Spaces.SpectralElementSpace1D)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return Nh * Nq
end
function ncolumns(space::Spaces.SpectralElementSpace2D)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return Nh * Nq * Nq
end

# potential TODO:
# - define a ColumnIndices type, make it work with https://github.com/JuliaFolds/FLoops.jl



struct SlabIndex{VIdx, HIdx}
    v::VIdx
    h::HIdx
end


Base.@propagate_inbounds Base.getindex(field::Field, slabidx::SlabIndex) =
    slab(field, slabidx)

Base.@propagate_inbounds function slab(
    field::SpectralElementField,
    slabidx::SlabIndex{Nothing},
)
    slab(field, slabidx.h)
end
Base.@propagate_inbounds function slab(
    field::CenterExtrudedFiniteDifferenceField,
    slabidx::SlabIndex{Int},
)
    slab(field, slabidx.v, slabidx.h)
end
Base.@propagate_inbounds function slab(
    field::FaceExtrudedFiniteDifferenceField,
    slabidx::SlabIndex{PlusHalf{Int}},
)
    slab(field, slabidx.v + half, slabidx.h)
end

function byslab(fn, space::Spaces.AbstractSpace)
    byslab(fn, ClimaComms.device(space), space)
end

function byslab(
    fn,
    ::ClimaComms.CPUSingleThreaded,
    space::Spaces.AbstractSpectralElementSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))::Int
    @inbounds for h in 1:Nh
        fn(SlabIndex(nothing, h))
    end
end

function byslab(
    fn,
    ::ClimaComms.CPUMultiThreaded,
    space::Spaces.AbstractSpectralElementSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))::Int
    @inbounds begin
        Threads.@threads for h in 1:Nh
            fn(SlabIndex(nothing, h))
        end
    end
end


function byslab(
    fn,
    ::ClimaComms.CPUSingleThreaded,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    @inbounds begin
        for h in 1:Nh
            for v in 1:Nv
                fn(SlabIndex(v, h))
            end
        end
    end
end

function byslab(
    fn,
    ::ClimaComms.CPUMultiThreaded,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for v in 1:Nv
                fn(SlabIndex(v, h))
            end
        end
    end
end

function byslab(
    fn,
    ::ClimaComms.CPUSingleThreaded,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    @inbounds begin
        for h in 1:Nh
            for v in 1:Nv
                fn(SlabIndex(v - half, h))
            end
        end
    end
end

function byslab(
    fn,
    ::ClimaComms.CPUMultiThreaded,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for v in 1:Nv
                fn(SlabIndex(v - half, h))
            end
        end
    end
end
