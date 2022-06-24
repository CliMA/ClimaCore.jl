
struct SlabIndex{VIdx}
    v::VIdx
    h::Int
end




Base.getindex(field::Field, slabidx::SlabIndex) = slab(field, slabidx)

function slab(field::SpectralElementField, slabidx::SlabIndex{Nothing})
    slab(field, slabidx.h)
end
function slab(
    field::CenterExtrudedFiniteDifferenceField,
    slabidx::SlabIndex{Int},
)
    slab(field, slabidx.v, slabidx.h)
end
function slab(
    field::FaceExtrudedFiniteDifferenceField,
    slabidx::SlabIndex{PlusHalf{Int}},
)
    slab(field, slabidx.v + half, slabidx.h)
end

function byslab(fn, space::Spaces.AbstractSpectralElementSpace)
    Nh = Topologies.nlocalelems(space.topology)::Int
    Threads.@threads for h in 1:Nh
        fn(SlabIndex(nothing, h))
    end
end
function byslab(fn, space::Spaces.CenterExtrudedFiniteDifferenceSpace)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    for h in 1:Nh
        for v in 1:Nv
            fn(SlabIndex(v, h))
        end
    end
end
function byslab(fn, space::Spaces.FaceExtrudedFiniteDifferenceSpace)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    Nv = Spaces.nlevels(space)
    for h in 1:Nh
        for v in 1:Nv
            fn(SlabIndex(v - half, h))
        end
    end
end

