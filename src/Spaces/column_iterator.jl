
struct ColumnIterator{Nh, Nj, Ni}
    function ColumnIterator(space::AbstractSpace)
        Ni, Nj, _, _, Nh = size(local_geometry_data(space))
        return new{Nh, Nj, Ni}()
    end
end

Base.size(::ColumnIterator{Nh, Nj, Ni}) where {Nh, Nj, Ni} = (Nh, Nj, Ni)
Base.length(::ColumnIterator{Nh, Nj, Ni}) where {Nh, Nj, Ni} =
    prod((Nh, Nj, Ni))

function iterate_columns(space::AbstractSpace)
    (Nh, Nj, Ni) = size(ColumnIterator(space))
    return Iterators.product(1:Ni, 1:Nj, 1:Nh)
end
