@inline check_for_non_column_args(args::Tuple, inds...) = (
    check_for_non_column(args[1], inds...),
    check_for_non_column_args(Base.tail(args), inds...)...,
)
@inline check_for_non_column_args(args::Tuple{Any}, inds...) =
    (check_for_non_column(args[1], inds...),)
@inline check_for_non_column_args(args::Tuple{}, inds...) = ()

@inline function check_for_non_column(
    bc::StencilBroadcasted{Style},
    inds...
) where {Style}
    StencilBroadcasted{Style}(
        bc.op,
        check_for_non_column_args(bc.args, inds...),
        bc.axes
    )
end
@inline function check_for_non_column(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...
) where {Style}
    Base.Broadcast.Broadcasted{Style}(
        bc.f,
        check_for_non_column_args(bc.args, inds...),
        bc.axes
    )
end
@inline function check_for_non_column(f::Fields.Field, inds...)
    check_for_non_column(Fields.field_values(f), inds...)
    return Fields.Field(Fields.field_values(f), axes(f))
end
@inline check_for_non_column(x::Tuple, inds...) =
    (check_for_non_column(first(x), inds...),
    check_for_non_column(Base.tail(x), inds...)...)
@inline check_for_non_column(x::Tuple{Any}, inds...) =
    (check_for_non_column(first(x), inds...),)
@inline check_for_non_column(x::Tuple{}, inds...) = ()

@inline check_for_non_column(x, inds...) = x
@inline check_for_non_column(x::DataLayouts.VIJFH, inds...) = error("Found non-column data $x.")
@inline check_for_non_column(x::DataLayouts.VIFH, inds...) = error("Found non-column data $x.")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

@inline check_for_non_column_style_args(args::Tuple, inds...) = (
    check_for_non_column_style(args[1], inds...),
    check_for_non_column_style_args(Base.tail(args), inds...)...,
)
@inline check_for_non_column_style_args(args::Tuple{Any}, inds...) =
    (check_for_non_column_style(args[1], inds...),)
@inline check_for_non_column_style_args(args::Tuple{}, inds...) = ()

@inline function check_for_non_column_style(
    bc::StencilBroadcasted{Style},
    inds...
) where {Style}
    check_for_non_column_style(Style)
    StencilBroadcasted{Style}(
        bc.op,
        check_for_non_column_style_args(bc.args, inds...),
        bc.axes
    )
end
@inline function check_for_non_column_style(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...
) where {Style}
    check_for_non_column_style(Style)
    Base.Broadcast.Broadcasted{Style}(
        bc.f,
        check_for_non_column_style_args(bc.args, inds...),
        bc.axes
    )
end
@inline check_for_non_column_style(::Fields.FieldStyle{DS}, inds...) where {DS} = check_for_non_column_style(DS)
@inline check_for_non_column_style(::Type{Fields.FieldStyle{DS}}, inds...) where {DS} = check_for_non_column_style(DS)
@inline check_for_non_column_style(::Type{DS}, inds...) where {DS <: DataLayouts.VIJFHStyle} = error("Found non-column style")
@inline check_for_non_column_style(::Type{DS}, inds...) where {DS <: DataLayouts.VIFHStyle} = error("Found non-column style")
@inline check_for_non_column_style(x::Tuple) =
    (check_for_non_column_style(first(x), inds...),
    check_for_non_column_style(Base.tail(x), inds...)...)
@inline check_for_non_column_style(x::Tuple{Any}, inds...) =
    (check_for_non_column_style(first(x), inds...),)
@inline check_for_non_column_style(x::Tuple{}, inds...) = ()
@inline check_for_non_column_style(x, inds...) = x
