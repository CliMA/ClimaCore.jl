module ClimaCorePlots

import RecipesBase
import TriplotBase

import ClimaComms
# Keep in sync with definition(s) in ClimaCore.DataLayouts.
@inline slab_index(i::T, j::T) where {T} =
    CartesianIndex(i, j, T(1), T(1), T(1))
@inline slab_index(i::T) where {T} = CartesianIndex(i, T(1), T(1), T(1), T(1))

import ClimaCore:
    ClimaCore,
    DataLayouts,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Topologies,
    Spaces,
    Utilities,
    Remapping

using StaticArrays


RecipesBase.@recipe function f(
    field::Fields.SpectralElementField1D;
    interpolate = 10,
)
    @assert interpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

    # compute the interpolated data to plot
    space = axes(field)
    nelems = Topologies.nlocalelems(space)
    QS = Spaces.quadrature_style(space)
    quad_name = Base.typename(typeof(QS)).name
    Nq = Quadratures.degrees_of_freedom(QS)
    Nu = max(interpolate, Nq)
    coord_field = Fields.coordinate_field(space)

    lagrange_quad = Quadratures.ClosedUniform{Nu}()
    dataspace =
        Spaces.SpectralElementSpace1D(Spaces.topology(space), lagrange_quad)
    interp = Operators.Interpolate(dataspace)
    ifield = interp.(field)

    coords = vec(parent(Fields.coordinate_field(ifield).x))
    data = vec(parent(Fields.field_values(ifield)))

    # NaN injection element boundaries
    r = range(1, nelems + 1, length(coords) + 1)
    xcoords = [coords[1]]
    xdata = [data[1]]
    for k in 2:(length(r) - 1)
        if floor(r[k - 1]) != floor(r[k]) # check this.
            push!(xcoords, NaN)
            push!(xdata, NaN)
        end
        push!(xcoords, coords[k])
        push!(xdata, data[k])
    end

    # set the plot attributes
    coord_symbols = propertynames(coord_field)
    seriestype := :path
    xguide --> "$(coord_symbols[1])"
    label --> "$nelems $quad_name{$Nq} element space"

    (xcoords, xdata)
end

RecipesBase.@recipe function f(space::Spaces.RectilinearSpectralElementSpace2D;)
    quad = Spaces.quadrature_style(space)
    quad_name = Base.typename(typeof(quad)).name
    dof = Quadratures.degrees_of_freedom(quad)

    topology = Spaces.topology(space)
    mesh = topology.mesh
    @assert mesh isa Meshes.RectilinearMesh "plotting only defined for RectilinearMesh"

    n1 = Meshes.nelements(mesh.intervalmesh1)
    n2 = Meshes.nelements(mesh.intervalmesh2)

    coord_field = Fields.coordinate_field(space)
    x1coord = vec(parent(coord_field)[:, :, 1, :])
    x2coord = vec(parent(coord_field)[:, :, 2, :])

    coord_symbols = propertynames(coord_field)

    seriestype := :scatter
    title --> "$n1 × $n2 $quad_name{$dof} element space"
    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    marker --> :cross
    markersize --> 1
    seriescolor --> :blue
    legend --> false

    (x1coord, x2coord)
end

RecipesBase.@recipe function f(space::Spaces.ExtrudedFiniteDifferenceSpace)
    coord_field = Fields.coordinate_field(space)
    data = Fields.field_values(coord_field)
    Ni, Nj, _, Nv, Nh = size(data)

    #TODO: assumes VIFH layout
    @assert Nj == 1 "plotting only defined for 1D extruded fields"

    hspace = Spaces.horizontal_space(space)

    quad = Spaces.quadrature_style(hspace)
    quad_name = Base.typename(typeof(quad)).name
    dof = Quadratures.degrees_of_freedom(quad)

    coord_symbols = propertynames(coord_field)
    hcoord = vec(parent(coord_field)[:, :, 1, :])
    vcoord = vec(parent(coord_field)[:, :, 2, :])

    stagger = space.staggering isa Spaces.CellCenter ? :center : :face

    seriestype := :scatter
    title --> "$Nh $quad_name{$dof} element × $Nv level $stagger space"
    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    marker --> :cross
    markersize --> 1
    seriescolor --> :blue
    legend --> false

    (hcoord, vcoord)
end

RecipesBase.@recipe function f(field::Fields.FiniteDifferenceField)
    # unwrap the data to plot
    space = axes(field)
    coord_field = Fields.coordinate_field(space)

    xdata = parent(field)[:, 1]
    ydata = parent(Spaces.coordinates_data(space).z)[:, 1]

    coord_symbols = propertynames(coord_field)

    # set the plot attributes
    xguide --> "value"
    yguide --> (
        if field isa Spaces.FaceFiniteDifferenceSpace
            "$(coord_symbols[1]) faces"
        else
            "$(coord_symbols[1]) centers"
        end
    )

    # fix the ylim to the column space (domain)
    ylims --> extrema(ydata)
    linewidth --> 2
    legend --> false

    (xdata, ydata)
end

RecipesBase.@recipe function f(
    field::Fields.RectilinearSpectralElementField2D;
    interpolate = 10,
)
    @assert interpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

    # compute the interpolated data to plot
    space = axes(field)
    topology = Spaces.topology(space)
    mesh = topology.mesh

    Nu = interpolate
    coord_field = Fields.coordinate_field(space)

    M_coords = Operators.matrix_interpolate(coord_field, Nu)
    M = Operators.matrix_interpolate(field, Nu)

    domain = Meshes.domain(mesh)
    x1min = Geometry.component(domain.interval1.coord_min, 1)
    x2min = Geometry.component(domain.interval2.coord_min, 1)
    x1max = Geometry.component(domain.interval1.coord_max, 1)
    x2max = Geometry.component(domain.interval2.coord_max, 1)

    # our interpolated field is transposed
    x1coord = [Geometry.component(pt, 1) for pt in M_coords[:, 1]]
    x2coord = [Geometry.component(pt, 2) for pt in M_coords[1, :]]

    coord_symbols = propertynames(coord_field)

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    seriescolor --> :balance

    (x1coord, x2coord, M')
end


function _slice_triplot(field, hinterpolate, ncolors)
    data = Fields.field_values(field)
    Ni, Nj, _, Nv, Nh = size(data)

    space = axes(field)
    htopology = Spaces.topology(space)
    hdomain = Topologies.domain(htopology)
    vdomain = Topologies.domain(Spaces.vertical_topology(space))

    @assert Nj == 1

    hcoord_field = getproperty(Fields.coordinate_field(space), 1)
    vcoord_field = getproperty(Fields.coordinate_field(space), 2)

    if hinterpolate ≥ 1
        Nu = hinterpolate
        uquad = Quadratures.ClosedUniform{Nu}()
        M_hcoord = Operators.matrix_interpolate(hcoord_field, uquad)
        M_vcoord = Operators.matrix_interpolate(vcoord_field, uquad)
        M_data = Operators.matrix_interpolate(field, uquad)

        hcoord_data = vec(M_hcoord)
        vcoord_data = vec(M_vcoord)
        data = vec(M_data)
        triangles = hcat(Spaces.triangles(Nv, Nu, Nh)...)'
    else
        hcoord_data = vec(parent(hcoord_field))
        vcoord_data = vec(parent(vcoord_field))
        data = vec(parent(data))
        triangles = hcat(Spaces.triangles(Nv, Ni, Nh)...)'
    end
    cmap = range(extrema(data)..., length = ncolors)

    # unique number of nodal element coords
    # (number of nodal values minus number of shared faces)
    Px = (Ni * Nh) - (Nh - 1)
    Py = Nv
    cdata = TriplotBase.tripcolor(
        hcoord_data,
        vcoord_data,
        data,
        triangles,
        cmap;
        bg = NaN,
        px = Px,
        py = Py,
    )
    domain_xmin = Geometry.component(hdomain.coord_min, 1)
    domain_xmax = Geometry.component(hdomain.coord_max, 1)
    domain_ymin = Geometry.component(vdomain.coord_min, 1)
    domain_ymax = Geometry.component(vdomain.coord_max, 1)
    cx_coords = range(start = domain_xmin, stop = domain_xmax, length = Px)
    cy_coords = range(start = domain_ymin, stop = domain_ymax, length = Py)
    return (cx_coords, cy_coords, cdata')
end

# 2D hybrid plot
RecipesBase.@recipe function f(
    field::Fields.ExtrudedSpectralElementField2D;
    hinterpolate = 0,
    ncolors = 256,
)
    hcoord, vcoord, data = _slice_triplot(field, hinterpolate, ncolors)
    coord_symbols = propertynames(Fields.coordinate_field(axes(field)))

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    seriescolor --> :balance

    (hcoord, vcoord, data)
end

function _slice_along(field, coord)
    # which axis are we slicing?
    axis = if coord isa ClimaCore.Geometry.XPoint
        1
    elseif coord isa ClimaCore.Geometry.YPoint
        2
    else
        error(
            "coord must be a ClimaCore.Geometry.XPoint or ClimaCore.Geometry.YPoint",
        )
    end
    space = axes(field)
    hspace = Spaces.horizontal_space(space)
    htopo = ClimaCore.Spaces.topology(hspace)
    hmesh = htopo.mesh
    linear_idx = LinearIndices(ClimaCore.Meshes.elements(hmesh))

    # get the axis interval mesh we are slicing along
    hmesh_slice, hmesh_ortho = if axis == 1
        (htopo.mesh.intervalmesh1, htopo.mesh.intervalmesh2)
    else
        (htopo.mesh.intervalmesh2, htopo.mesh.intervalmesh1)
    end

    # check that the coordinate is in the domain
    domain = hmesh_slice.domain
    if coord < domain.coord_min || coord > domain.coord_max
        error("$coord is outside of the field axis domain: $domain")
    end

    # get the elment offset given the axis slice
    slice_h = ClimaCore.Meshes.containing_element(hmesh_slice, coord)
    hidx = axis == 1 ? linear_idx[slice_h, 1] : linear_idx[1, slice_h]

    # find the node idx we want to slice along the given axis element
    hcoord_data = Spaces.local_geometry_data(hspace).coordinates
    hdata = ClimaCore.slab(hcoord_data, hidx)
    hnode_idx = 1
    for i in axes(hdata)[axis]
        pt = axis == 1 ? hdata[slab_index(i, 1)] : hdata[slab_index(1, i)]
        axis_value = Geometry.component(pt, axis)
        coord_value = Geometry.component(coord, 1)
        if axis_value > coord_value
            break
        end
        hnode_idx = i
    end

    # construct the slice field space
    context = ClimaComms.context(field)
    htopo_ortho = ClimaCore.Topologies.IntervalTopology(
        ClimaComms.device(context),
        hmesh_ortho,
    )
    hspace_ortho = ClimaCore.Spaces.SpectralElementSpace1D(
        htopo_ortho,
        ClimaCore.Spaces.quadrature_style(hspace),
    )
    vspace_ortho = ClimaCore.Spaces.CenterFiniteDifferenceSpace(
        Spaces.vertical_topology(space),
    )

    if space.staggering === ClimaCore.Spaces.CellFace
        vspace_ortho = ClimaCore.Spaces.FaceFiniteDifferenceSpace(slice_vspace)
    end
    space_ortho = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
        hspace_ortho,
        vspace_ortho,
    )

    # construct the slice field ortho to the slice coordinate
    ortho_field = Fields.Field(eltype(field), space_ortho)

    # copy the data into this field for the given node idx for all elements intersecting slice coord
    field_data = ClimaCore.Fields.todata(field)
    ortho_data = ClimaCore.Fields.todata(ortho_field)

    for i in 1:size(linear_idx, axis == 1 ? 2 : 1)
        for v in 1:ClimaCore.Spaces.nlevels(space_ortho)
            hidx = axis == 1 ? linear_idx[slice_h, i] : linear_idx[i, slice_h]
            ijslab = ClimaCore.slab(field_data, v, hidx)
            islab = ClimaCore.slab(ortho_data, v, i)
            # copy the nodal data
            for ni in 1:size(islab)[1]
                islab[slab_index(ni)] =
                    axis == 1 ? ijslab[slab_index(hnode_idx, ni)] :
                    ijslab[slab_index(ni, hnode_idx)]
            end
        end
    end
    return ortho_field
end

# 3D hybrid plot
RecipesBase.@recipe function f(
    field::Fields.ExtrudedFiniteDifferenceField3D;
    slice = nothing,
    hinterpolate = 0,
    ncolors = 256,
)
    if slice === nothing
        error("must specify coordinate axis slice for 3D hybrid plots")
    end
    if length(slice) != 3
        error("must specify a length 3 slice for 3D hybrid plots")
    end
    if count(c -> c isa Colon, slice) != 2
        error("can only specify one axis to slice")
    end
    if !(slice[3] isa Colon)
        error("slicing only along the X and Y axis is supported")
    end
    slice_coord = if slice[1] isa Colon
        ClimaCore.Geometry.YPoint{eltype(slice[2])}(slice[2])
    else
        ClimaCore.Geometry.XPoint{eltype(slice[1])}(slice[1])
    end
    slice_field = _slice_along(field, slice_coord)
    hcoord, vcoord, data = _slice_triplot(slice_field, hinterpolate, ncolors)

    coord_symbols = propertynames(Fields.coordinate_field(axes(slice_field)))

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    seriescolor --> :balance

    (hcoord, vcoord, data)
end

function _unfolded_pannel_matrix(field, interpolate)
    space = axes(field)
    FT = Spaces.undertype(space)
    topology = Spaces.topology(space)
    mesh = topology.mesh
    nelem = Meshes.nelements(mesh)
    panel_size = Meshes.n_elements_per_panel_direction(mesh)

    quad_from = Spaces.quadrature_style(space)
    quad_to = Quadratures.Uniform{interpolate}()
    Imat = Quadratures.interpolation_matrix(FT, quad_to, quad_from)

    dof = interpolate

    pannel_range(i) =
        ((panel_size * dof) * (i - 1) + 1):((panel_size * dof) * i)

    # construct a matrix to fill in the rotated / flipped pannel data
    unfolded_panels =
        fill(NaN, ((panel_size * dof) * 3, (panel_size * dof) * 4))

    # temporary pannels as we have to rotate / flip some and not all operators are in place
    # TODO: inefficient memory wise, but good enough for now
    panels = [fill(NaN, (panel_size * dof, panel_size * dof)) for _ in 1:6]

    field_data = Fields.field_values(field)
    fdim = DataLayouts.field_dim(DataLayouts.singleton(field_data))
    interpolated_data_type = if fdim == ndims(field_data)
        DataLayouts.IJHF
    else
        DataLayouts.IJFH
    end
    interpolated_data =
        interpolated_data_type{FT, interpolate}(Array{FT}, nelem)

    Operators.tensor_product!(interpolated_data, field_data, Imat)

    # element index ordering defined by a specific layout
    for (lidx, elem) in enumerate(topology.elemorder)
        ex1, ex2, panel_idx = elem.I
        panel_data = panels[panel_idx]
        # compute the nodal extent index range for this element
        x1_nodal_range = (dof * (ex1 - 1) + 1):(dof * ex1)
        x2_nodal_range = (dof * (ex2 - 1) + 1):(dof * ex2)
        # transpose the data as our plotting axis order is
        # reverse nodal element order (x1 axis varies fastest)
        data_element = permutedims(parent(interpolated_data)[:, :, 1, lidx])
        panel_data[x2_nodal_range, x1_nodal_range] = data_element
    end

    # (px, py, rot) for each panel
    # px, py are the locations of the panel
    # rot is number of clockwise rotations (0:3)
    # https://extranet.gfdl.noaa.gov/~atw/ferret/cubed_sphere/
    # Equatorial strip, poles connected through the Americas
    panel_locations =
        [(4, 2, 0), (1, 2, 0), (3, 3, 2), (2, 2, 1), (3, 2, 1), (3, 1, 1)]
    for (i, (px, py, rot)) in enumerate(panel_locations)
        unfolded_panels[pannel_range(py), pannel_range(px)] .= if rot == 0
            panels[i]
        elseif rot == 1
            reverse(transpose(panels[i]), dims = 1)
        elseif rot == 2
            reverse(panels[i], dims = (1, 2))
        else
            reverse(transpose(panels[i]), dims = 2)
        end
    end
    #=
    unfolded_panels[pannel_range(1), pannel_range(2)] =
        reverse(panels[5], dims = 1)
    unfolded_panels[pannel_range(2), pannel_range(1)] =
        reverse(panels[4], dims = 2)
    unfolded_panels[pannel_range(2), pannel_range(2)] = transpose(panels[1])
    unfolded_panels[pannel_range(2), pannel_range(3)] = transpose(panels[2])
    unfolded_panels[pannel_range(2), pannel_range(4)] =
        reverse(panels[6], dims = 2)
    unfolded_panels[pannel_range(3), pannel_range(2)] =
        reverse(panels[3], dims = 2)
    =#
    return unfolded_panels
end

RecipesBase.@recipe function f(
    field::Fields.CubedSphereSpectralElementField2D;
    interpolate = 10,
    remap_to_latlon = false,
    nlat = 180,
    nlon = 180,
)
    FT = eltype(field)

    if remap_to_latlon

        Δh_scale = Spaces.node_horizontal_length_scale(Spaces.axes(field))
        planet_radius = FT(6.378e6)
        npts = Int(round(2π * planet_radius / Δh_scale))
        npts = ifelse(rem(npts, 2) == 0, npts, npts + 1)
        npts *= 4
        longpts = range(FT(-180), FT(180.0), Int(nlon))
        latpts = range(FT(-90), FT(90), Int(nlat))
        hcoords = [
            Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts
        ]

        # @Main.infiltrate

        # hcoords = Remapping.default_target_hcoords(axes(field))

        remapper = Remapping.Remapper(axes(field), hcoords)
        remapped = Array(Remapping.interpolate(remapper, field))

        # Set plot attributes
        seriestype := :heatmap
        xguide --> "Longitude"
        yguide --> "Latitude"
        seriescolor --> :balance
        title --> "Lat-Lon Remapped Field ($nlat × $nlon)"

        return (longpts, latpts, transpose(remapped))

    else
        # Original panel plotting branch
        @assert interpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

        unfolded_panels = _unfolded_pannel_matrix(field, interpolate)

        # construct the title for info about the field space
        space = axes(field)
        nelem = Topologies.nlocalelems(Spaces.topology(space))
        quad_from = Spaces.quadrature_style(space)
        dof_in = Quadratures.degrees_of_freedom(quad_from)
        quad_from_name = Base.typename(typeof(quad_from)).name

        # set the plot attributes
        seriestype := :heatmap
        title --> "$nelem $quad_from_name{$dof_in} element space"
        xguide --> "panel x1"
        yguide --> "panel x2"
        seriescolor --> :balance

        (unfolded_panels)
    end
end

RecipesBase.@recipe function f(
    field::Fields.ExtrudedCubedSphereSpectralElementField3D;
    level = nothing,
    hinterpolate = 10,
)
    @assert hinterpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

    space = axes(field)
    if level === nothing
        if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
            level = Utilities.PlusHalf(0)
        else
            level = 1
        end
    end
    level_field = Fields.level(field, level)
    unfolded_panels = _unfolded_pannel_matrix(level_field, hinterpolate)

    # construct the title for info about the field space
    nlevel = Spaces.nlevels(space)
    nelem = Topologies.nlocalelems(Spaces.topology(space))
    quad_from = Spaces.quadrature_style(space)
    dof_in = Quadratures.degrees_of_freedom(quad_from)
    quad_from_name = Base.typename(typeof(quad_from)).name

    # set the plot attributes
    seriestype := :heatmap
    title -->
    "level $level of $nlevel × $nelem $quad_from_name{$dof_in} element space"
    xguide --> "panel x1"
    yguide --> "panel x2"
    seriescolor --> :balance

    (unfolded_panels)
end

end # module
