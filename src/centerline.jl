"""
    Centerline extraction + GENERIC tree assembly.

Extracts centerlines from NURBS vessel surfaces and assembles them into
tree structures driven by OrganConfig (no hardcoded organ names).
"""

struct XCATCenterline
    name::String
    centers::Vector{SVector{3, Float64}}
    radii::Vector{Float64}
    axial_param::Symbol
end

struct XCATTreeConnection
    parent_segment::String
    child_segment::String
    parent_index::Int
    child_index::Int
    gap_mm::Float64
end

struct XCATCenterlineTree
    name::String
    segments::Dict{String, XCATCenterline}
    root_segment::String
    connections::Vector{XCATTreeConnection}
end

# ── Axis detection ──

function xcat_surface_axis(surface::XCATNurbsSurface)
    pts = surface.control_points
    u_closure = mean(norm.(pts[:, 1] .- pts[:, end]))
    v_closure = mean(norm.(pts[1, :] .- pts[end, :]))
    return u_closure <= v_closure ? :v : :u
end

# ── Centerline extraction ──

function _trim_centerline_caps(centers::Vector{SVector{3, Float64}}, radii::Vector{Float64}; min_radius_fraction::Float64=0.2)
    positive = filter(>(0.0), radii)
    isempty(positive) && return centers, radii
    threshold = min_radius_fraction * median(positive)
    first_keep = findfirst(r -> r >= threshold, radii)
    last_keep = findlast(r -> r >= threshold, radii)
    (first_keep === nothing || last_keep === nothing || first_keep > last_keep) && return centers, radii
    return centers[first_keep:last_keep], radii[first_keep:last_keep]
end

function xcat_centerline_from_surface(surface::XCATNurbsSurface; circumferential_samples::Int=48, axial_samples::Union{Nothing, Int}=nothing)
    n_u, n_v = xcat_uv_counts(surface)
    axial = xcat_surface_axis(surface)
    if axial === :v
        n_axial = something(axial_samples, max(16, n_v * 2))
        n_circ = max(12, circumferential_samples)
        points, _, _, _ = xcat_sample_surface(surface; n_u=n_circ, n_v=n_axial, orient_outward=false)
        centers = Vector{SVector{3, Float64}}(undef, n_axial)
        radii = Vector{Float64}(undef, n_axial)
        for j in 1:n_axial
            row = points[j, :]
            center = let acc = SVector(0.0, 0.0, 0.0)
                for p in row; acc += p; end
                acc / length(row)
            end
            centers[j] = center
            radii[j] = mean(norm.(row .- Ref(center)))
        end
        centers, radii = _trim_centerline_caps(centers, radii)
        return XCATCenterline(surface.name, centers, radii, :v)
    else
        n_axial = something(axial_samples, max(16, n_u * 2))
        n_circ = max(12, circumferential_samples)
        points, _, _, _ = xcat_sample_surface(surface; n_u=n_axial, n_v=n_circ, orient_outward=false)
        centers = Vector{SVector{3, Float64}}(undef, n_axial)
        radii = Vector{Float64}(undef, n_axial)
        for i in 1:n_axial
            col = points[:, i]
            center = let acc = SVector(0.0, 0.0, 0.0)
                for p in col; acc += p; end
                acc / length(col)
            end
            centers[i] = center
            radii[i] = mean(norm.(col .- Ref(center)))
        end
        centers, radii = _trim_centerline_caps(centers, radii)
        return XCATCenterline(surface.name, centers, radii, :u)
    end
end

function xcat_centerline_length_mm(centerline::XCATCenterline)
    total = 0.0
    for i in 1:(length(centerline.centers) - 1)
        total += norm(centerline.centers[i + 1] - centerline.centers[i])
    end
    return total
end

function xcat_reverse_centerline(centerline::XCATCenterline)
    XCATCenterline(centerline.name, reverse(centerline.centers), reverse(centerline.radii), centerline.axial_param)
end

# ── Orientation helpers (work with arbitrary surface names) ──

function _point_to_centerline_distance(point::SVector{3, Float64}, centerline::XCATCenterline)
    minimum(norm(point - c) for c in centerline.centers)
end

function _connection_distance(a::XCATCenterline, b::XCATCenterline)
    norm(last(a.centers) - first(b.centers))
end

function _orient_root_to_anchor(centerline::XCATCenterline, anchor::XCATCenterline)
    d_start = _point_to_centerline_distance(first(centerline.centers), anchor)
    d_end = _point_to_centerline_distance(last(centerline.centers), anchor)
    d_start <= d_end ? centerline : xcat_reverse_centerline(centerline)
end

function _orient_to_previous(previous::XCATCenterline, current::XCATCenterline)
    forward = _connection_distance(previous, current)
    reversed = _connection_distance(previous, xcat_reverse_centerline(current))
    forward <= reversed ? current : xcat_reverse_centerline(current)
end

function _nearest_centerline_pair(parent::XCATCenterline, child::XCATCenterline)
    best_dist = Inf
    best_parent_idx = 1
    best_child_idx = 1
    for (i, p) in enumerate(parent.centers)
        for (j, q) in enumerate(child.centers)
            d = norm(p - q)
            if d < best_dist
                best_dist = d
                best_parent_idx = i
                best_child_idx = j
            end
        end
    end
    return best_parent_idx, best_child_idx, best_dist
end

function _orient_child_to_parent(parent::XCATCenterline, child::XCATCenterline)
    _, _, forward_dist = _nearest_centerline_pair(parent, child)
    reversed = xcat_reverse_centerline(child)
    _, _, reversed_dist = _nearest_centerline_pair(parent, reversed)
    forward_dist <= reversed_dist ? child : reversed
end

function _xcat_slice_centerline_from(centerline::XCATCenterline, start_idx::Int)
    start_idx = clamp(start_idx, 1, length(centerline.centers))
    XCATCenterline(centerline.name, centerline.centers[start_idx:end], centerline.radii[start_idx:end], centerline.axial_param)
end

function _snap_connection_to_parent_endpoint(parent::XCATCenterline, child::XCATCenterline, parent_idx::Int, child_idx::Int, gap::Float64; max_tail_points::Int=2, max_extra_gap_mm::Float64=1.0)
    parent_tail_points = length(parent.centers) - parent_idx
    parent_tail_points <= max_tail_points || return parent_idx, child_idx, gap
    endpoint = last(parent.centers)
    best_child_idx = child_idx
    best_gap = Inf
    for (j, q) in enumerate(child.centers)
        d = norm(endpoint - q)
        if d < best_gap
            best_gap = d
            best_child_idx = j
        end
    end
    if best_gap <= gap + max_extra_gap_mm
        return length(parent.centers), best_child_idx, best_gap
    end
    return parent_idx, child_idx, gap
end

function _make_tree_connection(parent::XCATCenterline, child::XCATCenterline)
    oriented_child = _orient_child_to_parent(parent, child)
    parent_idx, child_idx, gap = _nearest_centerline_pair(parent, oriented_child)
    parent_idx, child_idx, gap = _snap_connection_to_parent_endpoint(parent, oriented_child, parent_idx, child_idx, gap)
    trimmed_child = _xcat_slice_centerline_from(oriented_child, child_idx)
    connection = XCATTreeConnection(parent.name, trimmed_child.name, parent_idx, 1, gap)
    return trimmed_child, connection
end

# ── GENERIC tree assembly (config-driven) ──

"""
    build_vessel_trees(centerlines, config::OrganConfig)

Assemble centerlines into named tree structures using the vessel_trees
specification from config. Each VesselTreeSpec defines:
  - name: tree display name
  - surface_names: ordered list of XCAT surface names forming the chain
  - root_anchor_surface: surface used to orient the root segment

Returns a Dict{String, XCATCenterlineTree}.
"""
function build_vessel_trees(centerlines::AbstractVector{XCATCenterline}, config::OrganConfig)
    cmap = Dict(line.name => line for line in centerlines)

    # Find the reference surface (e.g. aorta) for root orientation
    anchor_cl = haskey(cmap, config.reference_surface) ? cmap[config.reference_surface] : nothing

    result = Dict{String, XCATCenterlineTree}()

    # If there is a reference surface, include it as a tree
    if anchor_cl !== nothing
        result[config.reference_surface] = XCATCenterlineTree(
            config.reference_surface,
            Dict(anchor_cl.name => anchor_cl),
            anchor_cl.name,
            XCATTreeConnection[],
        )
    end

    for spec in config.vessel_trees
        isempty(spec.surface_names) && continue

        # Resolve the anchor for root orientation
        root_anchor = if !isempty(spec.root_anchor_surface) && haskey(cmap, spec.root_anchor_surface)
            cmap[spec.root_anchor_surface]
        elseif anchor_cl !== nothing
            anchor_cl
        else
            nothing
        end

        # First segment: orient toward the anchor
        first_name = spec.surface_names[1]
        haskey(cmap, first_name) || continue
        root_cl = if root_anchor !== nothing
            _orient_root_to_anchor(cmap[first_name], root_anchor)
        else
            cmap[first_name]
        end

        segments = Dict{String, XCATCenterline}(root_cl.name => root_cl)
        connections = XCATTreeConnection[]

        # Chain subsequent segments
        prev = root_cl
        for k in 2:length(spec.surface_names)
            seg_name = spec.surface_names[k]
            haskey(cmap, seg_name) || continue
            oriented = _orient_to_previous(prev, cmap[seg_name])
            child_oriented, conn = _make_tree_connection(prev, oriented)
            segments[child_oriented.name] = child_oriented
            push!(connections, conn)
            prev = child_oriented
        end

        result[spec.name] = XCATCenterlineTree(spec.name, segments, root_cl.name, connections)
    end

    return result
end
