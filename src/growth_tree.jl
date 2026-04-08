"""
    GrowthTree struct + manipulation.

Includes tree construction from XCAT centerlines, seed-point initialization,
branch addition, Murray's law updates, and distance helpers.
"""

mutable struct GrowthTree
    name::String
    vertices::Vector{SVector{3, Float64}}
    parent_vertex::Vector{Int}
    incoming_segment::Vector{Int}
    children::Vector{Vector{Int}}
    segment_start::Vector{Int}
    segment_end::Vector{Int}
    segment_diameter_cm::Vector{Float64}
    segment_label::Vector{String}
    root_vertex::Int
end

# ── Distance helpers ──

function _distance_point_segment_cm(point::SVector{3, Float64}, a::SVector{3, Float64}, b::SVector{3, Float64})
    ab = b - a
    denom = dot(ab, ab)
    if denom <= 1e-12
        return norm(point - a)
    end
    t = clamp(dot(point - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return norm(point - proj)
end

function _tree_segment_distance_cm(tree::GrowthTree, point::SVector{3, Float64})
    # If tree has no segments yet (seed tree), use distance to root vertex
    if isempty(tree.segment_start)
        return norm(point - tree.vertices[tree.root_vertex])
    end
    best = Inf
    for s in eachindex(tree.segment_start)
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        best = min(best, _distance_point_segment_cm(point, a, b))
    end
    return best
end

function _nearest_tree_vertex(tree::GrowthTree, point::SVector{3, Float64})
    best_idx = 1
    best_d = Inf
    for (i, p) in enumerate(tree.vertices)
        d = norm(point - p)
        if d < best_d
            best_d = d
            best_idx = i
        end
    end
    return best_idx, best_d
end

function _nearest_tree_segment_projection(tree::GrowthTree, point::SVector{3, Float64})
    # If tree has no segments (seed tree), return root vertex info
    if isempty(tree.segment_start)
        return 0, 0.0, tree.vertices[tree.root_vertex], norm(point - tree.vertices[tree.root_vertex])
    end
    best_seg = 1
    best_dist = Inf
    best_t = 0.0
    best_proj = tree.vertices[tree.segment_start[1]]
    for s in eachindex(tree.segment_start)
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        ab = b - a
        denom = dot(ab, ab)
        if denom <= 1e-12
            d = norm(point - a)
            t = 0.0
            proj = a
        else
            t = clamp(dot(point - a, ab) / denom, 0.0, 1.0)
            proj = a + t * ab
            d = norm(point - proj)
        end
        if d < best_dist
            best_dist = d
            best_seg = s
            best_t = t
            best_proj = proj
        end
    end
    return best_seg, best_t, best_proj, best_dist
end

# ── Tree modification ──

function _split_segment!(tree::GrowthTree, seg_id::Int, point::SVector{3, Float64})
    start_v = tree.segment_start[seg_id]
    end_v = tree.segment_end[seg_id]
    old_end_children = tree.children[end_v]
    parent_children = tree.children[start_v]
    idx = findfirst(==(end_v), parent_children)
    idx === nothing && return end_v
    push!(tree.vertices, point)
    mid_v = length(tree.vertices)
    push!(tree.parent_vertex, start_v)
    push!(tree.incoming_segment, seg_id)
    push!(tree.children, Int[end_v])
    parent_children[idx] = mid_v
    tree.parent_vertex[end_v] = mid_v
    old_d = tree.segment_diameter_cm[seg_id]
    old_label = tree.segment_label[seg_id]
    tree.segment_end[seg_id] = mid_v
    push!(tree.segment_start, mid_v)
    push!(tree.segment_end, end_v)
    push!(tree.segment_diameter_cm, old_d)
    push!(tree.segment_label, old_label)
    tree.incoming_segment[end_v] = length(tree.segment_start)
    return mid_v
end

function _choose_anchor_vertex(tree::GrowthTree, point::SVector{3, Float64}; split_range=(0.2, 0.8))
    seg_id, t, proj, _ = _nearest_tree_segment_projection(tree, point)
    # Seed tree: no segments, anchor at root
    if seg_id == 0
        return tree.root_vertex, tree.vertices[tree.root_vertex]
    end
    if split_range[1] <= t <= split_range[2]
        return _split_segment!(tree, seg_id, proj), proj
    end
    start_v = tree.segment_start[seg_id]
    end_v = tree.segment_end[seg_id]
    ds = norm(point - tree.vertices[start_v])
    de = norm(point - tree.vertices[end_v])
    vid = ds <= de ? start_v : end_v
    return vid, tree.vertices[vid]
end

function _update_upstream_murray!(tree::GrowthTree, start_vertex::Int; gamma::Float64=3.0)
    v = start_vertex
    while v != 0
        child_vertices = tree.children[v]
        if length(child_vertices) >= 2 && tree.incoming_segment[v] != 0
            child_segments = [tree.incoming_segment[c] for c in child_vertices if tree.incoming_segment[c] != 0]
            if !isempty(child_segments)
                target_d = (sum(tree.segment_diameter_cm[s]^gamma for s in child_segments))^(1.0 / gamma)
                tree.segment_diameter_cm[tree.incoming_segment[v]] = max(tree.segment_diameter_cm[tree.incoming_segment[v]], target_d)
            end
        end
        v = tree.parent_vertex[v]
    end
    return nothing
end

_branch_terminals(tree::GrowthTree) = [i for i in eachindex(tree.vertices) if isempty(tree.children[i])]

# ── Path helpers ──

function _path_total_length(pts::Vector{SVector{3, Float64}})
    s = 0.0
    for i in 2:length(pts)
        s += norm(pts[i] - pts[i-1])
    end
    return s
end

function _densify_path(pts::Vector{SVector{3, Float64}}, max_seg_cm::Float64)
    length(pts) <= 1 && return pts
    out = SVector{3, Float64}[pts[1]]
    for i in 2:length(pts)
        a = out[end]
        b = pts[i]
        seg_len = norm(b - a)
        if seg_len > max_seg_cm
            n_sub = ceil(Int, seg_len / max_seg_cm)
            for k in 1:n_sub
                t = k / n_sub
                push!(out, (1.0 - t) .* a .+ t .* b)
            end
        else
            push!(out, b)
        end
    end
    return out
end

function _add_branch_path!(tree::GrowthTree, anchor_vertex::Int, path_points::Vector{SVector{3, Float64}}; cutoff_diameter_cm::Float64=8e-4, gamma::Float64=3.0, max_branch_length_cm::Float64=Inf, max_segment_length_cm::Float64=0.1)
    isempty(path_points) && return false
    local_pts = copy(path_points)
    if norm(local_pts[1] - tree.vertices[anchor_vertex]) < 1e-8
        local_pts = local_pts[2:end]
    end
    isempty(local_pts) && return false

    total_len = _path_total_length(vcat([tree.vertices[anchor_vertex]], local_pts))
    if isfinite(max_branch_length_cm) && total_len > max_branch_length_cm
        return false
    end

    local_pts = _densify_path(local_pts, max_segment_length_cm)

    parent_seg = tree.incoming_segment[anchor_vertex]
    parent_d = parent_seg == 0 ? 0.04 : tree.segment_diameter_cm[parent_seg]
    branch_root_d = max(cutoff_diameter_cm, min(0.5 * parent_d, 0.05))
    prev = anchor_vertex
    nseg = length(local_pts)
    for (j, p) in enumerate(local_pts)
        push!(tree.vertices, p)
        vid = length(tree.vertices)
        push!(tree.parent_vertex, prev)
        push!(tree.incoming_segment, length(tree.segment_start) + 1)
        push!(tree.children, Int[])
        frac = nseg == 1 ? 1.0 : (j - 1) / (nseg - 1)
        seg_d = branch_root_d * (1.0 - frac) + cutoff_diameter_cm * frac
        push!(tree.segment_start, prev)
        push!(tree.segment_end, vid)
        push!(tree.segment_diameter_cm, seg_d)
        push!(tree.segment_label, "grown")
        push!(tree.children[prev], vid)
        prev = vid
    end
    _update_upstream_murray!(tree, anchor_vertex; gamma=gamma)
    return true
end

# ── Construction from XCAT centerlines ──

function growth_tree_from_xcat(name::String, tree::XCATCenterlineTree)
    vertices = SVector{3, Float64}[]
    parent_vertex = Int[]
    incoming_segment = Int[]
    children = Vector{Int}[]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    point_map = Dict{Tuple{String, Int}, Int}()

    root = tree.segments[tree.root_segment]
    for (i, p) in enumerate(root.centers)
        push!(vertices, 0.1 .* p)
        push!(parent_vertex, i == 1 ? 0 : i - 1)
        push!(incoming_segment, i == 1 ? 0 : length(segment_start) + 1)
        push!(children, Int[])
        point_map[(root.name, i)] = i
        if i > 1
            push!(segment_start, i - 1)
            push!(segment_end, i)
            push!(segment_diameter_cm, 0.1 * (root.radii[i - 1] + root.radii[i]))
            push!(segment_label, root.name)
            push!(children[i - 1], i)
        end
    end

    for conn in tree.connections
        parent_seg = tree.segments[conn.parent_segment]
        child_seg = tree.segments[conn.child_segment]
        anchor = point_map[(parent_seg.name, conn.parent_index)]
        point_map[(child_seg.name, 1)] = anchor
        prev = anchor
        for j in 2:length(child_seg.centers)
            push!(vertices, 0.1 .* child_seg.centers[j])
            vid = length(vertices)
            push!(parent_vertex, prev)
            push!(incoming_segment, length(segment_start) + 1)
            push!(children, Int[])
            point_map[(child_seg.name, j)] = vid
            push!(segment_start, prev)
            push!(segment_end, vid)
            push!(segment_diameter_cm, 0.1 * (child_seg.radii[j - 1] + child_seg.radii[j]))
            push!(segment_label, child_seg.name)
            push!(children[prev], vid)
            prev = vid
        end
    end

    return GrowthTree(name, vertices, parent_vertex, incoming_segment, children, segment_start, segment_end, segment_diameter_cm, segment_label, 1)
end

# ── Construction from seed point (for :seed_point mode) ──

"""
    growth_tree_from_seed(name, seed_point_cm)

Create a minimal GrowthTree with a single root vertex at the given seed point.
Branches will be grown from this root during the growth phase.
"""
function growth_tree_from_seed(name::String, seed_point_cm::SVector{3, Float64})
    vertices = SVector{3, Float64}[seed_point_cm]
    parent_vertex = Int[0]
    incoming_segment = Int[0]
    children = Vector{Int}[Int[]]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    return GrowthTree(name, vertices, parent_vertex, incoming_segment, children, segment_start, segment_end, segment_diameter_cm, segment_label, 1)
end
