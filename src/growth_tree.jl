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
    is_xcat::Vector{Bool}                  # per-segment: true = from XCAT data; false = grown
    subtree_terminal_count::Vector{Int}    # per-vertex: number of grown terminals downstream
    terminal_diameter_cm::Float64          # tree-wide terminal (leaf) diameter
    root_vertex::Int
    root_diameter_cm::Float64              # locked diameter of root (ostium) segment
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
    parent_children = tree.children[start_v]
    idx = findfirst(==(end_v), parent_children)
    idx === nothing && return end_v
    push!(tree.vertices, point)
    mid_v = length(tree.vertices)
    push!(tree.parent_vertex, start_v)
    push!(tree.incoming_segment, seg_id)
    push!(tree.children, Int[end_v])
    push!(tree.subtree_terminal_count, tree.subtree_terminal_count[end_v])
    parent_children[idx] = mid_v
    tree.parent_vertex[end_v] = mid_v
    old_d = tree.segment_diameter_cm[seg_id]
    old_label = tree.segment_label[seg_id]
    old_xcat = tree.is_xcat[seg_id]
    tree.segment_end[seg_id] = mid_v
    push!(tree.segment_start, mid_v)
    push!(tree.segment_end, end_v)
    push!(tree.segment_diameter_cm, old_d)
    push!(tree.segment_label, old_label)
    push!(tree.is_xcat, old_xcat)
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

"""
    _can_add_terminal_at_anchor(tree, anchor_vertex; gamma=3.0) -> Bool

Check whether the tree has capacity for one more grown terminal, based on the
locked root (ostium) diameter budget:
    total_terminals + 1 ≤ (d_root / d_term)^γ

Since ALL segment diameters (including XCAT) are now Murray-derived from
subtree_terminal_count, Murray's law is automatically satisfied at every
junction. The only fixed constraint is the root segment diameter.
"""
function _can_add_terminal_at_anchor(tree::GrowthTree, anchor_vertex::Int; gamma::Float64=3.0)
    tree.root_diameter_cm <= 0.0 && return true  # seed tree: no budget limit
    max_terminals = (tree.root_diameter_cm / tree.terminal_diameter_cm)^gamma
    current = tree.subtree_terminal_count[tree.root_vertex]
    return current + 1 <= max_terminals + 0.5  # +0.5 for float tolerance
end

"""
    _strict_murray_propagate!(tree, start_vertex; gamma=3.0, added=1)

Walk from start_vertex up to root. Increment subtree_terminal_count by `added`
at every ancestor (including start_vertex itself). Recompute diameter for ALL
segments (both XCAT and grown) as:
    d = d_term × N^(1/γ)
where N is the updated subtree terminal count at that segment's end vertex.

All segment diameters are purely Murray-derived — no segment is locked.
The `root_diameter_cm` field is only used as a budget ceiling in
`_can_add_terminal_at_anchor`, not for diameter assignment.
"""
function _strict_murray_propagate!(tree::GrowthTree, start_vertex::Int; gamma::Float64=3.0, added::Int=1)
    v = start_vertex
    while v != 0
        tree.subtree_terminal_count[v] += added
        seg = tree.incoming_segment[v]
        if seg != 0
            n = tree.subtree_terminal_count[v]
            tree.segment_diameter_cm[seg] = tree.terminal_diameter_cm * n^(1.0 / gamma)
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

function _add_branch_path!(tree::GrowthTree, anchor_vertex::Int, path_points::Vector{SVector{3, Float64}}; gamma::Float64=3.0, max_branch_length_cm::Float64=Inf, max_segment_length_cm::Float64=0.1)
    isempty(path_points) && return false

    # Murray budget check FIRST — refuse if any XCAT ancestor would over-allocate.
    _can_add_terminal_at_anchor(tree, anchor_vertex; gamma=gamma) || return false

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

    d_term = tree.terminal_diameter_cm
    prev = anchor_vertex
    for p in local_pts
        push!(tree.vertices, p)
        vid = length(tree.vertices)
        push!(tree.parent_vertex, prev)
        push!(tree.incoming_segment, length(tree.segment_start) + 1)
        push!(tree.children, Int[])
        push!(tree.subtree_terminal_count, 1)  # this new vertex's subtree currently holds the new terminal
        push!(tree.segment_start, prev)
        push!(tree.segment_end, vid)
        push!(tree.segment_diameter_cm, d_term)  # all new grown segments at terminal diameter
        push!(tree.segment_label, "grown")
        push!(tree.is_xcat, false)
        push!(tree.children[prev], vid)
        prev = vid
    end

    # Cascade subtree_terminal_count upward; for grown segments along the path,
    # recompute diameter via strict Murray d = d_term × N^(1/γ).
    _strict_murray_propagate!(tree, anchor_vertex; gamma=gamma, added=1)
    return true
end

# ── Construction from XCAT centerlines ──

function growth_tree_from_xcat(name::String, tree::XCATCenterlineTree; terminal_diameter_cm::Float64=0.004)
    vertices = SVector{3, Float64}[]
    parent_vertex = Int[]
    incoming_segment = Int[]
    children = Vector{Int}[]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = Int[]
    point_map = Dict{Tuple{String, Int}, Int}()

    root = tree.segments[tree.root_segment]
    for (i, p) in enumerate(root.centers)
        push!(vertices, 0.1 .* p)
        push!(parent_vertex, i == 1 ? 0 : i - 1)
        push!(incoming_segment, i == 1 ? 0 : length(segment_start) + 1)
        push!(children, Int[])
        push!(subtree_terminal_count, 0)
        point_map[(root.name, i)] = i
        if i > 1
            push!(segment_start, i - 1)
            push!(segment_end, i)
            push!(segment_diameter_cm, 0.1 * (root.radii[i - 1] + root.radii[i]))
            push!(segment_label, root.name)
            push!(is_xcat, true)
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
            push!(subtree_terminal_count, 0)
            point_map[(child_seg.name, j)] = vid
            push!(segment_start, prev)
            push!(segment_end, vid)
            push!(segment_diameter_cm, 0.1 * (child_seg.radii[j - 1] + child_seg.radii[j]))
            push!(segment_label, child_seg.name)
            push!(is_xcat, true)
            push!(children[prev], vid)
            prev = vid
        end
    end

    # Root (ostium) diameter = maximum XCAT diameter in the chain (locked).
    # Using max instead of seg1 because XCAT chains can widen distally due to
    # segmentation artifacts; the peak diameter represents the vessel's true capacity.
    root_d = isempty(segment_diameter_cm) ? 0.0 : maximum(segment_diameter_cm)
    return GrowthTree(name, vertices, parent_vertex, incoming_segment, children, segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat, subtree_terminal_count, terminal_diameter_cm, 1, root_d)
end

# ── Construction from seed point (for :seed_point mode) ──

"""
    growth_tree_from_seed(name, seed_point_cm)

Create a minimal GrowthTree with a single root vertex at the given seed point.
Branches will be grown from this root during the growth phase.
"""
function growth_tree_from_seed(name::String, seed_point_cm::SVector{3, Float64}; terminal_diameter_cm::Float64=0.004)
    vertices = SVector{3, Float64}[seed_point_cm]
    parent_vertex = Int[0]
    incoming_segment = Int[0]
    children = Vector{Int}[Int[]]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = Int[0]
    return GrowthTree(name, vertices, parent_vertex, incoming_segment, children, segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat, subtree_terminal_count, terminal_diameter_cm, 1, 0.0)
end

# ── Post-growth junction smoothing ──

"""
    smooth_junction_taper!(tree; gamma=3.0)

Post-processing pass: at every XCAT vertex where grown branches attach, compute
the Murray-law residual capacity and use it to set a smooth top-down taper for
the grown subtree. This eliminates the visual "sudden diameter drop" at
XCAT↔grown junctions.

Algorithm:
1. For each XCAT vertex V with incoming XCAT segment (d_xcat):
   - Compute xcat_child_demand = Σ d^γ of XCAT children
   - Murray residual = d_xcat^γ - xcat_child_demand  (flow budget for grown)
   - Distribute residual among grown children by subtree_terminal_count
2. For each grown child subtree rooted at C:
   - junction_d = allocated share of Murray residual
   - bottom_up_d = current Murray-strict value
   - If junction_d > bottom_up_d, apply exponentially decaying boost from
     junction_d (k=0) to 1.0× at leaves (k=D)
"""
function smooth_junction_taper!(tree::GrowthTree; gamma::Float64=3.0)
    nv = length(tree.vertices)
    for v in 1:nv
        xcat_in = tree.incoming_segment[v]
        has_xcat_context = (xcat_in != 0 && tree.is_xcat[xcat_in])
        has_xcat_context || continue

        # Collect grown children at this XCAT vertex
        grown_children = Tuple{Int, Int}[]  # (child_vertex, child_segment)
        for c in tree.children[v]
            cseg = tree.incoming_segment[c]
            (cseg == 0 || tree.is_xcat[cseg]) && continue
            push!(grown_children, (c, cseg))
        end
        isempty(grown_children) && continue

        # Murray residual at this vertex
        d_in_gamma = tree.segment_diameter_cm[xcat_in]^gamma
        xcat_child_demand = 0.0
        for c in tree.children[v]
            cseg = tree.incoming_segment[c]
            (cseg == 0 || !tree.is_xcat[cseg]) && continue
            xcat_child_demand += tree.segment_diameter_cm[cseg]^gamma
        end
        residual = max(0.0, d_in_gamma - xcat_child_demand)
        residual <= 0.0 && continue

        # Distribute residual proportionally by subtree terminal count
        total_tc = sum(max(1, tree.subtree_terminal_count[c]) for (c, _) in grown_children)
        total_tc == 0 && continue

        for (child_v, child_seg) in grown_children
            tc = max(1, tree.subtree_terminal_count[child_v])
            share = (tc / total_tc) * residual
            junction_d = share^(1.0 / gamma)
            current_d = tree.segment_diameter_cm[child_seg]
            junction_d <= current_d && continue

            max_depth = _subtree_max_depth(tree, child_v)
            max_depth <= 0 && continue
            boost = junction_d / current_d
            _apply_junction_boost!(tree, child_v, boost, 0, max_depth)
        end
    end
    return nothing
end

"""Compute max depth of the grown subtree rooted at vertex v."""
function _subtree_max_depth(tree::GrowthTree, v::Int)
    best = 0
    for c in tree.children[v]
        cseg = tree.incoming_segment[c]
        (cseg == 0 || tree.is_xcat[cseg]) && continue
        best = max(best, 1 + _subtree_max_depth(tree, c))
    end
    return best
end

"""
Apply exponentially decaying diameter boost through a grown subtree.
At depth k / max_depth D: effective_boost = boost^((D-k)/D).
boost at junction (k=0), 1.0× at leaves (k=D).
"""
function _apply_junction_boost!(tree::GrowthTree, v::Int, boost::Float64,
                                depth::Int, max_depth::Int)
    seg = tree.incoming_segment[v]
    if seg != 0 && !tree.is_xcat[seg]
        t = clamp((max_depth - depth) / max_depth, 0.0, 1.0)
        tree.segment_diameter_cm[seg] *= boost^t
    end
    for c in tree.children[v]
        cseg = tree.incoming_segment[c]
        (cseg == 0 || tree.is_xcat[cseg]) && continue
        _apply_junction_boost!(tree, c, boost, depth + 1, max_depth)
    end
end

# ── Post-growth terminal subdivision ──

"""
    subdivide_terminals!(tree; target_diameter_cm, gamma=3.0, branch_half_angle=0.4, ld_ratio=12.0)

Recursively bifurcate each terminal of `tree` until segment diameter reaches
`target_diameter_cm`. Each bifurcation is symmetric (Murray's law: d^γ = 2 d_child^γ),
producing two children at ±`branch_half_angle` radians from the parent direction.

After subdivision, ALL diameters in the tree are recomputed from the new terminal
counts using `d = target_diameter_cm × N^(1/γ)`, so Murray's law is satisfied
at every junction in the entire tree.
"""
function subdivide_terminals!(tree::GrowthTree;
        target_diameter_cm::Float64,
        gamma::Float64=3.0,
        branch_half_angle::Float64=0.4,
        ld_ratio::Float64=12.0,
        domain::Union{Nothing, VoxelShellDomain}=nothing)

    target_diameter_cm >= tree.terminal_diameter_cm && return nothing

    # Collect current terminals before we start adding new ones
    terminals = _branch_terminals(tree)
    n_before = length(tree.vertices)
    rng = Random.MersenneTwister(hash(tree.name))
    clipped = Ref(0)  # count of sub-branches rejected for leaving domain

    for tip_v in terminals
        seg = tree.incoming_segment[tip_v]
        seg == 0 && continue
        d_cm = tree.segment_diameter_cm[seg]
        d_cm <= target_diameter_cm && continue

        # Direction from parent → tip
        pv = tree.parent_vertex[tip_v]
        dir = tree.vertices[tip_v] - tree.vertices[pv]
        len = norm(dir)
        len < 1e-12 && continue
        dir = dir / len

        _subdivide_recursive!(tree, tip_v, d_cm, dir, target_diameter_cm, gamma,
                              branch_half_angle, ld_ratio, rng, domain, clipped)
    end

    # Recompute terminal counts and diameters for the ENTIRE tree
    _recompute_all_murray!(tree; target_diameter_cm=target_diameter_cm, gamma=gamma)

    n_after = length(tree.vertices)
    n_new_terminals = length(_branch_terminals(tree))
    clip_msg = domain === nothing ? "" : " (clipped $(clipped[]) out-of-domain sub-branches)"
    println("[subdivide] $(tree.name): $(length(terminals)) terminals → $(n_new_terminals) sub-terminals, $(n_after - n_before) new vertices$(clip_msg)")
    flush(stdout)
    return nothing
end

function _subdivide_recursive!(tree::GrowthTree, vertex::Int, parent_d_cm::Float64,
        direction::SVector{3,Float64}, target_d_cm::Float64, gamma::Float64,
        half_angle::Float64, ld_ratio::Float64, rng::Random.AbstractRNG,
        domain::Union{Nothing, VoxelShellDomain}=nothing,
        clipped::Union{Nothing, Ref{Int}}=nothing)

    child_d_cm = parent_d_cm / 2.0^(1.0 / gamma)
    child_d_cm <= target_d_cm && return

    # Two perpendicular vectors to direction
    u, v = _perp_pair(direction)
    t = tan(half_angle)
    seg_len = child_d_cm * ld_ratio
    origin = tree.vertices[vertex]

    # Try multiple random rotations of bifurcation plane; pick the one where the
    # most children land inside the domain. Prevents cascading subtree loss when
    # a single random plane happens to straddle the domain boundary.
    dir1 = SVector(0.0, 0.0, 0.0)
    dir2 = SVector(0.0, 0.0, 0.0)
    best_score = -1
    max_attempts = domain === nothing ? 1 : 8
    for attempt in 1:max_attempts
        theta = rand(rng) * 2π
        perp = cos(theta) * u + sin(theta) * v
        c1 = LinearAlgebra.normalize(direction + t * perp)
        c2 = LinearAlgebra.normalize(direction - t * perp)
        score = 2
        if domain !== nothing
            score = (point_in_domain(domain, origin + seg_len * c1) ? 1 : 0) +
                    (point_in_domain(domain, origin + seg_len * c2) ? 1 : 0)
        end
        if score > best_score
            best_score = score
            dir1, dir2 = c1, c2
            score == 2 && break
        end
    end

    for child_dir in (dir1, dir2)
        new_pos = origin + seg_len * child_dir

        # Domain clip: reject sub-branches that leave the myocardial shell.
        # The remaining Murray recompute handles diameter adjustments automatically.
        if domain !== nothing && !point_in_domain(domain, new_pos)
            clipped !== nothing && (clipped[] += 1)
            continue
        end

        push!(tree.vertices, new_pos)
        vid = length(tree.vertices)
        push!(tree.parent_vertex, vertex)
        push!(tree.incoming_segment, length(tree.segment_start) + 1)
        push!(tree.children, Int[])
        push!(tree.subtree_terminal_count, 0)  # will be recomputed later
        push!(tree.segment_start, vertex)
        push!(tree.segment_end, vid)
        push!(tree.segment_diameter_cm, child_d_cm)  # temporary, recomputed later
        push!(tree.segment_label, "subdivided")
        push!(tree.is_xcat, false)
        push!(tree.children[vertex], vid)

        _subdivide_recursive!(tree, vid, child_d_cm, child_dir, target_d_cm, gamma,
                              half_angle, ld_ratio, rng, domain, clipped)
    end
end

function _perp_pair(dir::SVector{3,Float64})
    ref = abs(dir[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    u = LinearAlgebra.normalize(cross(dir, ref))
    v = cross(dir, u)
    return u, v
end

"""
    _recompute_all_murray!(tree; target_diameter_cm, gamma)

Bottom-up pass: recompute subtree_terminal_count and segment diameters for the
entire tree using `target_diameter_cm` as the new terminal diameter.
"""
function _recompute_all_murray!(tree::GrowthTree; target_diameter_cm::Float64, gamma::Float64=3.0)
    nv = length(tree.vertices)

    # Reset all counts
    fill!(tree.subtree_terminal_count, 0)

    # Bottom-up: leaves first. Use topological sort via DFS post-order.
    visited = falses(nv)
    order = Int[]
    sizehint!(order, nv)

    # Iterative DFS post-order from root
    stack = [(tree.root_vertex, false)]
    while !isempty(stack)
        v, processed = pop!(stack)
        if processed
            push!(order, v)
            continue
        end
        visited[v] && continue
        visited[v] = true
        push!(stack, (v, true))
        for c in tree.children[v]
            visited[c] || push!(stack, (c, false))
        end
    end

    # Process in post-order (leaves first)
    for v in order
        if isempty(tree.children[v])
            tree.subtree_terminal_count[v] = 1
        else
            tree.subtree_terminal_count[v] = sum(tree.subtree_terminal_count[c] for c in tree.children[v])
        end
        seg = tree.incoming_segment[v]
        if seg != 0
            n = tree.subtree_terminal_count[v]
            tree.segment_diameter_cm[seg] = target_diameter_cm * n^(1.0 / gamma)
        end
    end

    # Update tree's terminal diameter
    tree.terminal_diameter_cm = target_diameter_cm
    return nothing
end
