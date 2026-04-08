"""
    DomainGraph, Dijkstra shortest path, Catmull-Rom resampling, Laplacian smoothing.
"""

struct DomainGraph
    points::Vector{SVector{3, Float64}}
    neighbors::Vector{Vector{Int}}
    costs::Vector{Vector{Float64}}
end

struct GraphSpatialGrid
    points::Vector{SVector{3,Float64}}
    grid::PointCloudGrid
end

# ── k-nearest neighbor helpers ──

function _sample_k_nearest(points::Vector{SVector{3, Float64}}, i::Int, k::Int)
    pivot = points[i]
    dists = Tuple{Float64, Int}[]
    sizehint!(dists, max(length(points) - 1, 0))
    for j in eachindex(points)
        i == j && continue
        push!(dists, (norm(points[j] - pivot), j))
    end
    sort!(dists, by=first)
    return dists[1:min(k, length(dists))]
end

function _sample_k_nearest_grid(points::Vector{SVector{3, Float64}}, grid::PointCloudGrid, i::Int, k::Int; max_rings::Int=4)
    pivot = points[i]
    candidates = _surface_candidates(grid, (pivot[1], pivot[2], pivot[3]); max_rings=max_rings, min_candidates=max(32, 8 * k))
    if isempty(candidates)
        return _sample_k_nearest(points, i, k)
    end
    dists = Tuple{Float64, Int}[]
    sizehint!(dists, length(candidates))
    for j in candidates
        i == j && continue
        push!(dists, (norm(points[j] - pivot), j))
    end
    if length(dists) < k
        return _sample_k_nearest(points, i, k)
    end
    sort!(dists, by=first)
    return dists[1:min(k, length(dists))]
end

# ── Graph construction ──

function build_domain_graph(points_cm::Matrix{Float64}, domain; k::Int=10)
    pts = [SVector(points_cm[i, 1], points_cm[i, 2], points_cm[i, 3]) for i in axes(points_cm, 1)]
    point_matrix = Matrix{Float64}(undef, length(pts), 3)
    for i in eachindex(pts)
        point_matrix[i, 1] = pts[i][1]
        point_matrix[i, 2] = pts[i][2]
        point_matrix[i, 3] = pts[i][3]
    end
    pad = 1e-6
    lo = (
        minimum(point_matrix[:, 1]) - pad,
        minimum(point_matrix[:, 2]) - pad,
        minimum(point_matrix[:, 3]) - pad,
    )
    hi = (
        maximum(point_matrix[:, 1]) + pad,
        maximum(point_matrix[:, 2]) + pad,
        maximum(point_matrix[:, 3]) + pad,
    )
    grid = _build_point_grid(point_matrix, lo, hi)
    n = length(pts)
    neighbors = [Int[] for _ in 1:n]
    costs = [Float64[] for _ in 1:n]
    for i in 1:n
        for (d, j) in _sample_k_nearest_grid(pts, grid, i, k)
            mid = 0.5 .* (pts[i] + pts[j])
            base_cost = d * shell_midwall_cost(domain, (mid[1], mid[2], mid[3]))
            push!(neighbors[i], j)
            push!(costs[i], base_cost)
        end
    end
    return DomainGraph(pts, neighbors, costs)
end

function _build_graph_spatial_grid(graph::DomainGraph)
    mat = Matrix{Float64}(undef, length(graph.points), 3)
    for i in eachindex(graph.points)
        mat[i, 1] = graph.points[i][1]
        mat[i, 2] = graph.points[i][2]
        mat[i, 3] = graph.points[i][3]
    end
    pad = 1e-6
    lo = (minimum(mat[:, 1]) - pad, minimum(mat[:, 2]) - pad, minimum(mat[:, 3]) - pad)
    hi = (maximum(mat[:, 1]) + pad, maximum(mat[:, 2]) + pad, maximum(mat[:, 3]) + pad)
    grid = _build_point_grid(mat, lo, hi)
    return GraphSpatialGrid(graph.points, grid)
end

function _nearest_graph_index(sgrid::GraphSpatialGrid, point::SVector{3, Float64})
    candidates = _surface_candidates(sgrid.grid, (point[1], point[2], point[3]); max_rings=4, min_candidates=32)
    best_idx = 1
    best_d = Inf
    if !isempty(candidates)
        for i in candidates
            d = norm(point - sgrid.points[i])
            if d < best_d
                best_d = d
                best_idx = i
            end
        end
    else
        for (i, p) in enumerate(sgrid.points)
            d = norm(point - p)
            if d < best_d
                best_d = d
                best_idx = i
            end
        end
    end
    return best_idx, best_d
end

function _nearest_graph_index(graph::DomainGraph, point::SVector{3, Float64})
    best_idx = 1
    best_d = Inf
    for (i, p) in enumerate(graph.points)
        d = norm(point - p)
        if d < best_d
            best_d = d
            best_idx = i
        end
    end
    return best_idx, best_d
end

# ── Heap-based Dijkstra ──

function _heap_sift_up!(heap::Vector{Tuple{Float64,Int}}, i::Int)
    while i > 1
        p = i >> 1
        heap[p][1] <= heap[i][1] && break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
    end
end

function _heap_sift_down!(heap::Vector{Tuple{Float64,Int}}, i::Int)
    n = length(heap)
    while true
        s = i
        l = 2i
        r = 2i + 1
        l <= n && heap[l][1] < heap[s][1] && (s = l)
        r <= n && heap[r][1] < heap[s][1] && (s = r)
        s == i && break
        heap[s], heap[i] = heap[i], heap[s]
        i = s
    end
end

function _heap_push!(heap::Vector{Tuple{Float64,Int}}, item::Tuple{Float64,Int})
    push!(heap, item)
    _heap_sift_up!(heap, length(heap))
end

function _heap_pop!(heap::Vector{Tuple{Float64,Int}})
    item = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        _heap_sift_down!(heap, 1)
    end
    return item
end

function _shortest_path(graph::DomainGraph, source::Int, target::Int)
    source == target && return [source]
    n = length(graph.points)
    dist = fill(Inf, n)
    prev = fill(0, n)
    dist[source] = 0.0
    heap = Tuple{Float64,Int}[(0.0, source)]
    sizehint!(heap, min(n, 4096))
    while !isempty(heap)
        d_u, u = _heap_pop!(heap)
        d_u > dist[u] && continue
        u == target && break
        for (v, c) in zip(graph.neighbors[u], graph.costs[u])
            alt = dist[u] + c
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
                _heap_push!(heap, (alt, v))
            end
        end
    end
    if prev[target] == 0
        return [source, target]
    end
    path = Int[]
    u = target
    while u != 0
        push!(path, u)
        u = prev[u]
    end
    reverse!(path)
    return path
end

# ── Path processing: subsample, dedupe, smooth, Catmull-Rom ──

function _subsample_path(points::Vector{SVector{3, Float64}}; max_nodes::Int=12)
    n = length(points)
    n <= max_nodes && return points
    idxs = unique(round.(Int, range(1, n; length=max_nodes)))
    return points[idxs]
end

function _point_in_domain(domain, p::SVector{3, Float64})
    dims = size(domain.mask)
    i = floor(Int, (p[1] - domain.origin_cm[1]) / domain.spacing_cm[1]) + 1
    j = floor(Int, (p[2] - domain.origin_cm[2]) / domain.spacing_cm[2]) + 1
    k = floor(Int, (p[3] - domain.origin_cm[3]) / domain.spacing_cm[3]) + 1
    return 1 <= i <= dims[1] && 1 <= j <= dims[2] && 1 <= k <= dims[3] && domain.mask[i, j, k]
end

function _dedupe_path(points::Vector{SVector{3, Float64}}; tol::Float64=1e-8)
    isempty(points) && return points
    out = SVector{3, Float64}[points[1]]
    for p in Iterators.drop(points, 1)
        norm(p - out[end]) > tol && push!(out, p)
    end
    return out
end

function _smooth_path_in_domain(points::Vector{SVector{3, Float64}}, domain; passes::Int=3)
    length(points) <= 2 && return points
    current = copy(points)
    for _ in 1:passes
        next_pts = copy(current)
        for i in 2:length(current)-1
            candidate = 0.25 * current[i - 1] + 0.5 * current[i] + 0.25 * current[i + 1]
            if _point_in_domain(domain, candidate)
                next_pts[i] = candidate
            end
        end
        current = next_pts
    end
    return current
end

function _catmull_rom_point(p0::SVector{3,Float64}, p1::SVector{3,Float64},
                            p2::SVector{3,Float64}, p3::SVector{3,Float64}, t::Float64)
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2.0 .* p1) .+ (-p0 .+ p2) .* t .+
                   (2.0 .* p0 .- 5.0 .* p1 .+ 4.0 .* p2 .- p3) .* t2 .+
                   (-p0 .+ 3.0 .* p1 .- 3.0 .* p2 .+ p3) .* t3)
end

function _catmull_rom_resample(points::Vector{SVector{3,Float64}}; segments_per_span::Int=4)
    n = length(points)
    n <= 2 && return copy(points)
    result = SVector{3,Float64}[points[1]]
    for i in 1:n-1
        p0 = i > 1   ? points[i-1] : 2.0 .* points[i]   .- points[i+1]
        p1 = points[i]
        p2 = points[i+1]
        p3 = i < n-1  ? points[i+2] : 2.0 .* points[i+1] .- points[i]
        for j in 1:segments_per_span
            t = j / segments_per_span
            push!(result, _catmull_rom_point(p0, p1, p2, p3, t))
        end
    end
    return result
end

function _prepare_branch_path(points::Vector{SVector{3, Float64}}, domain;
                              max_nodes::Int=12, smooth_passes::Int=4, spline_density::Int=4)
    pts = _dedupe_path(points)
    length(pts) <= 2 && return pts
    waypoints = _subsample_path(pts; max_nodes=max(max_nodes, 6))
    waypoints = _dedupe_path(waypoints)
    length(waypoints) <= 2 && return waypoints
    dense = _catmull_rom_resample(waypoints; segments_per_span=spline_density)
    dense = filter(p -> _point_in_domain(domain, p), dense)
    if length(dense) < 2
        return _dedupe_path(waypoints)
    end
    dense = _smooth_path_in_domain(dense, domain; passes=smooth_passes)
    dense = _subsample_path(dense; max_nodes=max_nodes)
    return _dedupe_path(dense)
end

function _jitter_points_in_domain(points::Matrix{Float64}, domain; max_jitter_cm::Float64=0.0, max_tries::Int=6)
    max_jitter_cm <= 0 && return points
    out = copy(points)
    rng = Random.default_rng()
    for i in axes(points, 1)
        base = SVector(points[i, 1], points[i, 2], points[i, 3])
        accepted = false
        for _ in 1:max_tries
            jitter = SVector(
                (rand(rng) - 0.5) * 2 * max_jitter_cm,
                (rand(rng) - 0.5) * 2 * max_jitter_cm,
                (rand(rng) - 0.5) * 2 * max_jitter_cm,
            )
            candidate = base + jitter
            if _point_in_domain(domain, candidate)
                out[i, 1] = candidate[1]
                out[i, 2] = candidate[2]
                out[i, 3] = candidate[3]
                accepted = true
                break
            end
        end
        accepted || continue
    end
    return out
end
