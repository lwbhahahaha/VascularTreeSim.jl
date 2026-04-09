"""
    SegmentSpatialIndex — acceleration structure for nearest-segment queries.

Partitions 3-D space into a uniform grid. Each cell stores the indices of segments
whose bounding box overlaps that cell. Turns O(N_segments) brute-force scans into
O(~k) lookups where k is the number of segments in nearby cells.
"""

struct SegmentSpatialIndex
    cell_size::Float64
    origin::NTuple{3, Float64}
    dims::NTuple{3, Int}
    cells::Vector{Vector{Int}}   # flat array, index = _seg_cell_linear_idx
    # Pre-extracted segment endpoint arrays (SoA layout for cache friendliness)
    ax::Vector{Float64}
    ay::Vector{Float64}
    az::Vector{Float64}
    bx::Vector{Float64}
    by::Vector{Float64}
    bz::Vector{Float64}
end

function _seg_cell_idx(idx::SegmentSpatialIndex, x::Float64, y::Float64, z::Float64)
    ix = clamp(floor(Int, (x - idx.origin[1]) / idx.cell_size) + 1, 1, idx.dims[1])
    iy = clamp(floor(Int, (y - idx.origin[2]) / idx.cell_size) + 1, 1, idx.dims[2])
    iz = clamp(floor(Int, (z - idx.origin[3]) / idx.cell_size) + 1, 1, idx.dims[3])
    return ix, iy, iz
end

@inline function _seg_linear_idx(idx::SegmentSpatialIndex, ix::Int, iy::Int, iz::Int)
    return ix + (iy - 1) * idx.dims[1] + (iz - 1) * idx.dims[1] * idx.dims[2]
end

"""
    build_segment_index(tree::GrowthTree; cell_size=0.0) -> SegmentSpatialIndex

Build a spatial index over all segments in the tree. If `cell_size` is 0,
auto-computes a reasonable cell size based on mean segment length.
"""
function build_segment_index(tree::GrowthTree; cell_size::Float64=0.0)
    nseg = length(tree.segment_start)
    nseg == 0 && return _empty_segment_index()

    # Extract endpoints into SoA arrays
    ax = Vector{Float64}(undef, nseg)
    ay = Vector{Float64}(undef, nseg)
    az = Vector{Float64}(undef, nseg)
    bx = Vector{Float64}(undef, nseg)
    by = Vector{Float64}(undef, nseg)
    bz = Vector{Float64}(undef, nseg)

    x_min = Inf; y_min = Inf; z_min = Inf
    x_max = -Inf; y_max = -Inf; z_max = -Inf
    total_len = 0.0

    for s in 1:nseg
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        ax[s] = a[1]; ay[s] = a[2]; az[s] = a[3]
        bx[s] = b[1]; by[s] = b[2]; bz[s] = b[3]
        x_min = min(x_min, a[1], b[1]); x_max = max(x_max, a[1], b[1])
        y_min = min(y_min, a[2], b[2]); y_max = max(y_max, a[2], b[2])
        z_min = min(z_min, a[3], b[3]); z_max = max(z_max, a[3], b[3])
        total_len += sqrt((b[1]-a[1])^2 + (b[2]-a[2])^2 + (b[3]-a[3])^2)
    end

    mean_len = total_len / nseg
    cs = cell_size > 0 ? cell_size : max(mean_len * 3.0, 0.01)

    pad = cs
    origin = (x_min - pad, y_min - pad, z_min - pad)
    dims = (
        max(1, ceil(Int, (x_max - x_min + 2pad) / cs)),
        max(1, ceil(Int, (y_max - y_min + 2pad) / cs)),
        max(1, ceil(Int, (z_max - z_min + 2pad) / cs)),
    )

    n_cells = dims[1] * dims[2] * dims[3]
    cells = [Int[] for _ in 1:n_cells]

    idx = SegmentSpatialIndex(cs, origin, dims, cells, ax, ay, az, bx, by, bz)

    # Insert each segment into all cells its bounding box overlaps
    for s in 1:nseg
        lo_x = min(ax[s], bx[s]); hi_x = max(ax[s], bx[s])
        lo_y = min(ay[s], by[s]); hi_y = max(ay[s], by[s])
        lo_z = min(az[s], bz[s]); hi_z = max(az[s], bz[s])

        ix0, iy0, iz0 = _seg_cell_idx(idx, lo_x, lo_y, lo_z)
        ix1, iy1, iz1 = _seg_cell_idx(idx, hi_x, hi_y, hi_z)

        for iz in iz0:iz1, iy in iy0:iy1, ix in ix0:ix1
            push!(cells[_seg_linear_idx(idx, ix, iy, iz)], s)
        end
    end

    return idx
end

function _empty_segment_index()
    return SegmentSpatialIndex(1.0, (0.0, 0.0, 0.0), (1, 1, 1),
        [Int[]], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
end

"""
    _indexed_segment_distance(idx, px, py, pz) -> Float64

Find the minimum distance from point (px,py,pz) to any segment in the index.
Searches expanding rings of cells until a guaranteed minimum is found.
"""
function _indexed_segment_distance(idx::SegmentSpatialIndex, px::Float64, py::Float64, pz::Float64)
    isempty(idx.ax) && return Inf

    cix, ciy, ciz = _seg_cell_idx(idx, px, py, pz)
    best_d2 = Inf
    cs = idx.cell_size

    # Search expanding rings; stop when ring distance exceeds current best
    for ring in 0:max(idx.dims[1], idx.dims[2], idx.dims[3])
        ring_dist = (ring - 1) * cs
        ring_dist * ring_dist > best_d2 && ring > 0 && break

        for dz in -ring:ring
            iz = ciz + dz
            (iz < 1 || iz > idx.dims[3]) && continue
            for dy in -ring:ring
                iy = ciy + dy
                (iy < 1 || iy > idx.dims[2]) && continue
                for dx in -ring:ring
                    # Only process cells on the current ring shell
                    (max(abs(dx), abs(dy), abs(dz)) != ring) && continue
                    ix = cix + dx
                    (ix < 1 || ix > idx.dims[1]) && continue

                    cell = idx.cells[_seg_linear_idx(idx, ix, iy, iz)]
                    for s in cell
                        d2 = _point_seg_dist2(idx, s, px, py, pz)
                        d2 < best_d2 && (best_d2 = d2)
                    end
                end
            end
        end
    end

    return sqrt(best_d2)
end

"""
    _point_seg_dist2(idx, s, px, py, pz) -> Float64

Squared distance from point to segment s (inlined for speed).
"""
@inline function _point_seg_dist2(idx::SegmentSpatialIndex, s::Int,
                                   px::Float64, py::Float64, pz::Float64)
    abx = idx.bx[s] - idx.ax[s]
    aby = idx.by[s] - idx.ay[s]
    abz = idx.bz[s] - idx.az[s]
    apx = px - idx.ax[s]
    apy = py - idx.ay[s]
    apz = pz - idx.az[s]
    denom = abx*abx + aby*aby + abz*abz
    if denom <= 1e-24
        return apx*apx + apy*apy + apz*apz
    end
    t = clamp((apx*abx + apy*aby + apz*abz) / denom, 0.0, 1.0)
    dx = apx - t * abx
    dy = apy - t * aby
    dz = apz - t * abz
    return dx*dx + dy*dy + dz*dz
end

"""
    update_segment_index!(idx, tree, seg_start_idx) -> SegmentSpatialIndex

Incrementally add new segments (from seg_start_idx to end) to the index.
If the new segments are outside the current bounding box, rebuilds entirely.
"""
function update_segment_index!(idx::SegmentSpatialIndex, tree::GrowthTree, seg_start_idx::Int)
    nseg = length(tree.segment_start)
    seg_start_idx > nseg && return idx

    # Check if new segments fit within existing grid
    needs_rebuild = false
    for s in seg_start_idx:nseg
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        for p in (a, b)
            ix = floor(Int, (p[1] - idx.origin[1]) / idx.cell_size) + 1
            iy = floor(Int, (p[2] - idx.origin[2]) / idx.cell_size) + 1
            iz = floor(Int, (p[3] - idx.origin[3]) / idx.cell_size) + 1
            if ix < 1 || ix > idx.dims[1] || iy < 1 || iy > idx.dims[2] || iz < 1 || iz > idx.dims[3]
                needs_rebuild = true
                break
            end
        end
        needs_rebuild && break
    end

    needs_rebuild && return build_segment_index(tree)

    # Incremental update: extend SoA arrays and insert into cells
    old_nseg = length(idx.ax)
    resize!(idx.ax, nseg); resize!(idx.ay, nseg); resize!(idx.az, nseg)
    resize!(idx.bx, nseg); resize!(idx.by, nseg); resize!(idx.bz, nseg)

    for s in seg_start_idx:nseg
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        idx.ax[s] = a[1]; idx.ay[s] = a[2]; idx.az[s] = a[3]
        idx.bx[s] = b[1]; idx.by[s] = b[2]; idx.bz[s] = b[3]

        lo_x = min(a[1], b[1]); hi_x = max(a[1], b[1])
        lo_y = min(a[2], b[2]); hi_y = max(a[2], b[2])
        lo_z = min(a[3], b[3]); hi_z = max(a[3], b[3])

        ix0, iy0, iz0 = _seg_cell_idx(idx, lo_x, lo_y, lo_z)
        ix1, iy1, iz1 = _seg_cell_idx(idx, hi_x, hi_y, hi_z)

        for iz in iz0:iz1, iy in iy0:iy1, ix in ix0:ix1
            push!(idx.cells[_seg_linear_idx(idx, ix, iy, iz)], s)
        end
    end

    return idx
end
