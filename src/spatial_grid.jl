"""
    PointCloudGrid — spatial acceleration structure for nearest-neighbor queries.
"""

struct PointCloudGrid
    cell_size::Float64
    origin::NTuple{3, Float64}
    dims::NTuple{3, Int}
    cells::Dict{Int, Vector{Int}}
end

function _point_grid_index(grid::PointCloudGrid, x::Float64, y::Float64, z::Float64)
    ix = clamp(floor(Int, (x - grid.origin[1]) / grid.cell_size) + 1, 1, grid.dims[1])
    iy = clamp(floor(Int, (y - grid.origin[2]) / grid.cell_size) + 1, 1, grid.dims[2])
    iz = clamp(floor(Int, (z - grid.origin[3]) / grid.cell_size) + 1, 1, grid.dims[3])
    return ix + (iy - 1) * grid.dims[1] + (iz - 1) * grid.dims[1] * grid.dims[2]
end

function _point_grid_dims(lo::NTuple{3, Float64}, hi::NTuple{3, Float64}, cell_size::Float64)
    return (
        max(1, ceil(Int, (hi[1] - lo[1]) / cell_size)),
        max(1, ceil(Int, (hi[2] - lo[2]) / cell_size)),
        max(1, ceil(Int, (hi[3] - lo[3]) / cell_size)),
    )
end

function _default_point_grid_size(points::Matrix{Float64}, lo::NTuple{3, Float64}, hi::NTuple{3, Float64})
    dx = hi[1] - lo[1]
    dy = hi[2] - lo[2]
    dz = hi[3] - lo[3]
    extent = max(dx, dy, dz)
    bbox_vol = max(dx * dy * dz, 1e-12)
    nominal_spacing = cbrt(bbox_vol / max(size(points, 1), 1))
    return max(nominal_spacing * 2.5, extent / 64.0, 1e-4)
end

function _build_point_grid(points::Matrix{Float64}, lo::NTuple{3, Float64}, hi::NTuple{3, Float64})
    cell_size = _default_point_grid_size(points, lo, hi)
    dims = _point_grid_dims(lo, hi, cell_size)
    cells = Dict{Int, Vector{Int}}()
    grid = PointCloudGrid(cell_size, lo, dims, cells)
    for i in axes(points, 1)
        idx = _point_grid_index(grid, points[i, 1], points[i, 2], points[i, 3])
        push!(get!(cells, idx, Int[]), i)
    end
    return grid
end

function _surface_candidates(grid::PointCloudGrid, point; max_rings::Int=2, min_candidates::Int=32)
    x, y, z = point
    cx = clamp(floor(Int, (x - grid.origin[1]) / grid.cell_size) + 1, 1, grid.dims[1])
    cy = clamp(floor(Int, (y - grid.origin[2]) / grid.cell_size) + 1, 1, grid.dims[2])
    cz = clamp(floor(Int, (z - grid.origin[3]) / grid.cell_size) + 1, 1, grid.dims[3])
    candidates = Int[]
    for ring in 0:max_rings
        for dz in -ring:ring
            iz = cz + dz
            (iz < 1 || iz > grid.dims[3]) && continue
            for dy in -ring:ring
                iy = cy + dy
                (iy < 1 || iy > grid.dims[2]) && continue
                for dx in -ring:ring
                    ix = cx + dx
                    (ix < 1 || ix > grid.dims[1]) && continue
                    idx = ix + (iy - 1) * grid.dims[1] + (iz - 1) * grid.dims[1] * grid.dims[2]
                    haskey(grid.cells, idx) || continue
                    append!(candidates, grid.cells[idx])
                end
            end
        end
        length(candidates) >= min_candidates && break
    end
    return candidates
end

function _nearest_surface_info(points::Matrix{Float64}, normals::Matrix{Float64}, grid::PointCloudGrid, point)
    candidates = _surface_candidates(grid, point)
    if isempty(candidates)
        candidates = collect(axes(points, 1))
    end
    px, py, pz = point
    best_idx = first(candidates)
    best_d2 = Inf
    for idx in candidates
        dx = px - points[idx, 1]
        dy = py - points[idx, 2]
        dz = pz - points[idx, 3]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2
            best_d2 = d2
            best_idx = idx
        end
    end
    dx = px - points[best_idx, 1]
    dy = py - points[best_idx, 2]
    dz = pz - points[best_idx, 3]
    nx = normals[best_idx, 1]
    ny = normals[best_idx, 2]
    nz = normals[best_idx, 3]
    signed_dist = dx * nx + dy * ny + dz * nz
    euclid_dist = sqrt(best_d2)
    return best_idx, signed_dist, euclid_dist
end
