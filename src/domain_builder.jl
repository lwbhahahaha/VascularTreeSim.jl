"""
    VoxelShellDomain + flood-fill domain construction.

Config-driven: uses config.outer_surface and config.cavity_surfaces instead
of hardcoded constants.
"""

struct VoxelShellDomain
    mask::BitArray{3}
    origin_cm::SVector{3, Float64}
    spacing_cm::SVector{3, Float64}
    center_cm::SVector{3, Float64}
    outer_surface_points::Matrix{Float64}
    outer_surface_normals::Matrix{Float64}
    cavity_surface_points::Vector{Matrix{Float64}}
    cavity_surface_normals::Vector{Matrix{Float64}}
    outer_query_grid::PointCloudGrid
    cavity_query_grids::Vector{PointCloudGrid}
end

# ── Helpers ──

function _normalize_rows!(mat::Matrix{Float64})
    for i in axes(mat, 1)
        nx, ny, nz = mat[i, 1], mat[i, 2], mat[i, 3]
        nrm = sqrt(nx * nx + ny * ny + nz * nz)
        mat[i, 1] = nx / nrm
        mat[i, 2] = ny / nrm
        mat[i, 3] = nz / nrm
    end
    return mat
end

function voxel_center_cm(domain::VoxelShellDomain, i::Int, j::Int, k::Int)
    return domain.origin_cm + SVector((i - 0.5) * domain.spacing_cm[1], (j - 0.5) * domain.spacing_cm[2], (k - 0.5) * domain.spacing_cm[3])
end

function shell_distance_components(domain::VoxelShellDomain, point)
    _, outer_sd = _nearest_surface_info(domain.outer_surface_points, domain.outer_surface_normals, domain.outer_query_grid, point)
    nearest_cavity_sd = Inf
    for i in eachindex(domain.cavity_surface_points)
        _, cavity_sd = _nearest_surface_info(domain.cavity_surface_points[i], domain.cavity_surface_normals[i], domain.cavity_query_grids[i], point)
        nearest_cavity_sd = min(nearest_cavity_sd, cavity_sd)
    end
    return outer_sd, nearest_cavity_sd
end

function shell_midwall_cost(domain::VoxelShellDomain, point)
    outer_sd, cavity_sd = shell_distance_components(domain, point)
    if outer_sd > 0 || cavity_sd < 0
        return 1e6
    end
    d_outer = -outer_sd
    d_cavity = cavity_sd
    thickness = d_outer + d_cavity + 1e-9
    balance = abs(d_outer - d_cavity) / thickness
    return 1.0 + 1.5 * balance
end

# ── Point extraction ──

function voxel_mask_points(domain::VoxelShellDomain)
    pts = Matrix{Float64}(undef, count(domain.mask), 3)
    idx = 0
    for k in axes(domain.mask, 3), j in axes(domain.mask, 2), i in axes(domain.mask, 1)
        domain.mask[i, j, k] || continue
        idx += 1
        p = voxel_center_cm(domain, i, j, k)
        pts[idx, 1] = p[1]
        pts[idx, 2] = p[2]
        pts[idx, 3] = p[3]
    end
    return pts
end

function coverage_target_points(domain::VoxelShellDomain; stride::Int=1)
    collected = Float64[]
    for k in 1:stride:size(domain.mask, 3), j in 1:stride:size(domain.mask, 2), i in 1:stride:size(domain.mask, 1)
        domain.mask[i, j, k] || continue
        p = voxel_center_cm(domain, i, j, k)
        push!(collected, p[1], p[2], p[3])
    end
    n = length(collected) ÷ 3
    return n == 0 ? Matrix{Float64}(undef, 0, 3) : copy(reshape(collected, 3, n)')
end

function coverage_target_points_blockwise(domain::VoxelShellDomain; block_size::Int=3)
    block = max(block_size, 1)
    pts = Float64[]
    dims = size(domain.mask)
    for k0 in 1:block:dims[3], j0 in 1:block:dims[2], i0 in 1:block:dims[1]
        chosen = nothing
        best_cost = Inf
        ci = i0 + 0.5 * (min(i0 + block - 1, dims[1]) - i0)
        cj = j0 + 0.5 * (min(j0 + block - 1, dims[2]) - j0)
        ck = k0 + 0.5 * (min(k0 + block - 1, dims[3]) - k0)
        for k in k0:min(k0 + block - 1, dims[3]), j in j0:min(j0 + block - 1, dims[2]), i in i0:min(i0 + block - 1, dims[1])
            domain.mask[i, j, k] || continue
            c = (i - ci)^2 + (j - cj)^2 + (k - ck)^2
            if c < best_cost
                best_cost = c
                chosen = (i, j, k)
            end
        end
        chosen === nothing && continue
        p = voxel_center_cm(domain, chosen...)
        push!(pts, p[1], p[2], p[3])
    end
    n = length(pts) ÷ 3
    return n == 0 ? Matrix{Float64}(undef, 0, 3) : copy(reshape(pts, 3, n)')
end

# ── Surface sampling for domain construction ──

function _surface_sample_matrices_domain(surface::XCATNurbsSurface; n_u::Int, n_v::Int, orient_outward::Bool, coordinate_scale::Float64=1.0)
    pts, nrms, _, _ = xcat_sample_surface(surface; n_u=n_u, n_v=n_v, orient_outward=orient_outward)
    n = length(pts)
    points = Matrix{Float64}(undef, n, 3)
    normals = Matrix{Float64}(undef, n, 3)
    k = 1
    for j in axes(pts, 1), i in axes(pts, 2)
        p = pts[j, i]
        nrm = nrms[j, i]
        points[k, 1] = coordinate_scale * p[1]
        points[k, 2] = coordinate_scale * p[2]
        points[k, 3] = coordinate_scale * p[3]
        normals[k, 1] = nrm[1]
        normals[k, 2] = nrm[2]
        normals[k, 3] = nrm[3]
        k += 1
    end
    return points, normals
end

function _sample_surface_grid_domain(surface::XCATNurbsSurface; n_u::Int, n_v::Int, orient_outward::Bool, coordinate_scale::Float64=0.1)
    pts, nrms, _, _ = xcat_sample_surface(surface; n_u=n_u, n_v=n_v, orient_outward=orient_outward)
    nrows, ncols = size(pts)
    vertices = Vector{SVector{3, Float64}}(undef, nrows * ncols)
    normals = Vector{SVector{3, Float64}}(undef, nrows * ncols)
    idx = 1
    for j in 1:nrows, i in 1:ncols
        p = coordinate_scale .* pts[j, i]
        nrm = nrms[j, i]
        vertices[idx] = SVector{3, Float64}(p[1], p[2], p[3])
        normals[idx] = SVector{3, Float64}(nrm[1], nrm[2], nrm[3])
        idx += 1
    end
    return vertices, normals, nrows, ncols
end

# ── Voxelization helpers ──

function _point_to_voxel_index(origin::SVector{3, Float64}, spacing::SVector{3, Float64}, dims::NTuple{3, Int}, p::SVector{3, Float64})
    i = clamp(round(Int, (p[1] - origin[1]) / spacing[1] + 0.5), 1, dims[1])
    j = clamp(round(Int, (p[2] - origin[2]) / spacing[2] + 0.5), 1, dims[2])
    k = clamp(round(Int, (p[3] - origin[3]) / spacing[3] + 0.5), 1, dims[3])
    return i, j, k
end

function _mark_triangle_voxels!(mask::BitArray{3}, origin::SVector{3, Float64}, spacing::SVector{3, Float64},
                                a::SVector{3, Float64}, b::SVector{3, Float64}, c::SVector{3, Float64})
    max_edge = max(norm(b - a), norm(c - a), norm(c - b))
    nsteps = max(2, ceil(Int, max_edge / minimum(spacing) * 2.5))
    for iu in 0:nsteps
        u = iu / nsteps
        for iv in 0:(nsteps - iu)
            v = iv / nsteps
            w = 1.0 - u - v
            p = w * a + u * b + v * c
            i, j, k = _point_to_voxel_index(origin, spacing, size(mask), p)
            mask[i, j, k] = true
        end
    end
    return mask
end

function _dilate_mask(mask::BitArray{3}, radius::Int)
    radius <= 0 && return copy(mask)
    out = copy(mask)
    dims = size(mask)
    active = findall(mask)
    for idx in active
        i, j, k = Tuple(idx)
        for dk in -radius:radius, dj in -radius:radius, di in -radius:radius
            max(abs(di), abs(dj), abs(dk)) > radius && continue
            ii = i + di; jj = j + dj; kk = k + dk
            (1 <= ii <= dims[1] && 1 <= jj <= dims[2] && 1 <= kk <= dims[3]) || continue
            out[ii, jj, kk] = true
        end
    end
    return out
end

function _surface_wall_mask(surface::XCATNurbsSurface, origin::SVector{3, Float64}, spacing::SVector{3, Float64}, dims::NTuple{3, Int}; n_u::Int, n_v::Int, coordinate_scale::Float64=0.1, dilation_radius::Int=1)
    vertices, _, nrows, ncols = _sample_surface_grid_domain(surface; n_u=n_u, n_v=n_v, orient_outward=true, coordinate_scale=coordinate_scale)
    wall = falses(dims...)
    idx_fn(i, j) = (j - 1) * ncols + i
    for j in 1:(nrows - 1), i in 1:(ncols - 1)
        v11 = vertices[idx_fn(i, j)]
        v21 = vertices[idx_fn(i + 1, j)]
        v12 = vertices[idx_fn(i, j + 1)]
        v22 = vertices[idx_fn(i + 1, j + 1)]
        _mark_triangle_voxels!(wall, origin, spacing, v11, v21, v22)
        _mark_triangle_voxels!(wall, origin, spacing, v11, v22, v12)
    end
    return _dilate_mask(wall, dilation_radius)
end

# ── Flood fill ──

function _flood_fill_outside(wall::BitArray{3})
    dims = size(wall)
    outside = falses(dims...)
    queue = Vector{NTuple{3, Int}}()
    function try_push(i,j,k)
        if 1 <= i <= dims[1] && 1 <= j <= dims[2] && 1 <= k <= dims[3] && !wall[i,j,k] && !outside[i,j,k]
            outside[i,j,k] = true
            push!(queue, (i,j,k))
        end
    end
    for i in 1:dims[1], j in 1:dims[2]
        try_push(i,j,1); try_push(i,j,dims[3])
    end
    for i in 1:dims[1], k in 1:dims[3]
        try_push(i,1,k); try_push(i,dims[2],k)
    end
    for j in 1:dims[2], k in 1:dims[3]
        try_push(1,j,k); try_push(dims[1],j,k)
    end
    head = 1
    while head <= length(queue)
        i,j,k = queue[head]
        head += 1
        try_push(i+1,j,k); try_push(i-1,j,k)
        try_push(i,j+1,k); try_push(i,j-1,k)
        try_push(i,j,k+1); try_push(i,j,k-1)
    end
    return outside
end

function _seed_from_surface_centroid(surface::XCATNurbsSurface, origin::SVector{3, Float64}, spacing::SVector{3, Float64}, dims::NTuple{3, Int}; n_u::Int, n_v::Int, coordinate_scale::Float64=0.1)
    vertices, _, _, _ = _sample_surface_grid_domain(surface; n_u=n_u, n_v=n_v, orient_outward=true, coordinate_scale=coordinate_scale)
    c = reduce(+, vertices) / length(vertices)
    return _point_to_voxel_index(origin, spacing, dims, c)
end

function _surface_bbox(points::Matrix{Float64}, origin::SVector{3, Float64}, spacing::SVector{3, Float64}, dims::NTuple{3, Int}; pad_voxels::Int=2)
    lo = (minimum(@view points[:, 1]), minimum(@view points[:, 2]), minimum(@view points[:, 3]))
    hi = (maximum(@view points[:, 1]), maximum(@view points[:, 2]), maximum(@view points[:, 3]))
    i0, j0, k0 = _point_to_voxel_index(origin, spacing, dims, SVector(lo...))
    i1, j1, k1 = _point_to_voxel_index(origin, spacing, dims, SVector(hi...))
    return (
        max(1, i0 - pad_voxels), min(dims[1], i1 + pad_voxels),
        max(1, j0 - pad_voxels), min(dims[2], j1 + pad_voxels),
        max(1, k0 - pad_voxels), min(dims[3], k1 + pad_voxels),
    )
end

function _flood_fill_from_seed(wall::BitArray{3}, allowed::BitArray{3}, seed::NTuple{3, Int})
    dims = size(wall)
    region = falses(dims...)
    queue = Vector{NTuple{3, Int}}()
    i,j,k = seed
    if !(1 <= i <= dims[1] && 1 <= j <= dims[2] && 1 <= k <= dims[3]) || wall[i,j,k] || !allowed[i,j,k]
        return region
    end
    region[i,j,k] = true
    push!(queue, seed)
    head = 1
    while head <= length(queue)
        i,j,k = queue[head]
        head += 1
        for (ii,jj,kk) in ((i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k+1),(i,j,k-1))
            if 1 <= ii <= dims[1] && 1 <= jj <= dims[2] && 1 <= kk <= dims[3] && !wall[ii,jj,kk] && allowed[ii,jj,kk] && !region[ii,jj,kk]
                region[ii,jj,kk] = true
                push!(queue, (ii,jj,kk))
            end
        end
    end
    return region
end

function _midwall_seed(outer_interior::BitArray{3}, origin::SVector{3, Float64}, spacing::SVector{3, Float64},
                       outer_points::Matrix{Float64}, outer_normals::Matrix{Float64}, outer_grid::PointCloudGrid,
                       cavity_points::Vector{Matrix{Float64}}, cavity_normals::Vector{Matrix{Float64}}, cavity_grids::Vector{PointCloudGrid})
    dims = size(outer_interior)
    best_seed = nothing
    best_cost = Inf
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        outer_interior[i, j, k] || continue
        p = origin + SVector((i - 0.5) * spacing[1], (j - 0.5) * spacing[2], (k - 0.5) * spacing[3])
        _, outer_sd = _nearest_surface_info(outer_points, outer_normals, outer_grid, (p[1], p[2], p[3]))
        outer_sd <= 0.0 || continue
        nearest_cavity_sd = Inf
        for idx in eachindex(cavity_points)
            _, cavity_sd = _nearest_surface_info(cavity_points[idx], cavity_normals[idx], cavity_grids[idx], (p[1], p[2], p[3]))
            nearest_cavity_sd = min(nearest_cavity_sd, cavity_sd)
        end
        nearest_cavity_sd >= 0.0 || continue
        d_outer = -outer_sd
        d_cavity = nearest_cavity_sd
        thickness = d_outer + d_cavity + 1e-9
        cost = abs(d_outer - d_cavity) / thickness
        if cost < best_cost
            best_cost = cost
            best_seed = (i, j, k)
        end
    end
    if best_seed === nothing
        for idx in eachindex(outer_interior)
            outer_interior[idx] || continue
            return Tuple(idx)
        end
        return nothing
    end
    return best_seed
end

function _mapped_midwall_seed(outer_interior::BitArray{3}, origin::SVector{3, Float64}, spacing::SVector{3, Float64},
                              coarse_seed_cm::SVector{3, Float64}; search_radius_vox::Int=6)
    dims = size(outer_interior)
    ci = clamp(round(Int, (coarse_seed_cm[1] - origin[1]) / spacing[1] + 0.5), 1, dims[1])
    cj = clamp(round(Int, (coarse_seed_cm[2] - origin[2]) / spacing[2] + 0.5), 1, dims[2])
    ck = clamp(round(Int, (coarse_seed_cm[3] - origin[3]) / spacing[3] + 0.5), 1, dims[3])
    if outer_interior[ci, cj, ck]
        return (ci, cj, ck)
    end
    best = nothing
    best_d2 = typemax(Int)
    for dk in -search_radius_vox:search_radius_vox, dj in -search_radius_vox:search_radius_vox, di in -search_radius_vox:search_radius_vox
        ii = ci + di
        jj = cj + dj
        kk = ck + dk
        (1 <= ii <= dims[1] && 1 <= jj <= dims[2] && 1 <= kk <= dims[3]) || continue
        outer_interior[ii, jj, kk] || continue
        d2 = di * di + dj * dj + dk * dk
        if d2 < best_d2
            best_d2 = d2
            best = (ii, jj, kk)
        end
    end
    return best
end

# ── Main domain builder ──

function build_voxel_shell_domain_floodfill(outer_surface::XCATNurbsSurface, cavity_surfaces::Vector{XCATNurbsSurface}; coordinate_scale::Float64=0.1, voxel_spacing_cm::Float64=0.05, outer_samples::Tuple{Int, Int}=(96, 72), cavity_samples::Tuple{Int, Int}=(56, 40), dilation_radius::Int=1, coarse_seed_cm::Union{Nothing, SVector{3, Float64}}=nothing)
    println("[domain] build_voxel_shell_domain_floodfill spacing=$(voxel_spacing_cm) cm")
    flush(stdout)
    outer_points_s, outer_normals_s = _surface_sample_matrices_domain(outer_surface; n_u=outer_samples[1], n_v=outer_samples[2], orient_outward=true, coordinate_scale=coordinate_scale)
    cavity_points_s = Matrix{Float64}[]
    cavity_normals_s = Matrix{Float64}[]
    for surface in cavity_surfaces
        pts, nrms = _surface_sample_matrices_domain(surface; n_u=cavity_samples[1], n_v=cavity_samples[2], orient_outward=true, coordinate_scale=coordinate_scale)
        push!(cavity_points_s, pts)
        push!(cavity_normals_s, _normalize_rows!(copy(nrms)))
    end
    all_points = copy(outer_points_s)
    for pts in cavity_points_s
        all_points = vcat(all_points, pts)
    end
    lo_t = (minimum(@view all_points[:,1]), minimum(@view all_points[:,2]), minimum(@view all_points[:,3]))
    hi_t = (maximum(@view all_points[:,1]), maximum(@view all_points[:,2]), maximum(@view all_points[:,3]))
    outer_grid = _build_point_grid(outer_points_s, lo_t, hi_t)
    cavity_grids = [_build_point_grid(pts, lo_t, hi_t) for pts in cavity_points_s]
    lo = SVector(lo_t...) .- voxel_spacing_cm
    hi = SVector(hi_t...) .+ voxel_spacing_cm
    spacing = SVector(fill(voxel_spacing_cm, 3)...)
    dims = ntuple(d -> max(1, Int(ceil((hi[d] - lo[d]) / spacing[d]))), 3)
    println("[domain] dims=$(dims) outer_pts=$(size(outer_points_s,1)) cavities=$(length(cavity_surfaces))")
    flush(stdout)
    outer_wall = _surface_wall_mask(outer_surface, lo, spacing, dims; n_u=outer_samples[1], n_v=outer_samples[2], coordinate_scale=coordinate_scale, dilation_radius=dilation_radius)
    println("[domain] outer_wall ready vox=$(count(outer_wall))")
    flush(stdout)
    outside = _flood_fill_outside(outer_wall)
    println("[domain] outside floodfill ready vox=$(count(outside))")
    flush(stdout)
    outer_interior = .!outside .& .!outer_wall
    println("[domain] outer_interior ready vox=$(count(outer_interior))")
    flush(stdout)
    seed = coarse_seed_cm === nothing ?
        _midwall_seed(outer_interior, lo, spacing, outer_points_s, outer_normals_s, outer_grid, cavity_points_s, cavity_normals_s, cavity_grids) :
        _mapped_midwall_seed(outer_interior, lo, spacing, coarse_seed_cm)
    if seed !== nothing
        outer_interior = _flood_fill_from_seed(falses(dims...), outer_interior, seed)
        println("[domain] midwall component selected seed=$(seed) vox=$(count(outer_interior))")
        flush(stdout)
    end
    cavity_union = falses(dims...)
    cavity_margin_cm = voxel_spacing_cm
    for idx in eachindex(cavity_surfaces)
        surface = cavity_surfaces[idx]
        bbox = _surface_bbox(cavity_points_s[idx], lo, spacing, dims; pad_voxels=3)
        local_lo = lo + SVector((bbox[1] - 1) * spacing[1], (bbox[3] - 1) * spacing[2], (bbox[5] - 1) * spacing[3])
        local_dims = (
            bbox[2] - bbox[1] + 1,
            bbox[4] - bbox[3] + 1,
            bbox[6] - bbox[5] + 1,
        )
        println("[domain] cavity $(idx) local_dims=$(local_dims)")
        flush(stdout)
        cwall = _surface_wall_mask(surface, local_lo, spacing, local_dims; n_u=cavity_samples[1], n_v=cavity_samples[2], coordinate_scale=coordinate_scale, dilation_radius=dilation_radius)
        cseed = _seed_from_surface_centroid(surface, local_lo, spacing, local_dims; n_u=cavity_samples[1], n_v=cavity_samples[2], coordinate_scale=coordinate_scale)
        allowed = falses(local_dims...)
        for lk in 1:local_dims[3], lj in 1:local_dims[2], li in 1:local_dims[1]
            gi = bbox[1] + li - 1
            gj = bbox[3] + lj - 1
            gk = bbox[5] + lk - 1
            outer_interior[gi, gj, gk] || continue
            p = local_lo + SVector((li - 0.5) * spacing[1], (lj - 0.5) * spacing[2], (lk - 0.5) * spacing[3])
            _, cavity_sd = _nearest_surface_info(cavity_points_s[idx], cavity_normals_s[idx], cavity_grids[idx], (p[1], p[2], p[3]))
            cavity_sd <= cavity_margin_cm || continue
            allowed[li, lj, lk] = true
        end
        crec = _flood_fill_from_seed(cwall, allowed, cseed)
        println("[domain] cavity $(idx) region vox=$(count(crec))")
        flush(stdout)
        for lk in 1:local_dims[3], lj in 1:local_dims[2], li in 1:local_dims[1]
            crec[li, lj, lk] || continue
            gi = bbox[1] + li - 1
            gj = bbox[3] + lj - 1
            gk = bbox[5] + lk - 1
            cavity_union[gi, gj, gk] = true
        end
        GC.gc()
    end
    mask = outer_interior .& .!cavity_union
    println("[domain] final mask vox=$(count(mask))")
    flush(stdout)
    center = 0.5 .* (lo .+ hi)
    return VoxelShellDomain(mask, lo, spacing, center, outer_points_s, outer_normals_s, cavity_points_s, cavity_normals_s, outer_grid, cavity_grids)
end
