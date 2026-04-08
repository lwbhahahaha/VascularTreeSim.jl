"""
Example: grow a vascular tree in a synthetic cube shell domain.

Usage:
    julia --project=.. examples/synthetic_cube.jl

This demonstrates the seed-point workflow (no XCAT NRB required).
"""

using VascularTreeSim
using StaticArrays
using Random

Random.seed!(42)

# ── Step 1: Define domain geometry ──
# Cube shell: outer 6×6×6 cm, inner 4×4×4 cm, centered at (3,3,3)

spacing = 0.12  # cm voxel spacing

domain_pts = SVector{3,Float64}[]
for x in 0.0:spacing:6.0, y in 0.0:spacing:6.0, z in 0.0:spacing:6.0
    p = SVector(x, y, z)
    d = p - SVector(3.0, 3.0, 3.0)
    in_outer = all(abs.(d) .<= 3.0)
    in_inner = all(abs.(d) .< 2.0)
    in_outer && !in_inner && push!(domain_pts, p)
end
println("Domain: $(length(domain_pts)) points")

# ── Step 2: Build VoxelShellDomain (manually for synthetic geometry) ──

origin_cm = SVector(-spacing, -spacing, -spacing)
voxel_spacing = SVector(spacing, spacing, spacing)
dims = ntuple(d -> max(1, Int(ceil((6.0 + 2spacing) / spacing))), 3)
center = SVector(3.0, 3.0, 3.0)

mask = falses(dims...)
for p in domain_pts
    i = clamp(round(Int, (p[1] - origin_cm[1]) / spacing + 0.5), 1, dims[1])
    j = clamp(round(Int, (p[2] - origin_cm[2]) / spacing + 0.5), 1, dims[2])
    k = clamp(round(Int, (p[3] - origin_cm[3]) / spacing + 0.5), 1, dims[3])
    mask[i, j, k] = true
end

# Surface points (outer cube faces)
outer_pts = Float64[]; outer_nrm = Float64[]
for x in 0.0:0.3:6.0, y in 0.0:0.3:6.0
    for (z, nz) in [(0.0,-1.0),(6.0,1.0)]
        push!(outer_pts, x, y, z); push!(outer_nrm, 0.0, 0.0, nz)
    end
    for (z, ny) in [(0.0,-1.0),(6.0,1.0)]
        push!(outer_pts, x, z, y); push!(outer_nrm, 0.0, ny, 0.0)
    end
    for (z, nx) in [(0.0,-1.0),(6.0,1.0)]
        push!(outer_pts, z, x, y); push!(outer_nrm, nx, 0.0, 0.0)
    end
end
no = length(outer_pts) ÷ 3
outer_points = reshape(outer_pts, 3, no)' |> collect
outer_normals = reshape(outer_nrm, 3, no)' |> collect

# Cavity surface (inner cube faces)
cav_pts = Float64[]; cav_nrm = Float64[]
for x in 1.0:0.3:5.0, y in 1.0:0.3:5.0
    for (z, nz) in [(1.0,-1.0),(5.0,1.0)]
        push!(cav_pts, x, y, z); push!(cav_nrm, 0.0, 0.0, nz)
    end
    for (z, ny) in [(1.0,-1.0),(5.0,1.0)]
        push!(cav_pts, x, z, y); push!(cav_nrm, 0.0, ny, 0.0)
    end
    for (z, nx) in [(1.0,-1.0),(5.0,1.0)]
        push!(cav_pts, z, x, y); push!(cav_nrm, nx, 0.0, 0.0)
    end
end
nc = length(cav_pts) ÷ 3
cavity_points = reshape(cav_pts, 3, nc)' |> collect
cavity_normals = reshape(cav_nrm, 3, nc)' |> collect

lo_t = (-0.1, -0.1, -0.1); hi_t = (6.1, 6.1, 6.1)
outer_grid = VascularTreeSim._build_point_grid(outer_points, lo_t, hi_t)
cavity_grid = VascularTreeSim._build_point_grid(cavity_points, lo_t, hi_t)

domain = VoxelShellDomain(mask, origin_cm, voxel_spacing, center,
    outer_points, outer_normals, [cavity_points], [cavity_normals],
    outer_grid, [cavity_grid])

# ── Step 3: Create seed tree and grow ──

tree = growth_tree_from_seed("VTree", SVector(3.0, 0.5, 3.0))
trees = Dict("VTree" => tree)

# Build coverage/graph point matrices
pts_mat = Matrix{Float64}(undef, length(domain_pts), 3)
for (i, p) in enumerate(domain_pts)
    pts_mat[i, 1] = p[1]; pts_mat[i, 2] = p[2]; pts_mat[i, 3] = p[3]
end
cov_pts = pts_mat[1:2:end, :]

graph, territories, stats = grow_trees_mcp!(trees, domain;
    coverage_points_cm=cov_pts,
    graph_points_cm=copy(cov_pts),
    max_new_branches_per_tree=300,
    graph_neighbors=12,
    frontier_batch=8,
    gamma=3.0)

# ── Step 4: Export ──

outdir = joinpath(@__DIR__, "..", "output", "example_cube")
mkpath(outdir)

write_growth_csv(joinpath(outdir, "vtree_segments.csv"), "VTree", trees["VTree"])
growth_viewer_html(joinpath(outdir, "index.html"), domain, trees, stats,
    Dict("VTree" => "#e63946"))

st = stats["VTree"]
println("\nDone! $(st.added) branches, $(st.terminals) terminals")
println("Viewer: $(joinpath(outdir, "index.html"))")
