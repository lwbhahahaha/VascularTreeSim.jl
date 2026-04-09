"""
    GPU vs CPU Benchmark for VascularTreeSim

Runs the cube shell test with both CPU and GPU backends,
comparing wall-clock time for the competitive growth engine.

Usage:
    julia --project=. examples/benchmark_gpu.jl
    julia --project=. --threads=auto examples/benchmark_gpu.jl
"""

using VascularTreeSim
using CUDA
using StaticArrays
using Statistics
using Random

println("=" ^ 60)
println("VascularTreeSim GPU Benchmark")
println("=" ^ 60)
println("GPU: $(CUDA.name(CUDA.device()))")
println("GPU available: $(gpu_available())")
println("CPU threads: $(Threads.nthreads())")
println()

# ── Build a large cube shell domain ──

function build_benchmark_domain(; spacing=0.1, outer_half=5.0, inner_frac=0.6)
    center = SVector(outer_half, outer_half, outer_half)
    side = 2 * outer_half
    inner_half = outer_half * inner_frac

    # Domain points
    domain_pts_list = SVector{3,Float64}[]
    for x in 0.0:spacing:side, y in 0.0:spacing:side, z in 0.0:spacing:side
        d = SVector(x, y, z) - center
        if all(abs.(d) .<= outer_half) && !all(abs.(d) .< inner_half)
            push!(domain_pts_list, SVector(x, y, z))
        end
    end

    # Build mask
    origin_cm = SVector(-spacing, -spacing, -spacing)
    dims = ntuple(d -> max(1, Int(ceil((side + spacing + spacing) / spacing))), 3)
    mask = falses(dims...)
    for p in domain_pts_list
        i = clamp(round(Int, (p[1] - origin_cm[1]) / spacing + 0.5), 1, dims[1])
        j = clamp(round(Int, (p[2] - origin_cm[2]) / spacing + 0.5), 1, dims[2])
        k = clamp(round(Int, (p[3] - origin_cm[3]) / spacing + 0.5), 1, dims[3])
        mask[i, j, k] = true
    end

    # Outer surface (cube faces)
    surf_spacing = 0.5
    outer_pts_flat = Float64[]
    outer_nrm_flat = Float64[]
    for x in 0.0:surf_spacing:side, y in 0.0:surf_spacing:side
        for (z_val, nz) in [(0.0, -1.0), (side, 1.0)]
            push!(outer_pts_flat, x, y, z_val); push!(outer_nrm_flat, 0.0, 0.0, nz)
        end
        for (z_val, ny) in [(0.0, -1.0), (side, 1.0)]
            push!(outer_pts_flat, x, z_val, y); push!(outer_nrm_flat, 0.0, ny, 0.0)
        end
        for (z_val, nx) in [(0.0, -1.0), (side, 1.0)]
            push!(outer_pts_flat, z_val, x, y); push!(outer_nrm_flat, nx, 0.0, 0.0)
        end
    end
    n_outer = length(outer_pts_flat) ÷ 3
    outer_points = reshape(outer_pts_flat, 3, n_outer)' |> collect
    outer_normals = reshape(outer_nrm_flat, 3, n_outer)' |> collect

    # Inner cavity surface
    inner_lo = outer_half - inner_half
    inner_hi = outer_half + inner_half
    cavity_pts_flat = Float64[]
    cavity_nrm_flat = Float64[]
    for x in inner_lo:surf_spacing:inner_hi, y in inner_lo:surf_spacing:inner_hi
        for (z_val, nz) in [(inner_lo, -1.0), (inner_hi, 1.0)]
            push!(cavity_pts_flat, x, y, z_val); push!(cavity_nrm_flat, 0.0, 0.0, nz)
        end
        for (z_val, ny) in [(inner_lo, -1.0), (inner_hi, 1.0)]
            push!(cavity_pts_flat, x, z_val, y); push!(cavity_nrm_flat, 0.0, ny, 0.0)
        end
        for (z_val, nx) in [(inner_lo, -1.0), (inner_hi, 1.0)]
            push!(cavity_pts_flat, z_val, x, y); push!(cavity_nrm_flat, nx, 0.0, 0.0)
        end
    end
    n_cavity = length(cavity_pts_flat) ÷ 3
    cavity_points = reshape(cavity_pts_flat, 3, n_cavity)' |> collect
    cavity_normals = reshape(cavity_nrm_flat, 3, n_cavity)' |> collect

    # Build spatial grids
    lo_t = (minimum(outer_points[:, 1]) - 0.1, minimum(outer_points[:, 2]) - 0.1, minimum(outer_points[:, 3]) - 0.1)
    hi_t = (maximum(outer_points[:, 1]) + 0.1, maximum(outer_points[:, 2]) + 0.1, maximum(outer_points[:, 3]) + 0.1)
    outer_grid = VascularTreeSim._build_point_grid(outer_points, lo_t, hi_t)
    cavity_grid = VascularTreeSim._build_point_grid(cavity_points, lo_t, hi_t)

    domain = VoxelShellDomain(
        mask, origin_cm, SVector(spacing, spacing, spacing), center,
        outer_points, outer_normals,
        [cavity_points], [cavity_normals],
        outer_grid, [cavity_grid]
    )

    # Coverage and graph points as Nx3 matrix
    domain_pts = Matrix{Float64}(undef, length(domain_pts_list), 3)
    for (i, p) in enumerate(domain_pts_list)
        domain_pts[i, 1] = p[1]; domain_pts[i, 2] = p[2]; domain_pts[i, 3] = p[3]
    end

    return domain, domain_pts
end

# ── Build domain ──

Random.seed!(42)
domain, all_pts = build_benchmark_domain(spacing=0.12)

# Use stride=2 for coverage, stride=3 for graph
coverage_pts = all_pts[1:2:end, :]
graph_pts = all_pts[1:3:end, :]

n_branches = 400
println("Domain: cube shell (spacing=0.12)")
println("Total domain points: $(size(all_pts, 1))")
println("Coverage points: $(size(coverage_pts, 1))")
println("Graph points: $(size(graph_pts, 1))")
println("Branches per tree: $(n_branches)")
println()

growth_kwargs = (
    coverage_points_cm=coverage_pts,
    graph_points_cm=graph_pts,
    effective_supply_radius_cm=0.01,
    capillary_diameter_cm=8e-4,
    max_new_branches_per_tree=n_branches,
    graph_neighbors=12,
    min_frontier_separation_cm=0.20,
    max_path_nodes=20,
    frontier_batch=10,
    gamma=3.0,
    smooth_passes=10,
    spline_density=4,
    max_segment_length_cm=0.1,
)

# ── CPU Benchmark ──

println("─" ^ 40)
println("Running CPU benchmark...")
Random.seed!(42)
tree_cpu = Dict("CubeTree" => growth_tree_from_seed("CubeTree", SVector(5.0, 0.5, 5.0)))
t_cpu = @elapsed begin
    _, _, stats_cpu = grow_trees_mcp!(tree_cpu, domain; growth_kwargs..., use_gpu=false)
end
st = stats_cpu["CubeTree"]
println("\nCPU time: $(round(t_cpu; digits=2))s")
println("CPU stats: added=$(st.added) terminals=$(st.terminals) p95=$(round(st.p95; digits=4))")
println()

# ── GPU Benchmark ──

println("─" ^ 40)
println("Running GPU warm-up (kernel compilation)...")
Random.seed!(99)
warmup_tree = Dict("W" => growth_tree_from_seed("W", SVector(5.0, 0.5, 5.0)))
warmup_pts = coverage_pts[1:100:end, :]
warmup_gpts = graph_pts[1:100:end, :]
try
    grow_trees_mcp!(warmup_tree, domain;
        coverage_points_cm=warmup_pts, graph_points_cm=warmup_gpts,
        max_new_branches_per_tree=5, frontier_batch=5,
        effective_supply_radius_cm=0.01, use_gpu=true)
catch e
    println("  warm-up note: $(e)")
end
CUDA.synchronize()

println("\nRunning GPU benchmark...")
Random.seed!(42)
tree_gpu = Dict("CubeTree" => growth_tree_from_seed("CubeTree", SVector(5.0, 0.5, 5.0)))
t_gpu = @elapsed begin
    _, _, stats_gpu = grow_trees_mcp!(tree_gpu, domain; growth_kwargs..., use_gpu=true)
end
st = stats_gpu["CubeTree"]
println("\nGPU time: $(round(t_gpu; digits=2))s")
println("GPU stats: added=$(st.added) terminals=$(st.terminals) p95=$(round(st.p95; digits=4))")
println()

# ── Summary ──

println("=" ^ 60)
speedup = t_cpu / t_gpu
println("  CPU ($(Threads.nthreads()) threads): $(round(t_cpu; digits=2))s")
println("  GPU ($(CUDA.name(CUDA.device()))): $(round(t_gpu; digits=2))s")
println("  Speedup: $(round(speedup; digits=2))x")
println("=" ^ 60)
