@testset "Cube Shell" begin
    Random.seed!(42)

    # ── Generate cube shell domain points ──
    spacing = 0.15  # cm
    outer_half = 3.0   # outer cube from 0..6
    inner_half = 2.0   # inner cube from 1..5
    center = SVector(3.0, 3.0, 3.0)

    domain_pts_list = SVector{3,Float64}[]
    for x in 0.0:spacing:6.0, y in 0.0:spacing:6.0, z in 0.0:spacing:6.0
        p = SVector(x, y, z)
        d = p - center
        in_outer = all(abs.(d) .<= outer_half)
        in_inner = all(abs.(d) .< inner_half)
        if in_outer && !in_inner
            push!(domain_pts_list, p)
        end
    end

    @test length(domain_pts_list) > 10000

    # Convert to Nx3 matrix
    domain_pts = Matrix{Float64}(undef, length(domain_pts_list), 3)
    for (i, p) in enumerate(domain_pts_list)
        domain_pts[i, 1] = p[1]; domain_pts[i, 2] = p[2]; domain_pts[i, 3] = p[3]
    end

    # ── Build synthetic VoxelShellDomain ──
    voxel_spacing = SVector(spacing, spacing, spacing)
    origin_cm = SVector(0.0 - spacing, 0.0 - spacing, 0.0 - spacing)
    dims = ntuple(d -> max(1, Int(ceil((6.0 + spacing - (0.0 - spacing)) / spacing))), 3)

    mask = falses(dims...)
    for p in domain_pts_list
        i = clamp(round(Int, (p[1] - origin_cm[1]) / spacing + 0.5), 1, dims[1])
        j = clamp(round(Int, (p[2] - origin_cm[2]) / spacing + 0.5), 1, dims[2])
        k = clamp(round(Int, (p[3] - origin_cm[3]) / spacing + 0.5), 1, dims[3])
        mask[i, j, k] = true
    end

    @test count(mask) > 5000

    # Outer surface points (6x6x6 cube faces)
    outer_surface_pts = Float64[]
    outer_surface_nrm = Float64[]
    surf_spacing = 0.3
    for x in 0.0:surf_spacing:6.0, y in 0.0:surf_spacing:6.0
        for (z_val, nz) in [(0.0, -1.0), (6.0, 1.0)]
            push!(outer_surface_pts, x, y, z_val)
            push!(outer_surface_nrm, 0.0, 0.0, nz)
        end
        for (z_val, ny) in [(0.0, -1.0), (6.0, 1.0)]
            push!(outer_surface_pts, x, z_val, y)
            push!(outer_surface_nrm, 0.0, ny, 0.0)
        end
        for (z_val, nx) in [(0.0, -1.0), (6.0, 1.0)]
            push!(outer_surface_pts, z_val, x, y)
            push!(outer_surface_nrm, nx, 0.0, 0.0)
        end
    end
    n_outer = length(outer_surface_pts) ÷ 3
    outer_points = reshape(outer_surface_pts, 3, n_outer)' |> collect
    outer_normals = reshape(outer_surface_nrm, 3, n_outer)' |> collect

    # Inner cavity surface points (4x4x4 cube faces)
    cavity_surface_pts = Float64[]
    cavity_surface_nrm = Float64[]
    for x in 1.0:surf_spacing:5.0, y in 1.0:surf_spacing:5.0
        for (z_val, nz) in [(1.0, -1.0), (5.0, 1.0)]
            push!(cavity_surface_pts, x, y, z_val)
            push!(cavity_surface_nrm, 0.0, 0.0, nz)
        end
        for (z_val, ny) in [(1.0, -1.0), (5.0, 1.0)]
            push!(cavity_surface_pts, x, z_val, y)
            push!(cavity_surface_nrm, 0.0, ny, 0.0)
        end
        for (z_val, nx) in [(1.0, -1.0), (5.0, 1.0)]
            push!(cavity_surface_pts, z_val, x, y)
            push!(cavity_surface_nrm, nx, 0.0, 0.0)
        end
    end
    n_cavity = length(cavity_surface_pts) ÷ 3
    cavity_points = reshape(cavity_surface_pts, 3, n_cavity)' |> collect
    cavity_normals = reshape(cavity_surface_nrm, 3, n_cavity)' |> collect

    # Build spatial grids
    lo_t = (minimum(outer_points[:, 1]) - 0.1, minimum(outer_points[:, 2]) - 0.1, minimum(outer_points[:, 3]) - 0.1)
    hi_t = (maximum(outer_points[:, 1]) + 0.1, maximum(outer_points[:, 2]) + 0.1, maximum(outer_points[:, 3]) + 0.1)
    outer_grid = VascularTreeSim._build_point_grid(outer_points, lo_t, hi_t)
    cavity_grid = VascularTreeSim._build_point_grid(cavity_points, lo_t, hi_t)

    domain = VoxelShellDomain(
        mask, origin_cm, voxel_spacing, center,
        outer_points, outer_normals,
        [cavity_points], [cavity_normals],
        outer_grid, [cavity_grid]
    )

    # ── Create seed tree ──
    seed_point = SVector(3.0, 0.5, 3.0)
    tree = growth_tree_from_seed("CubeTree", seed_point)
    trees = Dict("CubeTree" => tree)

    # ── Subsample points ──
    stride = 2
    coverage_pts = domain_pts[1:stride:end, :]
    graph_pts = copy(coverage_pts)

    # ── Run growth ──
    graph, territories, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_pts,
        graph_points_cm=graph_pts,
        effective_supply_radius_cm=0.01,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=200,
        graph_neighbors=12,
        min_frontier_separation_cm=0.25,
        max_path_nodes=20,
        frontier_batch=8,
        gamma=3.0,
        smooth_passes=10,
        spline_density=4,
        max_segment_length_cm=0.15)

    st = stats["CubeTree"]
    @test st.added == 200
    @test st.terminals > 50
    @test st.p95 < 2.0   # p95 coverage < 2 cm

    # ── Export results ──
    outdir = joinpath(@__DIR__, "output", "cube_shell")
    mkpath(outdir)

    csv_path = joinpath(outdir, "cubetree_segments.csv")
    write_growth_csv(csv_path, "CubeTree", trees["CubeTree"])
    @test isfile(csv_path)

    html_path = joinpath(outdir, "index.html")
    color_map = Dict("CubeTree" => "#e63946")
    growth_viewer_html(html_path, domain, trees, stats, color_map)
    @test isfile(html_path)
    @test filesize(html_path) > 1000

    println("  Cube Shell: $(st.added) branches, $(st.terminals) terminals, p95=$(round(st.p95*10; digits=1))mm")
end
