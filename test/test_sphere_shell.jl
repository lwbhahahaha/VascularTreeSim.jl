@testset "Sphere Shell" begin
    Random.seed!(123)

    # ── Generate spherical shell domain points ──
    spacing = 0.15
    R_outer = 3.0
    R_inner = 2.0
    center = SVector(0.0, 0.0, 0.0)

    domain_pts_list = SVector{3,Float64}[]
    for x in -R_outer:spacing:R_outer, y in -R_outer:spacing:R_outer, z in -R_outer:spacing:R_outer
        p = SVector(x, y, z)
        r = norm(p - center)
        if r <= R_outer && r >= R_inner
            push!(domain_pts_list, p)
        end
    end

    @test length(domain_pts_list) > 10000

    domain_pts = Matrix{Float64}(undef, length(domain_pts_list), 3)
    for (i, p) in enumerate(domain_pts_list)
        domain_pts[i, 1] = p[1]; domain_pts[i, 2] = p[2]; domain_pts[i, 3] = p[3]
    end

    # ── Build synthetic VoxelShellDomain ──
    voxel_spacing = SVector(spacing, spacing, spacing)
    origin_cm = SVector(-R_outer - spacing, -R_outer - spacing, -R_outer - spacing)
    hi_cm = SVector(R_outer + spacing, R_outer + spacing, R_outer + spacing)
    dims = ntuple(d -> max(1, Int(ceil((hi_cm[d] - origin_cm[d]) / spacing))), 3)

    mask = falses(dims...)
    for p in domain_pts_list
        i = clamp(round(Int, (p[1] - origin_cm[1]) / spacing + 0.5), 1, dims[1])
        j = clamp(round(Int, (p[2] - origin_cm[2]) / spacing + 0.5), 1, dims[2])
        k = clamp(round(Int, (p[3] - origin_cm[3]) / spacing + 0.5), 1, dims[3])
        mask[i, j, k] = true
    end

    # Outer sphere surface
    outer_surface_pts = Float64[]
    outer_surface_nrm = Float64[]
    n_theta = 40; n_phi = 20
    for it in 0:n_theta-1, ip in 0:n_phi
        theta = 2pi * it / n_theta
        phi = pi * ip / n_phi
        x = R_outer * sin(phi) * cos(theta)
        y = R_outer * sin(phi) * sin(theta)
        z = R_outer * cos(phi)
        push!(outer_surface_pts, x, y, z)
        push!(outer_surface_nrm, x/R_outer, y/R_outer, z/R_outer)
    end
    n_outer = length(outer_surface_pts) ÷ 3
    outer_points = reshape(outer_surface_pts, 3, n_outer)' |> collect
    outer_normals = reshape(outer_surface_nrm, 3, n_outer)' |> collect

    # Inner sphere surface (cavity, inward normals)
    cavity_surface_pts = Float64[]
    cavity_surface_nrm = Float64[]
    for it in 0:n_theta-1, ip in 0:n_phi
        theta = 2pi * it / n_theta
        phi = pi * ip / n_phi
        x = R_inner * sin(phi) * cos(theta)
        y = R_inner * sin(phi) * sin(theta)
        z = R_inner * cos(phi)
        push!(cavity_surface_pts, x, y, z)
        push!(cavity_surface_nrm, -x/R_inner, -y/R_inner, -z/R_inner)
    end
    n_cavity = length(cavity_surface_pts) ÷ 3
    cavity_points = reshape(cavity_surface_pts, 3, n_cavity)' |> collect
    cavity_normals = reshape(cavity_surface_nrm, 3, n_cavity)' |> collect

    # Build spatial grids
    lo_t = (origin_cm[1], origin_cm[2], origin_cm[3])
    hi_t = (hi_cm[1], hi_cm[2], hi_cm[3])
    outer_grid = VascularTreeSim._build_point_grid(outer_points, lo_t, hi_t)
    cavity_grid = VascularTreeSim._build_point_grid(cavity_points, lo_t, hi_t)

    domain = VoxelShellDomain(
        mask, origin_cm, voxel_spacing, center,
        outer_points, outer_normals,
        [cavity_points], [cavity_normals],
        outer_grid, [cavity_grid]
    )

    # ── Two competing seed trees ──
    trees = Dict(
        "TopTree"    => growth_tree_from_seed("TopTree",    SVector(0.0, 0.0, 2.85)),
        "BottomTree" => growth_tree_from_seed("BottomTree", SVector(0.0, 0.0, -2.5)),
    )

    # ── Subsample ──
    coverage_pts = domain_pts[1:2:end, :]
    graph_pts = copy(coverage_pts)

    # ── Run competitive growth ──
    graph, territories, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_pts,
        graph_points_cm=graph_pts,
        effective_supply_radius_cm=0.01,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=300,
        graph_neighbors=12,
        min_frontier_separation_cm=0.2,
        max_path_nodes=20,
        frontier_batch=8,
        gamma=3.0,
        smooth_passes=10,
        spline_density=4,
        max_segment_length_cm=0.15)

    # ── Assertions ──
    for (name, st) in stats
        @test st.added > 100
        @test st.terminals > 50
        @test st.p95 < 1.5
    end
    @test length(trees) == 2

    # Both trees should claim territory (competitive)
    @test length(territories["TopTree"]) > 1000
    @test length(territories["BottomTree"]) > 1000

    # ── Export results ──
    outdir = joinpath(@__DIR__, "output", "sphere_shell")
    mkpath(outdir)

    for (name, tree) in trees
        csv_path = joinpath(outdir, "$(lowercase(name))_segments.csv")
        write_growth_csv(csv_path, name, tree)
        @test isfile(csv_path)
    end

    html_path = joinpath(outdir, "index.html")
    color_map = Dict("TopTree" => "#2563eb", "BottomTree" => "#dc2626")
    growth_viewer_html(html_path, domain, trees, stats, color_map)
    @test isfile(html_path)

    for (name, st) in stats
        println("  Sphere $(name): $(st.added) branches, $(st.terminals) terminals, p95=$(round(st.p95*10; digits=1))mm")
    end
end
