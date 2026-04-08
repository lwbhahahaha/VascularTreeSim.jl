@testset "Cylinder" begin
    Random.seed!(99)

    # ── Generate solid cylinder domain points ──
    spacing = 0.15
    R = 2.0
    H = 8.0
    center = SVector(0.0, 0.0, 0.0)

    domain_pts_list = SVector{3,Float64}[]
    for x in -R:spacing:R, y in -R:spacing:R, z in (-H/2):spacing:(H/2)
        if sqrt(x^2 + y^2) <= R
            push!(domain_pts_list, SVector(x, y, z))
        end
    end

    @test length(domain_pts_list) > 10000

    domain_pts = Matrix{Float64}(undef, length(domain_pts_list), 3)
    for (i, p) in enumerate(domain_pts_list)
        domain_pts[i, 1] = p[1]; domain_pts[i, 2] = p[2]; domain_pts[i, 3] = p[3]
    end

    # ── Build synthetic VoxelShellDomain ──
    voxel_spacing = SVector(spacing, spacing, spacing)
    origin_cm = SVector(-R - spacing, -R - spacing, -H/2 - spacing)
    hi_cm = SVector(R + spacing, R + spacing, H/2 + spacing)
    dims = ntuple(d -> max(1, Int(ceil((hi_cm[d] - origin_cm[d]) / spacing))), 3)

    mask = falses(dims...)
    for p in domain_pts_list
        i = clamp(round(Int, (p[1] - origin_cm[1]) / spacing + 0.5), 1, dims[1])
        j = clamp(round(Int, (p[2] - origin_cm[2]) / spacing + 0.5), 1, dims[2])
        k = clamp(round(Int, (p[3] - origin_cm[3]) / spacing + 0.5), 1, dims[3])
        mask[i, j, k] = true
    end

    # Cylinder barrel + end caps
    outer_surface_pts = Float64[]
    outer_surface_nrm = Float64[]
    n_theta_barrel = 48; z_step = 0.4
    for it in 0:n_theta_barrel-1
        theta = 2pi * it / n_theta_barrel
        nx = cos(theta); ny = sin(theta)
        for z in (-H/2):z_step:(H/2)
            push!(outer_surface_pts, R*nx, R*ny, z)
            push!(outer_surface_nrm, nx, ny, 0.0)
        end
    end
    cap_spacing = 0.4
    for x in -R:cap_spacing:R, y in -R:cap_spacing:R
        if x^2 + y^2 <= R^2
            push!(outer_surface_pts, x, y, -H/2)
            push!(outer_surface_nrm, 0.0, 0.0, -1.0)
            push!(outer_surface_pts, x, y, H/2)
            push!(outer_surface_nrm, 0.0, 0.0, 1.0)
        end
    end
    n_outer = length(outer_surface_pts) ÷ 3
    outer_points = reshape(outer_surface_pts, 3, n_outer)' |> collect
    outer_normals = reshape(outer_surface_nrm, 3, n_outer)' |> collect

    # Virtual cavity (tiny sphere at center) for midwall cost
    cavity_surface_pts = Float64[]
    cavity_surface_nrm = Float64[]
    r_virtual = 0.1
    for it in 0:11, ip in 0:6
        theta = 2pi * it / 12
        phi = pi * ip / 6
        x = r_virtual * sin(phi) * cos(theta)
        y = r_virtual * sin(phi) * sin(theta)
        z = r_virtual * cos(phi)
        nrm = r_virtual < 1e-8 ? (0.0, 0.0, -1.0) : (-x/r_virtual, -y/r_virtual, -z/r_virtual)
        push!(cavity_surface_pts, x, y, z)
        push!(cavity_surface_nrm, nrm[1], nrm[2], nrm[3])
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

    # ── Create seed tree ──
    tree = growth_tree_from_seed("CylinderTree", SVector(0.0, 0.0, 0.0))
    trees = Dict("CylinderTree" => tree)

    # ── Subsample ──
    coverage_pts = domain_pts[1:2:end, :]
    graph_pts = copy(coverage_pts)

    # ── Run growth ──
    graph, territories, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_pts,
        graph_points_cm=graph_pts,
        effective_supply_radius_cm=0.01,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=400,
        graph_neighbors=12,
        min_frontier_separation_cm=0.2,
        max_path_nodes=20,
        frontier_batch=10,
        gamma=3.0,
        smooth_passes=10,
        spline_density=4,
        max_segment_length_cm=0.15)

    st = stats["CylinderTree"]
    @test st.added == 400
    @test st.terminals > 100
    @test st.p95 < 1.5

    # ── Export results ──
    outdir = joinpath(@__DIR__, "output", "cylinder")
    mkpath(outdir)

    csv_path = joinpath(outdir, "cylindertree_segments.csv")
    write_growth_csv(csv_path, "CylinderTree", trees["CylinderTree"])
    @test isfile(csv_path)

    html_path = joinpath(outdir, "index.html")
    color_map = Dict("CylinderTree" => "#059669")
    growth_viewer_html(html_path, domain, trees, stats, color_map)
    @test isfile(html_path)

    println("  Cylinder: $(st.added) branches, $(st.terminals) terminals, p95=$(round(st.p95*10; digits=1))mm")
end
