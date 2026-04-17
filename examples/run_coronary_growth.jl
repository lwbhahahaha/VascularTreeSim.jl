"""
    run_coronary_growth.jl — Grow coronary trees from XCAT using the v5 proven pipeline.

This is the correct pipeline:
  Step 1: Parse NRB → build domain → show to user for confirmation (no holes!)
  Step 2: Extract existing LAD/LCX/RCA from XCAT NRB surfaces
  Step 3: Continue growing from existing vessels (competitive round-robin)
  Step 4: Export CSVs + HTML viewer

All parameters come from configs/coronary.toml and match v5's proven fine run.

Usage:
    julia --project=. --threads=auto examples/run_coronary_growth.jl
"""

using CUDA              # GPU acceleration (triggers VascularTreeSimCUDAExt)
using VascularTreeSim
using Dates
using Printf
using Random

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
const CONFIG_PATH = joinpath(dirname(@__DIR__), "configs", "coronary.toml")
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
const RUN_TIMESTAMP = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

println("═" ^ 60)
println("  Coronary Tree Growth (v5 pipeline)")
println("  Config: $(CONFIG_PATH)")
println("  Output: $(OUTPUT_DIR)")
println("  Timestamp: $(RUN_TIMESTAMP)")
println("═" ^ 60)

mkpath(OUTPUT_DIR)

# ══════════════════════════════════════════════════════════════
# Step 1: Load config + parse NRB + build domain
# ══════════════════════════════════════════════════════════════
println("\n[Step 1/4] Loading config and building domain...")
flush(stdout)

config = load_organ_config(CONFIG_PATH)
println("  Organ: $(config.organ_name)")
println("  Growth mode: $(config.growth_mode)")
println("  NRB: $(config.nrb_path)")
println("  Voxel spacing: $(config.voxel_spacing_cm) cm")
println("  Max branches/tree: $(config.max_new_branches_per_tree)")
flush(stdout)

# Parse NRB surfaces
surfaces = parse_xcat_nrb(config.nrb_path)
obj = xcat_object_dict(surfaces)
println("  Parsed $(length(surfaces)) NRB surfaces")

# Build domain from NRB epicardial/endocardial surfaces (flood-fill, v5 approach)
outer_surface = obj[config.outer_surface]
cavity_surface_list = [obj[name] for name in config.cavity_surfaces]
domain = build_voxel_shell_domain_floodfill(outer_surface, cavity_surface_list;
    coordinate_scale=config.coordinate_scale,
    voxel_spacing_cm=config.voxel_spacing_cm,
    outer_samples=config.outer_samples,
    cavity_samples=config.cavity_samples,
    dilation_radius=config.dilation_radius,
    coarse_seed_cm=config.coarse_seed_cm)

n_domain = count(domain.mask)
println("  Domain built: $(n_domain) voxels")
println("  Origin: $(domain.origin_cm)")
println("  Spacing: $(domain.spacing_cm)")
flush(stdout)

# ══════════════════════════════════════════════════════════════
# Step 1b: Show domain to user for confirmation
# ══════════════════════════════════════════════════════════════
println("\n[Step 1b] Generating domain confirmation viewer...")
flush(stdout)

# Generate domain-only HTML for user confirmation
domain_html = joinpath(OUTPUT_DIR, "domain_check.html")
domain_check_html(domain_html, domain; max_display=500_000)
println("  Domain viewer: $(domain_html)")
flush(stdout)

# Interactive mode: wait for user confirmation; non-interactive: auto-continue
if isinteractive() || isa(stdin, Base.TTY)
    println("\n" * "═" ^ 60)
    println("  DOMAIN CONFIRMATION")
    println("  → $(domain_html)")
    println("  Press ENTER to continue with growth, or Ctrl+C to abort.")
    println("═" ^ 60)
    flush(stdout)
    http_proc = run(`python3 -m http.server 8001 --directory $OUTPUT_DIR`, wait=false)
    readline()
    kill(http_proc)
else
    println("  [non-interactive] Auto-continuing past domain check")
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════
# Step 1c: Export domain + anatomy CSVs (NRB coordinate system)
# ══════════════════════════════════════════════════════════════
println("\n[Step 1c] Exporting domain & anatomy CSVs (NRB coordinates)...")
flush(stdout)

# Domain (myocardium) points — subsample for viewer (600K of ~40M)
domain_pts = voxel_mask_points(domain)
n_domain_pts = size(domain_pts, 1)
max_domain_csv = 800_000
if n_domain_pts > max_domain_csv
    rng = Random.MersenneTwister(42)
    indices = sort(Random.randperm(rng, n_domain_pts)[1:max_domain_csv])
else
    indices = 1:n_domain_pts
end
open(joinpath(OUTPUT_DIR, "domain_points.csv"), "w") do io
    println(io, "x_cm,y_cm,z_cm")
    for i in indices
        @printf(io, "%.3f,%.3f,%.3f\n", domain_pts[i,1], domain_pts[i,2], domain_pts[i,3])
    end
end
println("  domain_points.csv: $(length(indices)) / $(n_domain_pts) points")

# Cavity surfaces → chambers_points.csv
let n_chambers = Ref(0)
    open(joinpath(OUTPUT_DIR, "chambers_points.csv"), "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for cavity_pts in domain.cavity_surface_points
            for i in 1:size(cavity_pts, 1)
                @printf(io, "%.3f,%.3f,%.3f\n", cavity_pts[i,1], cavity_pts[i,2], cavity_pts[i,3])
                n_chambers[] += 1
            end
        end
    end
    println("  chambers_points.csv: $(n_chambers[]) points")
end

# Outer surface (pericardium) → pericardium_points.csv
epi = domain.outer_surface_points
open(joinpath(OUTPUT_DIR, "pericardium_points.csv"), "w") do io
    println(io, "x_cm,y_cm,z_cm")
    for i in 1:size(epi, 1)
        @printf(io, "%.3f,%.3f,%.3f\n", epi[i,1], epi[i,2], epi[i,3])
    end
end
println("  pericardium_points.csv: $(size(epi, 1)) points")

# Reference surface (aorta / great vessels) → great_vessels_points.csv
if !isempty(config.reference_surface) && haskey(obj, config.reference_surface)
    ref_surf = obj[config.reference_surface]
    ref_pts, _, _, _ = xcat_sample_surface(ref_surf; n_u=96, n_v=72, orient_outward=true)
    n_gv = length(ref_pts)
    open(joinpath(OUTPUT_DIR, "great_vessels_points.csv"), "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for j in axes(ref_pts, 1), i in axes(ref_pts, 2)
            p = ref_pts[j, i] .* config.coordinate_scale
            @printf(io, "%.3f,%.3f,%.3f\n", p[1], p[2], p[3])
        end
    end
    println("  great_vessels_points.csv: $(n_gv) points")
end

# XCAT coronary centerlines (from NRB surfaces, same coordinate system as trees)
let n_cor = Ref(0)
    open(joinpath(OUTPUT_DIR, "coronary_arteries_points.csv"), "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for spec in config.vessel_trees
            for sname in spec.surface_names
                haskey(obj, sname) || continue
                surf = obj[sname]
                cline = xcat_centerline_from_surface(surf)
                for pt in cline.centers
                    p = pt .* config.coordinate_scale
                    @printf(io, "%.3f,%.3f,%.3f\n", p[1], p[2], p[3])
                    n_cor[] += 1
                end
            end
        end
    end
    println("  coronary_arteries_points.csv: $(n_cor[]) points (NRB)")
end
flush(stdout)

# ══════════════════════════════════════════════════════════════
# Step 2: Extract existing XCAT vessels
# ══════════════════════════════════════════════════════════════
println("\n[Step 2/4] Extracting XCAT vessel centerlines...")
flush(stdout)

growth_trees = Dict{String, GrowthTree}()

if config.growth_mode == :continue_from_xcat
    all_vessel_surface_names = String[]
    !isempty(config.reference_surface) && push!(all_vessel_surface_names, config.reference_surface)
    for spec in config.vessel_trees
        append!(all_vessel_surface_names, spec.surface_names)
    end
    unique!(all_vessel_surface_names)
    vessel_surfaces = [obj[name] for name in all_vessel_surface_names if haskey(obj, name)]
    centerlines = [xcat_centerline_from_surface(s) for s in vessel_surfaces]
    xcat_trees = build_vessel_trees(centerlines, config)
    for spec in config.vessel_trees
        haskey(xcat_trees, spec.name) || continue
        growth_trees[spec.name] = growth_tree_from_xcat(spec.name, xcat_trees[spec.name];
            terminal_diameter_cm=config.terminal_diameter_cm)
        n_seg = length(growth_trees[spec.name].segment_start)
        n_vert = length(growth_trees[spec.name].vertices)
        println("  $(spec.name): $(n_seg) XCAT segments, $(n_vert) vertices")
    end
elseif config.growth_mode == :seed_point
    for spec in config.vessel_trees
        haskey(config.seed_points, spec.name) || continue
        growth_trees[spec.name] = growth_tree_from_seed(spec.name, config.seed_points[spec.name];
            terminal_diameter_cm=config.terminal_diameter_cm)
        println("  $(spec.name): seeded at $(config.seed_points[spec.name])")
    end
else
    error("Unknown growth mode: $(config.growth_mode)")
end
isempty(growth_trees) && error("No vessel trees initialized! Check config.")
flush(stdout)

# ══════════════════════════════════════════════════════════════
# Step 3: Competitive round-robin growth
# ══════════════════════════════════════════════════════════════
println("\n[Step 3/5] Growing trees (competitive round-robin)...")
flush(stdout)

coverage_block = max(config.coverage_stride, 1)
route_block = config.graph_stride <= 0 ? coverage_block : config.graph_stride
coverage_points = coverage_target_points_blockwise(domain; block_size=coverage_block)
graph_points = coverage_target_points_blockwise(domain; block_size=route_block)
graph_points = VascularTreeSim._jitter_points_in_domain(graph_points, domain; max_jitter_cm=config.graph_jitter_cm)
println("  Coverage points: $(size(coverage_points, 1))")
println("  Graph points: $(size(graph_points, 1))")
flush(stdout)

graph, territories, stats = grow_trees_mcp!(growth_trees, domain;
    coverage_points_cm=coverage_points, graph_points_cm=graph_points,
    effective_supply_radius_cm=config.effective_supply_radius_cm,
    capillary_diameter_cm=config.capillary_diameter_cm,
    max_new_branches_per_tree=config.max_new_branches_per_tree,
    graph_neighbors=config.graph_neighbors,
    min_frontier_separation_cm=config.min_frontier_separation_cm,
    max_path_nodes=config.max_path_nodes,
    frontier_batch=config.frontier_batch,
    gamma=config.murray_gamma,
    smooth_passes=config.smooth_passes,
    spline_density=config.spline_density,
    max_segment_length_cm=config.max_segment_length_cm,
    target_p95_distance_cm=config.target_p95_distance_cm,
    target_max_distance_cm=config.target_max_distance_cm,
    turn_penalty=config.turn_penalty,
    graph_jitter_cm=0.0)  # jitter already applied above

for (name, st) in stats
    println("  $(name): $(st.added) grown + XCAT segments, $(st.terminals) terminals, p95=$(round(st.p95*10; digits=2))mm")
end
flush(stdout)

# ── Post-growth subdivision: bifurcate terminals to finer diameter ──
if config.subdivision_terminal_diameter_cm > 0.0 && config.subdivision_terminal_diameter_cm < config.terminal_diameter_cm
    println("\n[Step 3b] Subdividing terminals: $(round(config.terminal_diameter_cm*1e4; digits=0)) μm → $(round(config.subdivision_terminal_diameter_cm*1e4; digits=0)) μm ...")
    flush(stdout)
    for (name, tree) in growth_trees
        subdivide_terminals!(tree;
            target_diameter_cm=config.subdivision_terminal_diameter_cm,
            gamma=config.murray_gamma,
            domain=domain)  # clip sub-branches that leave myocardium
    end
    # Print diameter range after subdivision
    for (name, tree) in growth_trees
        dmin = minimum(tree.segment_diameter_cm) * 1e4
        dmax = maximum(tree.segment_diameter_cm) * 1e4
        nt = length(VascularTreeSim._branch_terminals(tree))
        println("  $(name): $(nt) terminals, diam $(round(dmin; digits=1))-$(round(dmax; digits=1)) μm")
    end
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════
# Step 4: Export CSVs + viewer
# ══════════════════════════════════════════════════════════════
println("\n[Step 4/5] Exporting results...")
flush(stdout)

# CSVs — latest + timestamped
for (name, tree) in growth_trees
    lname = lowercase(name)
    # Latest
    csv_path = joinpath(OUTPUT_DIR, "$(lname)_segments.csv")
    write_growth_csv(csv_path, name, tree)
    println("  $(csv_path)")
    # Timestamped
    ts_csv = joinpath(OUTPUT_DIR, "$(lname)_segments_$(RUN_TIMESTAMP).csv")
    write_growth_csv(ts_csv, name, tree)
    println("  $(ts_csv)")
end

# HTML viewer
color_map = Dict(spec.name => spec.color for spec in config.vessel_trees)
html_path = joinpath(OUTPUT_DIR, "xcat_coronary_viewer.html")
growth_viewer_html(html_path, domain, growth_trees, stats, color_map;
    domain_stride=1, surface_stride=4)
println("  $(html_path)")

# Also run build_viewer.py if available (for the proven v5-style viewer)
viewer_script = joinpath(OUTPUT_DIR, "build_viewer.py")
if isfile(viewer_script)
    try
        run(`python3 $viewer_script`)
        println("  Python viewer rebuilt")
    catch e
        @warn "build_viewer.py failed" exception=e
    end
end

# Run summary
summary_path = joinpath(OUTPUT_DIR, "run_summary_$(RUN_TIMESTAMP).txt")
open(summary_path, "w") do io
    println(io, "Coronary Tree Growth — $(RUN_TIMESTAMP)")
    println(io, "=" ^ 60)
    println(io, "Config: $(CONFIG_PATH)")
    println(io, "Growth mode: $(config.growth_mode)")
    println(io, "Domain voxels: $(n_domain)")
    println(io, "Voxel spacing: $(config.voxel_spacing_cm) cm")
    println(io, "Max branches/tree: $(config.max_new_branches_per_tree)")
    println(io, "")
    for (name, st) in stats
        println(io, "$(name): added=$(st.added) terminals=$(st.terminals) p95=$(round(st.p95*10; digits=2))mm")
    end
end
println("  $(summary_path)")

# ══════════════════════════════════════════════════════════════
# Step 5: Embed trees into vmale50 phantom → raw file
# ══════════════════════════════════════════════════════════════
println("\n[Step 5/5] Embedding trees into XCAT phantom...")
flush(stdout)

# Read phantom config from TOML (not in OrganConfig struct)
using TOML
toml_cfg = TOML.parsefile(CONFIG_PATH)
phantom_path = get(get(toml_cfg, "organ", Dict()), "phantom_path", "")
phantom_dims_arr = get(get(toml_cfg, "organ", Dict()), "phantom_dims", [1600, 1400, 500])
ph_nx, ph_ny, ph_nz = Int(phantom_dims_arr[1]), Int(phantom_dims_arr[2]), Int(phantom_dims_arr[3])

if !isempty(phantom_path) && isfile(phantom_path)
    println("  Loading phantom: $(phantom_path)")
    println("  Dims: $(ph_nx)×$(ph_ny)×$(ph_nz) = $(round(ph_nx*ph_ny*ph_nz/1024^2; digits=0)) MB")
    flush(stdout)

    phantom = Array{UInt8}(undef, ph_nx, ph_ny, ph_nz)
    read!(phantom_path, phantom)

    # Embed: XCAT coronaries→125, grown→255
    embed_stats = embed_trees_in_phantom!(phantom, growth_trees;
        voxel_cm=config.voxel_spacing_cm,
        min_render_diameter_cm=config.voxel_spacing_cm)

    # Write full phantom
    raw_path = joinpath(OUTPUT_DIR, "vmale50_with_grown_coronaries.raw")
    info_path = joinpath(OUTPUT_DIR, "phantom_info.txt")
    write_phantom_raw(raw_path, phantom;
        info_path=info_path,
        trees=growth_trees,
        embed_stats=embed_stats,
        growth_stats=stats)

    # Timestamped copy
    ts_raw = joinpath(OUTPUT_DIR, "vmale50_with_grown_coronaries_$(RUN_TIMESTAMP).raw")
    cp(raw_path, ts_raw)
    println("  $(ts_raw)")
else
    @warn "Phantom path not configured or file not found: $(phantom_path)"
end

# Serve (only in interactive mode)
println("\n" * "═" ^ 60)
println("  Growth complete!")
println("  Viewer HTML: $(html_path)")
if @isdefined(raw_path)
    println("  Raw phantom: $(raw_path)")
end
println("═" ^ 60)
flush(stdout)

if isinteractive() || isa(stdin, Base.TTY)
    println("  → http://localhost:8001/xcat_coronary_viewer.html")
    run(`python3 -m http.server 8001 --directory $OUTPUT_DIR`, wait=false)
    println("Press Ctrl+C to stop.")
    while true; sleep(60); end
end
