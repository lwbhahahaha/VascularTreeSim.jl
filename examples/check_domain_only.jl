"""
    check_domain_only.jl — Build domain and render viewer, skip growth.

Use this to sanity-check domain changes (e.g. adding/removing cavity surfaces
in configs/coronary.toml) without paying the full pipeline cost.

Usage:
    julia --project=. --threads=auto examples/check_domain_only.jl
"""

using VascularTreeSim
using Dates

const CONFIG_PATH = joinpath(dirname(@__DIR__), "configs", "coronary.toml")
const OUTPUT_DIR  = joinpath(dirname(@__DIR__), "output")
const OUT_HTML    = joinpath(OUTPUT_DIR, "domain_check.html")

println("═" ^ 60)
println("  Domain-only check")
println("  Config: $(CONFIG_PATH)")
println("  Output: $(OUT_HTML)")
println("  Started: $(Dates.now())")
println("═" ^ 60)

config = load_organ_config(CONFIG_PATH)
println("  Organ: $(config.organ_name)")
println("  Outer: $(config.outer_surface)")
println("  Reference: $(config.reference_surface)")
println("  Cavities ($(length(config.cavity_surfaces))):")
for name in config.cavity_surfaces
    println("    - $(name)")
end
flush(stdout)

surfaces = parse_xcat_nrb(config.nrb_path)
obj = xcat_object_dict(surfaces)
println("  Parsed $(length(surfaces)) NRB surfaces")

outer_surface = obj[config.outer_surface]
cavity_surface_list = [obj[name] for name in config.cavity_surfaces]

t0 = time()
domain = build_voxel_shell_domain_floodfill(outer_surface, cavity_surface_list;
    coordinate_scale=config.coordinate_scale,
    voxel_spacing_cm=config.voxel_spacing_cm,
    outer_samples=config.outer_samples,
    cavity_samples=config.cavity_samples,
    dilation_radius=config.dilation_radius,
    coarse_seed_cm=config.coarse_seed_cm)
dt = time() - t0

n_domain = count(domain.mask)
vox_cm3 = prod(domain.spacing_cm)
vol_cm3 = n_domain * vox_cm3
println("  Domain built in $(round(dt, digits=1))s")
println("  Voxels: $(n_domain)")
println("  Volume: $(round(vol_cm3, digits=1)) cm³")
flush(stdout)

domain_check_html(OUT_HTML, domain; max_display=500_000)
println("  Wrote viewer: $(OUT_HTML)")
println("  Done: $(Dates.now())")
