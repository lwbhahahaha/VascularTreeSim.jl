"""
    VascularTreeSim

A Julia package for vascular tree generation using competitive constrained
constructive optimization (CCO) within anatomical domains.

# Quick Start
```julia
using VascularTreeSim

# From TOML config (XCAT phantom workflow)
config = load_organ_config("configs/coronary.toml")
result = run_growth(config; output_dir="output")

# From seed point (synthetic geometry)
tree = growth_tree_from_seed("MyTree", SVector(0.0, 0.0, 0.0))
```

See also: [`run_growth`](@ref), [`grow_trees_mcp!`](@ref), [`growth_tree_from_seed`](@ref)
"""
module VascularTreeSim

using Random
using DelimitedFiles
using LinearAlgebra
using StaticArrays
using Statistics
using TOML

include("organ_config.jl")
include("nrb_parser.jl")
include("centerline.jl")
include("spatial_grid.jl")
include("domain_builder.jl")
include("growth_tree.jl")
include("segment_index.jl")
include("gpu_interface.jl")
include("graph_routing.jl")
include("growth_engine.jl")
include("csv_io.jl")
include("viewer.jl")

# ── Types ──
export OrganConfig,
       VesselTreeSpec,
       XCATNurbsSurface,
       XCATCenterline,
       XCATTreeConnection,
       XCATCenterlineTree,
       PointCloudGrid,
       VoxelShellDomain,
       GrowthTree,
       DomainGraph,
       GraphSpatialGrid

# ── Config ──
export load_organ_config

# ── XCAT parsing ──
export parse_xcat_nrb,
       xcat_object_dict,
       xcat_sample_surface,
       xcat_centerline_from_surface,
       build_vessel_trees

# ── Domain construction ──
export build_voxel_shell_domain_floodfill,
       voxel_mask_points,
       coverage_target_points,
       coverage_target_points_blockwise,
       shell_distance_components,
       shell_midwall_cost

# ── Tree construction ──
export growth_tree_from_xcat,
       growth_tree_from_seed

# ── Acceleration ──
export SegmentSpatialIndex,
       build_segment_index,
       gpu_available

# ── Growth engine ──
export build_domain_graph,
       grow_trees_mcp!

# ── I/O & Visualization ──
export write_growth_csv,
       growth_viewer_html

# ── Top-level pipeline ──
export run_growth

end # module VascularTreeSim
