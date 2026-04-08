"""
    OrganConfig — organ-specific parameters loaded from TOML.

This is the key generalization layer. All organ-specific names, surfaces,
vessel definitions, and growth parameters are stored here and read from
a TOML configuration file.
"""

struct VesselTreeSpec
    name::String
    surface_names::Vector{String}
    color::String
    root_anchor_surface::String
end

struct OrganConfig
    # General
    organ_name::String
    nrb_path::String
    coordinate_scale::Float64

    # Surfaces
    outer_surface::String
    cavity_surfaces::Vector{String}

    # Vessel definitions
    vessel_trees::Vector{VesselTreeSpec}
    reference_surface::String   # e.g. aorta — used for root orientation

    # Domain parameters
    voxel_spacing_cm::Float64
    outer_samples::Tuple{Int,Int}
    cavity_samples::Tuple{Int,Int}
    dilation_radius::Int
    coarse_seed_cm::Union{Nothing, SVector{3,Float64}}

    # Growth parameters
    growth_mode::Symbol             # :continue_from_xcat or :seed_point
    effective_supply_radius_cm::Float64
    capillary_diameter_cm::Float64
    max_new_branches_per_tree::Int
    graph_neighbors::Int
    min_frontier_separation_cm::Float64
    max_path_nodes::Int
    frontier_batch::Int
    murray_gamma::Float64
    max_segment_length_cm::Float64
    smooth_passes::Int
    spline_density::Int
    coverage_stride::Int
    graph_stride::Int
    graph_jitter_cm::Float64
    target_p95_distance_cm::Float64
    target_max_distance_cm::Float64

    # Seed points (for :seed_point mode)
    seed_points::Dict{String, SVector{3,Float64}}
end

function _parse_svector3(arr)
    length(arr) == 3 || error("Expected 3-element array for SVector3, got $(length(arr))")
    return SVector{3, Float64}(Float64(arr[1]), Float64(arr[2]), Float64(arr[3]))
end

function load_organ_config(path::AbstractString)
    cfg = TOML.parsefile(path)

    organ_name = cfg["organ"]["name"]
    nrb_path = get(cfg["organ"], "nrb_path", "")
    coordinate_scale = get(cfg["organ"], "coordinate_scale", 0.1)

    outer_surface = cfg["surfaces"]["outer"]
    cavity_surfaces = Vector{String}(cfg["surfaces"]["cavities"])
    reference_surface = get(cfg["surfaces"], "reference", "")

    vessel_trees = VesselTreeSpec[]
    for vt in cfg["vessel_trees"]
        push!(vessel_trees, VesselTreeSpec(
            vt["name"],
            Vector{String}(vt["surface_names"]),
            get(vt, "color", "#888888"),
            get(vt, "root_anchor_surface", ""),
        ))
    end

    dom = get(cfg, "domain", Dict())
    voxel_spacing_cm = get(dom, "voxel_spacing_cm", 0.05)
    outer_samples = let os = get(dom, "outer_samples", [96, 72])
        (Int(os[1]), Int(os[2]))
    end
    cavity_samples = let cs = get(dom, "cavity_samples", [56, 40])
        (Int(cs[1]), Int(cs[2]))
    end
    dilation_radius = get(dom, "dilation_radius", 1)
    coarse_seed_cm = let cs = get(dom, "coarse_seed_cm", nothing)
        cs === nothing ? nothing : _parse_svector3(cs)
    end

    gr = get(cfg, "growth", Dict())
    growth_mode = Symbol(get(gr, "mode", "continue_from_xcat"))
    effective_supply_radius_cm = get(gr, "effective_supply_radius_cm", 1.25e-3)
    capillary_diameter_cm = get(gr, "capillary_diameter_cm", 8e-4)
    max_new_branches_per_tree = get(gr, "max_new_branches_per_tree", 220)
    graph_neighbors = get(gr, "graph_neighbors", 12)
    min_frontier_separation_cm = get(gr, "min_frontier_separation_cm", 0.18)
    max_path_nodes = get(gr, "max_path_nodes", 20)
    frontier_batch = get(gr, "frontier_batch", 8)
    murray_gamma = get(gr, "murray_gamma", 3.0)
    max_segment_length_cm = get(gr, "max_segment_length_cm", 0.1)
    smooth_passes = get(gr, "smooth_passes", 20)
    spline_density = get(gr, "spline_density", 5)
    coverage_stride = get(gr, "coverage_stride", 2)
    graph_stride = get(gr, "graph_stride", 0)
    graph_jitter_cm = get(gr, "graph_jitter_cm", 0.0)
    target_p95_distance_cm = get(gr, "target_p95_distance_cm", Inf)
    target_max_distance_cm = get(gr, "target_max_distance_cm", Inf)

    seed_points = Dict{String, SVector{3,Float64}}()
    if haskey(cfg, "seed_points")
        for (k, v) in cfg["seed_points"]
            seed_points[k] = _parse_svector3(v)
        end
    end

    return OrganConfig(
        organ_name, nrb_path, coordinate_scale,
        outer_surface, cavity_surfaces,
        vessel_trees, reference_surface,
        voxel_spacing_cm, outer_samples, cavity_samples, dilation_radius, coarse_seed_cm,
        growth_mode, effective_supply_radius_cm, capillary_diameter_cm,
        max_new_branches_per_tree, graph_neighbors, min_frontier_separation_cm,
        max_path_nodes, frontier_batch, murray_gamma, max_segment_length_cm,
        smooth_passes, spline_density, coverage_stride, graph_stride, graph_jitter_cm,
        target_p95_distance_cm, target_max_distance_cm,
        seed_points,
    )
end
