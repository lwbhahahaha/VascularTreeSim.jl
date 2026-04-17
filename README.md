# VascularTreeSim.jl

A Julia package for vascular tree generation using competitive constrained
constructive optimization (CCO) within anatomical domains.

Given an XCAT NURBS phantom file (`.nrb`) or a synthetic geometry, this package
builds a 3-D voxel domain of the organ wall, initializes vessel trees from either
XCAT centerlines or user-supplied seed points, and grows new branches via
competitive round-robin minimum-cost-path (MCP) routing until the tissue is
adequately perfused. Outputs include per-tree CSV segment files and an interactive
3-D HTML viewer.

## Features

- **TOML-driven configuration** -- all organ names, surfaces, vessel definitions, and
  growth parameters live in a single `.toml` file. No organ-specific logic is
  hardcoded.
- **Two growth modes**: `continue_from_xcat` (extend existing XCAT centerlines) and
  `seed_point` (grow from scratch at user-specified coordinates).
- **Competitive round-robin territory** -- multiple trees (e.g. LAD, LCX, RCA) grow
  simultaneously, each claiming the tissue closest to its existing branches.
- **Strict Murray-derived diameters** -- every segment diameter (XCAT and grown) is
  recomputed as `d = d_term × N^(1/γ)` from its subtree terminal count, so Murray's
  law is satisfied globally, not only at the branch being added. The XCAT ostium
  diameter acts only as a capacity ceiling via a per-tree terminal budget.
- **Hybrid growth + subdivision pipeline** -- grow first to a realistic 200 μm
  terminal, then recursively bifurcate each tip to 8 μm capillaries. This produces
  physiological root diameters (mm-scale) and true capillary counts (tens of
  millions) in a single pipeline.
- **Domain-clipped subdivision with rotation retry** -- bifurcation children are
  tested against the voxel mask; up to 8 random plane rotations are tried per
  split so the tree stays inside the anatomical shell without losing whole
  subtrees to a single unlucky random direction.
- **Chamber vs. tubular cavity auto-detection** -- tubular surfaces whose centroid
  falls outside the organ wall (e.g. aorta, SVC, IVC in a cardiac phantom) are
  automatically detected and given a bounded SDF subtraction + fallback flood-fill
  seed, preventing over-subtraction far from the actual surface samples.
- **Catmull-Rom spline smoothing** with domain-constrained Laplacian passes produces
  anatomically plausible branch geometry.
- **Heap-based Dijkstra routing** on a k-nearest-neighbor voxel graph ensures paths
  stay within the myocardial shell.
- **Coverage-driven + stall-aware stopping** -- growth halts when P95/max distance
  targets are met, when the branch budget is exhausted, or when P95 stops
  improving for 20 consecutive rounds.
- **Interactive HTML viewer** (Plotly-based) with diameter-binned line widths and
  per-tree color coding.

## Installation

The package uses only Julia standard libraries plus `StaticArrays`. Requires Julia >= 1.9.

```julia
using Pkg

# Install from a local clone:
Pkg.develop(path="/path/to/VascularTreeSim.jl")

# Or from a Git URL:
Pkg.add(url="https://github.com/your-org/VascularTreeSim.jl.git")
```

## Quick Start

### From TOML config (XCAT phantom workflow)

```julia
using VascularTreeSim

config = load_organ_config("configs/coronary.toml")
result = run_growth(config; output_dir="output")
# result.html_path → interactive 3-D viewer
```

### From seed point (synthetic geometry)

```julia
using VascularTreeSim
using StaticArrays

tree = growth_tree_from_seed("MyTree", SVector(3.0, 0.5, 3.0))
trees = Dict("MyTree" => tree)

# Build domain, graph, then grow (see examples/synthetic_cube.jl)
graph, territories, stats = grow_trees_mcp!(trees, domain; max_new_branches_per_tree=300)

# Export
write_growth_csv("segments.csv", "MyTree", trees["MyTree"])
```

## Testing

```julia
using Pkg
Pkg.test("VascularTreeSim")
```

Runs 3 synthetic geometry tests (cube shell, sphere shell, cylinder) with
`@testset` assertions on branch count, terminal count, and coverage metrics.

The `output/` directory will contain:

- `lad_grown_segments.csv`, `lcx_grown_segments.csv`, `rca_grown_segments.csv`
- `index.html` -- interactive 3-D viewer

## Architecture Overview

The pipeline executed by `run_growth` proceeds through these stages:

```
NRB file
  |  parse_xcat_nrb()
  v
NURBS surfaces (Dict{String, XCATNurbsSurface})
  |  xcat_centerline_from_surface()  +  build_vessel_trees()
  v
Centerline trees (Dict{String, XCATCenterlineTree})
  |  growth_tree_from_xcat() / growth_tree_from_seed()
  v
GrowthTree structs (mutable, per-vessel)
  |
  |  build_voxel_shell_domain_floodfill()  -- chamber/tube auto-detection
  v
VoxelShellDomain (3-D BitArray mask of organ wall)
  |  build_domain_graph()  -- k-NN graph over voxel points
  v
DomainGraph (nodes + weighted edges)
  |  grow_trees_mcp!()  -- competitive round-robin MCP growth
  v                         (strict Murray: d = d_term × N^(1/γ) everywhere)
Grown GrowthTree structs
  |
  |  subdivide_terminals!()  -- optional: recursive bifurcation to finer
  v                             terminal diameter, domain-clipped with
  v                             rotation retry
Subdivided GrowthTree structs
  |  write_growth_csv()  +  growth_viewer_html()
  v
CSV files  +  HTML viewer
```

### Key data structures

| Struct | Purpose |
|---|---|
| `OrganConfig` | All parameters from the TOML file |
| `VesselTreeSpec` | Per-tree name, surface names, color, anchor |
| `XCATNurbsSurface` | Parsed NURBS control points and knot vectors |
| `XCATCenterline` | Centerline points + radii extracted from a NURBS surface |
| `XCATCenterlineTree` | Connected tree of `XCATCenterline` segments |
| `VoxelShellDomain` | 3-D voxel mask + surface point clouds for distance queries |
| `DomainGraph` | k-NN graph with midwall-cost-weighted edges |
| `GrowthTree` | Mutable vertex/segment tree with diameters and labels |

## OrganConfig TOML Schema

### `[organ]`

| Field | Type | Description |
|---|---|---|
| `name` | `String` | Organ identifier (e.g. `"coronary"`) |
| `nrb_path` | `String` | Absolute path to the XCAT `.nrb` file |
| `coordinate_scale` | `Float64` | Multiplier converting NRB coordinates to cm (default `0.1` for mm-to-cm) |

### `[surfaces]`

| Field | Type | Description |
|---|---|---|
| `outer` | `String` | Name of the outer bounding surface (e.g. `"dias_pericardium"`) |
| `cavities` | `Array{String}` | Names of cavity surfaces to exclude (LV, RV, LA, RA, etc.) |
| `reference` | `String` | Reference surface for root orientation (e.g. `"dias_aorta"`) |

### `[[vessel_trees]]` (array of tables)

| Field | Type | Description |
|---|---|---|
| `name` | `String` | Tree identifier (e.g. `"LAD"`, `"LCX"`, `"RCA"`) |
| `surface_names` | `Array{String}` | XCAT surface names belonging to this tree |
| `color` | `String` | Hex color for the viewer (default `"#888888"`) |
| `root_anchor_surface` | `String` | Surface used to anchor the tree root (e.g. `"dias_aorta"`) |

### `[domain]`

| Field | Type | Default | Description |
|---|---|---|---|
| `voxel_spacing_cm` | `Float64` | `0.05` | Isotropic voxel edge length in cm |
| `outer_samples` | `[Int, Int]` | `[96, 72]` | `(n_u, n_v)` samples on the outer surface |
| `cavity_samples` | `[Int, Int]` | `[56, 40]` | `(n_u, n_v)` samples per cavity surface |
| `dilation_radius` | `Int` | `1` | Voxel dilation passes for the shell mask |
| `coarse_seed_cm` | `[Float64, Float64, Float64]` | `nothing` | Optional 3-D seed for flood fill; omit to auto-detect |

### `[growth]`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `String` | `"continue_from_xcat"` | Growth mode: `"continue_from_xcat"` or `"seed_point"` |
| `effective_supply_radius_cm` | `Float64` | `0.00125` | Radius within which a point is considered perfused |
| `capillary_diameter_cm` | `Float64` | `0.0008` | Terminal (capillary) diameter cutoff (legacy; use `terminal_diameter_cm`) |
| `terminal_diameter_cm` | `Float64` | `capillary_diameter_cm` | Murray-strict terminal (leaf) diameter for the growth phase. Root diameter scales as `d_term × N_terminals^(1/γ)`. |
| `subdivision_terminal_diameter_cm` | `Float64` | `0.0` | If > 0 and < `terminal_diameter_cm`, run post-growth recursive bifurcation on every terminal until segment diameter reaches this value. `0` disables subdivision. |
| `max_new_branches_per_tree` | `Int` | `220` | Maximum branches to add per tree |
| `graph_neighbors` | `Int` | `12` | k for the k-nearest-neighbor domain graph |
| `min_frontier_separation_cm` | `Float64` | `0.18` | Minimum spacing between frontier targets in a batch |
| `max_path_nodes` | `Int` | `20` | Maximum waypoints per branch path |
| `frontier_batch` | `Int` | `8` | Number of frontier targets per tree per round |
| `murray_gamma` | `Float64` | `3.0` | Exponent for Murray's law diameter propagation |
| `max_segment_length_cm` | `Float64` | `0.1` | Maximum length of a single segment; longer segments are densified |
| `smooth_passes` | `Int` | `20` | Laplacian smoothing iterations on branch paths |
| `spline_density` | `Int` | `5` | Catmull-Rom interpolation points per span |
| `coverage_stride` | `Int` | `2` | Block-stride for coverage target point sampling |
| `graph_stride` | `Int` | `0` | Block-stride for graph routing points (0 = same as `coverage_stride`) |
| `graph_jitter_cm` | `Float64` | `0.0` | Random jitter applied to graph points (0 = disabled) |
| `target_p95_distance_cm` | `Float64` | `Inf` | Stop growing when P95 distance drops below this |
| `target_max_distance_cm` | `Float64` | `Inf` | Stop growing when max distance drops below this |

### `[seed_points]` (only for `mode = "seed_point"`)

Key-value pairs where each key is a tree name and each value is a `[x, y, z]` array
in cm:

```toml
[seed_points]
LAD = [2.5, -1.0, 3.0]
LCX = [2.6, -0.8, 2.9]
```

## Growth Modes

### `continue_from_xcat`

Extracts centerlines from XCAT NURBS vessel surfaces, assembles them into
`XCATCenterlineTree` structures, and converts those into `GrowthTree` objects.
New branches are grown outward from the tips and along the existing tree.

```toml
[growth]
mode = "continue_from_xcat"
```

This is the default mode. It requires that the `surface_names` listed in each
`[[vessel_trees]]` entry match surfaces present in the NRB file.

### `seed_point`

Creates a minimal single-vertex `GrowthTree` at each specified seed coordinate.
All branches are grown from scratch. Useful when XCAT vessel surfaces are
unavailable or when modeling non-coronary organs.

```toml
[growth]
mode = "seed_point"

[seed_points]
Tree1 = [2.5, -1.0, 3.0]
Tree2 = [2.6, -0.8, 2.9]
```

Each key under `[seed_points]` must match a `name` in one of the `[[vessel_trees]]`
entries.

## Extending to New Organs

To add support for a new organ (e.g. liver, kidney):

1. **Obtain an XCAT NRB file** containing the organ's outer surface, cavity
   surfaces, and (optionally) vessel surfaces.

2. **Identify surface names** by parsing the NRB:
   ```julia
   surfaces = parse_xcat_nrb("/path/to/organ.nrb")
   for s in surfaces
       println(s.name)
   end
   ```

3. **Create a new TOML config** (e.g. `configs/liver.toml`):
   ```toml
   [organ]
   name = "liver"
   nrb_path = "/path/to/organ.nrb"
   coordinate_scale = 0.1

   [surfaces]
   outer = "liver_capsule"
   cavities = []
   reference = ""

   [[vessel_trees]]
   name = "HepaticArtery"
   surface_names = []
   color = "#ff4444"

   [domain]
   voxel_spacing_cm = 0.05

   [growth]
   mode = "seed_point"

   [seed_points]
   HepaticArtery = [5.0, 2.0, 1.0]
   ```

4. **Tune domain parameters** -- adjust `voxel_spacing_cm`, `outer_samples`, and
   `cavity_samples` for the organ's size and complexity.

5. **Tune growth parameters** -- start with defaults and iterate on
   `max_new_branches_per_tree`, `min_frontier_separation_cm`, and
   `effective_supply_radius_cm` to achieve desired density.

6. **Run**:
   ```julia
   config = load_organ_config("configs/liver.toml")
   run_growth(config; output_dir="output_liver")
   ```

## Output Format

Each tree produces a CSV file (e.g. `lad_grown_segments.csv`) with the following
columns:

| Column | Type | Unit | Description |
|---|---|---|---|
| `branch` | `String` | -- | Tree name (e.g. `"LAD"`) |
| `segment_id` | `Int` | -- | 1-based segment index |
| `parent_segment_id` | `Int` | -- | ID of the parent segment (0 for the root segment) |
| `x1_cm` | `Float64` | cm | Start-point x coordinate |
| `y1_cm` | `Float64` | cm | Start-point y coordinate |
| `z1_cm` | `Float64` | cm | Start-point z coordinate |
| `x2_cm` | `Float64` | cm | End-point x coordinate |
| `y2_cm` | `Float64` | cm | End-point y coordinate |
| `z2_cm` | `Float64` | cm | End-point z coordinate |
| `xmid_cm` | `Float64` | cm | Midpoint x coordinate |
| `ymid_cm` | `Float64` | cm | Midpoint y coordinate |
| `zmid_cm` | `Float64` | cm | Midpoint z coordinate |
| `length_mm` | `Float64` | mm | Segment length |
| `diameter_um` | `Float64` | um | Segment diameter |
| `label` | `String` | -- | `"grown"` for new branches; original XCAT surface name otherwise |

The `parent_segment_id` column enables deterministic reconstruction of the full tree
topology from the flat CSV.

## API Reference

### Configuration

```julia
load_organ_config(path::AbstractString) -> OrganConfig
```
Parse a TOML file and return a fully populated `OrganConfig`.

### NRB Parsing

```julia
parse_xcat_nrb(path::AbstractString) -> Vector{XCATNurbsSurface}
```
Read an XCAT `.nrb` file and return all NURBS surfaces.

```julia
xcat_object_dict(surfaces) -> Dict{String, XCATNurbsSurface}
```
Index surfaces by name for convenient lookup.

```julia
xcat_sample_surface(surface; n_u, n_v, orient_outward) -> (points, normals, ...)
```
Evaluate a NURBS surface on a regular `(n_u, n_v)` parameter grid.

### Centerline Extraction

```julia
xcat_centerline_from_surface(surface::XCATNurbsSurface) -> XCATCenterline
```
Extract a centerline (center points + radii) from a tubular NURBS surface.

```julia
build_vessel_trees(centerlines, config::OrganConfig) -> Dict{String, XCATCenterlineTree}
```
Assemble centerlines into connected tree structures based on config vessel specs.

### Domain Construction

```julia
build_voxel_shell_domain_floodfill(outer, cavities; ...) -> VoxelShellDomain
```
Build a 3-D voxel mask of the organ wall via flood fill between outer and cavity
surfaces.

```julia
coverage_target_points(domain; stride) -> Matrix{Float64}
coverage_target_points_blockwise(domain; block_size) -> Matrix{Float64}
```
Sample coverage target points from the domain mask.

### Tree Initialization

```julia
growth_tree_from_xcat(name::String, tree::XCATCenterlineTree) -> GrowthTree
```
Convert an XCAT centerline tree into a mutable `GrowthTree`.

```julia
growth_tree_from_seed(name::String, seed_point_cm::SVector{3,Float64};
                      terminal_diameter_cm::Float64=0.004) -> GrowthTree
```
Create a single-vertex `GrowthTree` at the given seed coordinate.

```julia
subdivide_terminals!(tree::GrowthTree;
                     target_diameter_cm::Float64,
                     gamma::Float64=3.0,
                     branch_half_angle::Float64=0.4,
                     ld_ratio::Float64=12.0,
                     domain::Union{Nothing, VoxelShellDomain}=nothing)
```
Recursively bifurcate every terminal until segment diameter reaches
`target_diameter_cm`. Pass `domain` to clip sub-branches that would leave the
voxel mask (uses 8-attempt rotation retry per split).

```julia
smooth_junction_taper!(tree::GrowthTree; gamma::Float64=3.0)
```
Optional post-pass: at each XCAT↔grown junction, distribute the Murray-residual
flow budget among grown children and apply an exponentially decaying diameter
boost through their subtrees. Not needed in the strict-Murray regime (all
diameters are already Murray-derived).

```julia
point_in_domain(domain::VoxelShellDomain, pt) -> Bool
```
Test whether a point (cm) lies inside the voxel mask. Used internally by
`subdivide_terminals!` for boundary-aware clipping.

### Graph and Routing

```julia
build_domain_graph(points_cm::Matrix{Float64}, domain; k) -> DomainGraph
```
Build a k-nearest-neighbor graph over domain points with midwall-cost-weighted edges.

### Growth

```julia
grow_trees_mcp!(trees::Dict{String,GrowthTree}, domain; ...) -> (graph, territories, stats)
```
Run competitive round-robin MCP growth on all trees simultaneously. Returns the
domain graph, per-tree territory indices, and per-tree statistics (terminals, P50,
P95, max distance, branches added).

### Export

```julia
write_growth_csv(path, branch_name, tree::GrowthTree) -> path
```
Write the tree to a CSV file with topology columns.

```julia
growth_viewer_html(path, domain, trees, stats, color_map)
```
Generate an interactive Plotly-based HTML viewer.

### Top-Level

```julia
run_growth(config::OrganConfig; output_dir="output") -> NamedTuple
```
End-to-end pipeline: parse NRB, build domain, initialize trees, grow, export CSVs
and viewer. Returns a named tuple with fields `html_path`, `domain`,
`coverage_points`, `trees`, `territories`, and `stats`.

## Algorithms

### Minimum-Cost-Path (MCP) Growth

Each growth round selects frontier targets -- domain points that are farthest from
any existing vessel segment and owned by the current tree's territory. For each
target, the nearest existing tree vertex is chosen as an anchor, and Dijkstra's
algorithm finds the lowest-cost path through the domain graph from anchor to target.
The path is smoothed, resampled, and added as a new branch.

### Heap-Based Dijkstra

Shortest paths are computed using a binary-heap priority queue implementation
(`_shortest_path`). Edge costs combine Euclidean distance with a midwall-proximity
penalty so that paths prefer the interior of the myocardial shell over regions near
surfaces.

### Catmull-Rom Smoothing

Raw graph paths are resampled using Catmull-Rom spline interpolation
(`_catmull_rom_resample`) with configurable `spline_density` points per span. This is
followed by domain-constrained Laplacian smoothing (`_smooth_path_in_domain`) that
averages each interior waypoint with its neighbors while rejecting moves that exit
the domain mask.

### Strict Murray-Derived Diameters

Every segment diameter -- XCAT and grown -- is derived purely from its subtree
terminal count:

```
d = d_term × N_terminals^(1/γ)
```

where `γ` defaults to 3.0 (classic Murray) and `d_term` is the tree's terminal
(leaf) diameter. This makes Murray's law hold at every junction by construction,
not just where branches are added. The XCAT ostium diameter acts only as a
capacity ceiling: before adding a new terminal, `_can_add_terminal_at_anchor`
rejects the branch if `total_terminals + 1 > (d_root / d_term)^γ`.

### Hybrid Growth + Subdivision

Direct growth to 8 μm terminals saturates coverage at ~10⁴ terminals, which under
Murray gives sub-mm root diameters (unphysiological). The hybrid pipeline splits
the problem:

1. **Growth phase** at `terminal_diameter_cm` (e.g. 200 μm) -- the MCP routing
   layer pushes branches until coverage stalls. This yields realistic
   mm-scale root diameters.
2. **Subdivision phase** at `subdivision_terminal_diameter_cm` (e.g. 8 μm) --
   `subdivide_terminals!` recursively bifurcates each tip with symmetric Murray
   splits (`d_child = d_parent / 2^(1/γ)`), doubling the tree depth until the
   target diameter is reached. `_recompute_all_murray!` then walks the whole
   tree bottom-up and re-derives every diameter from the new leaf counts, so the
   `d = d_term × N^(1/γ)` identity remains globally exact.

Subdivision is domain-aware: each candidate child vertex is tested against the
voxel mask, and up to eight random rotations of the bifurcation plane are tried
to find one where both children land inside the shell. Without the rotation
retry, a single unlucky direction near the boundary can cascade into losing
~2^depth descendants and producing visible holes in the perfused tissue.

### Chamber vs. Tubular Cavity Handling

Cardiac cavities span two very different shape classes: chambers (LV, RV, LA,
RA) whose centroids land inside the organ wall, and tubes (aorta, SVC, IVC)
whose centroids fall outside. `build_voxel_shell_domain_floodfill` auto-detects
the class from the centroid's location relative to `outer_interior` and:

- For chambers: standard SDF subtraction + centroid-seeded flood fill.
- For tubes: SDF subtraction bounded at 2 cm (Euclidean distance to nearest
  surface sample), plus a fallback seed that scans for any allowed voxel if the
  centroid-based seed is invalid. This prevents the unbounded SDF from
  subtracting large regions of perfectly valid wall far from the tube.

### Competitive Round-Robin

Multiple trees grow simultaneously in a round-robin schedule. A global distance
array tracks, for every coverage point, the nearest segment across all trees and
which tree owns it. Each round, every tree selects frontier targets only from its
own territory (points closer to it than to any other tree). After a tree adds
branches, the global distances are incrementally updated. This produces realistic
territorial partitioning analogous to coronary perfusion territories.

## References

1. **Kassab, G.S. et al.** (1993). "Morphometry of pig coronary arterial trees."
   *American Journal of Physiology -- Heart and Circulatory Physiology*, 265(1),
   H350--H365. Foundational morphometric data for coronary arterial branching.

2. **Huo, Y. and Kassab, G.S.** (2007). "A scaling law of vascular volume."
   *Biophysical Journal*, 92(10), 3554--3562. Scaling relationships used to
   calibrate supply radii and terminal diameters.

3. **Murray, C.D.** (1926). "The physiological principle of minimum work. I. The
   vascular system and the cost of blood volume." *Proceedings of the National
   Academy of Sciences*, 12(3), 207--214. The cubic law (`d^3 = sum d_i^3`)
   governing optimal vessel diameter at bifurcations.
