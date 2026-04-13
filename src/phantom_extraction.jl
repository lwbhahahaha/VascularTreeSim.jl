"""
    Phantom structure extraction from XCAT raw volumes.

Extracts cardiac structures (myocardium, blood pools, aorta, coronary arteries)
as point clouds from the raw phantom. This is step 1 of the growth pipeline —
must run before tree growth to establish the domain and reference anatomy.
"""

# XCAT activity label map (heart region)
const XCAT_LABELS = Dict{String, UInt8}(
    "myocardium_lv"     => 15,
    "myocardium_rv"     => 16,
    "myocardium_la"     => 17,
    "myocardium_ra"     => 18,
    "bloodpool_lv"      => 19,
    "bloodpool_rv"      => 20,
    "bloodpool_la"      => 21,
    "bloodpool_ra"      => 22,
    "aorta"             => 23,
    "pulmonary_artery"  => 24,
    "pulmonary_veins"   => 25,
    "coronary_arteries" => 26,
    "coronary_veins"    => 27,
    "vena_cava"         => 28,
    "pericardium"       => 29,
)

# Extraction groups: which labels belong to each structure
# NOTE: Label 23 is body surface, NOT aorta. Label 28 = great vessels near heart.
const EXTRACTION_GROUPS = Dict{String, Vector{UInt8}}(
    "domain"             => UInt8[15, 16, 17, 18],   # myocardium = growth domain
    "chambers"           => UInt8[19, 20, 21, 22],   # blood pools (heart cavities)
    "great_vessels"      => UInt8[28],                # IVC/SVC/great vessels
    "coronary_arteries"  => UInt8[26],                # existing XCAT coronary arteries
    "pericardium"        => UInt8[29],                # pericardium (outer heart envelope)
)

# Default strides per group (larger = fewer points, faster rendering)
# Domain uses stride=1 for complete coverage of all myocardium voxels
const EXTRACTION_STRIDES = Dict{String, Int}(
    "domain"            => 1,
    "chambers"          => 6,
    "great_vessels"     => 4,
    "coronary_arteries" => 2,
    "pericardium"       => 4,
)

"""
    PhantomData

Container for loaded XCAT phantom and its metadata.
"""
struct PhantomData
    volume::Array{UInt8, 3}
    nx::Int
    ny::Int
    nz::Int
    voxel_cm::Float64
end

"""
    load_xcat_phantom(path; nx=1600, ny=1400, nz=500, voxel_cm=0.02)

Load an XCAT phantom from a raw binary file (UInt8, column-major / Fortran order).
"""
function load_xcat_phantom(path::AbstractString; nx::Int=1600, ny::Int=1400, nz::Int=500,
                           voxel_cm::Float64=0.02)
    println("[phantom] Loading $(nx)×$(ny)×$(nz) phantom from $(basename(path))...")
    t0 = time()
    data = Vector{UInt8}(undef, nx * ny * nz)
    open(path, "r") do io
        read!(io, data)
    end
    volume = reshape(data, (nx, ny, nz))
    println("[phantom] Loaded in $(round(time()-t0; digits=1))s")
    return PhantomData(volume, nx, ny, nz, voxel_cm)
end

"""
    extract_structure_points(phantom, labels; stride=4)

Extract point cloud (Nx3 Matrix{Float64}, in cm) for voxels matching any label in `labels`.
Samples every `stride` voxels in each dimension.
"""
function extract_structure_points(phantom::PhantomData, labels::Vector{UInt8}; stride::Int=4)
    label_set = Set(labels)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    vol = phantom.volume
    vcm = phantom.voxel_cm
    for k in 1:stride:phantom.nz, j in 1:stride:phantom.ny, i in 1:stride:phantom.nx
        if vol[i, j, k] in label_set
            push!(xs, round((i - 0.5) * vcm; digits=4))
            push!(ys, round((j - 0.5) * vcm; digits=4))
            push!(zs, round((k - 0.5) * vcm; digits=4))
        end
    end
    return xs, ys, zs
end

"""
    save_structure_csv(xs, ys, zs, path)

Save point cloud as CSV with columns x_cm, y_cm, z_cm.
"""
function save_structure_csv(xs::Vector{Float64}, ys::Vector{Float64}, zs::Vector{Float64},
                            path::AbstractString)
    open(path, "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for (x, y, z) in zip(xs, ys, zs)
            println(io, "$x,$y,$z")
        end
    end
    return path
end

"""
    extract_all_structures(phantom, output_dir; timestamp=true, strides=EXTRACTION_STRIDES, groups=EXTRACTION_GROUPS)

Extract all cardiac structures from phantom and save as CSVs.
Returns a Dict mapping group name → (xs, ys, zs).

Saves both timestamped and "latest" versions of each CSV:
- `{group}_{timestamp}.csv` — for reproducibility
- `{group}_points.csv` — latest version for viewer
"""
function extract_all_structures(phantom::PhantomData, output_dir::AbstractString;
                                timestamp::Bool=true,
                                strides::Dict{String, Int}=EXTRACTION_STRIDES,
                                groups::Dict{String, Vector{UInt8}}=EXTRACTION_GROUPS)
    mkpath(output_dir)
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()

    for (group_name, label_list) in groups
        stride = get(strides, group_name, 4)
        println("[phantom] Extracting $(group_name) (labels=$(label_list), stride=$(stride))...")
        xs, ys, zs = extract_structure_points(phantom, label_list; stride=stride)
        n = length(xs)
        if n == 0
            @warn "[phantom] No points found for $(group_name)"
            continue
        end
        println("[phantom]   → $(n) points, x=[$(round(minimum(xs); digits=2)),$(round(maximum(xs); digits=2))] " *
                "y=[$(round(minimum(ys); digits=2)),$(round(maximum(ys); digits=2))] " *
                "z=[$(round(minimum(zs); digits=2)),$(round(maximum(zs); digits=2))]")

        # Save latest version (for viewer)
        latest_path = joinpath(output_dir, "$(group_name)_points.csv")
        save_structure_csv(xs, ys, zs, latest_path)
        println("[phantom]   Saved: $(latest_path)")

        # Save timestamped version (for reproducibility)
        if timestamp
            ts_path = joinpath(output_dir, "$(group_name)_$(ts).csv")
            save_structure_csv(xs, ys, zs, ts_path)
            println("[phantom]   Saved: $(ts_path)")
        end

        results[group_name] = (xs, ys, zs)
    end

    println("[phantom] Extraction complete: $(length(results)) groups")
    return results
end
