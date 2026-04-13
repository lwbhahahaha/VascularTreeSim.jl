"""
    Voxelizer — rasterize GrowthTrees into XCAT phantom or standalone volume.

Adapted from examples/tree_to_raw.jl and examples/xcat_coronary_to_raw.jl.
Each vessel segment is rendered as a capsule (cylinder + hemispherical caps).
"""

using LinearAlgebra: dot

# ── NRB → phantom coordinate transform ──
# Empirically determined from vmale50: identity permutation + translation.
# phantom_xyz = nrb_xyz + NRB_TO_PHANTOM_OFFSET
const NRB_TO_PHANTOM_OFFSET = SVector(2.1443, -9.5553, -20.0068)

"""
    nrb_to_phantom(p::SVector{3,Float64}) -> SVector{3,Float64}

Transform a point from NRB coordinates (cm) to phantom voxel coordinates (cm).
"""
nrb_to_phantom(p::SVector{3,Float64}) = p + NRB_TO_PHANTOM_OFFSET
nrb_to_phantom(p) = SVector{3,Float64}(p[1], p[2], p[3]) + NRB_TO_PHANTOM_OFFSET

"""
    embed_trees_in_phantom!(phantom, trees; kwargs...) -> stats

Write grown coronary trees into a copy of the XCAT phantom volume.
- Existing XCAT coronary arteries (label 26) and veins (label 27) → label 125
- Grown vessel segments → label 255
- Only writes to myocardium voxels (labels 15-18) or existing coronary (125)

The trees are in NRB coordinates; this function applies the NRB→phantom transform.

# Arguments
- `phantom`: UInt8 3D array (1600×1400×500), will be modified in-place
- `trees`: Dict{String, GrowthTree} in NRB coordinates
- `voxel_cm=0.02`: phantom voxel spacing
- `min_render_diameter_cm=0.005`: minimum rendered vessel diameter
"""
function embed_trees_in_phantom!(phantom::Array{UInt8, 3},
                                 trees::Dict{String, GrowthTree};
                                 voxel_cm::Float64=0.02,
                                 min_render_diameter_cm::Float64=0.005)
    nx, ny, nz = size(phantom)

    # Step A: Mark existing XCAT coronary arteries/veins as 125
    t0 = time()
    n_marked = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        lbl = phantom[i, j, k]
        if lbl == UInt8(26) || lbl == UInt8(27)
            phantom[i, j, k] = UInt8(125)
            n_marked += 1
        end
    end
    println("[embed] XCAT coronaries (26,27) → label 125: $(n_marked) voxels ($(round(time()-t0;digits=1))s)")
    flush(stdout)

    # Writable labels: myocardium (15-18) + already-marked coronaries (125)
    writable = Set{UInt8}([UInt8(15), UInt8(16), UInt8(17), UInt8(18), UInt8(125)])

    # Step B: Rasterize grown trees as label 255, constrained to myocardium
    branch_names = sort(collect(keys(trees)))
    total_written = 0
    total_clipped = 0

    for name in branch_names
        tree = trees[name]
        nseg = length(tree.segment_start)
        nt = Threads.nthreads()
        written_per_thread = zeros(Int, nt)
        clipped_per_thread = zeros(Int, nt)

        Threads.@threads for s in 1:nseg
            tid = Threads.threadid()
            # Transform segment endpoints from NRB → phantom coordinates
            a_nrb = tree.vertices[tree.segment_start[s]]
            b_nrb = tree.vertices[tree.segment_end[s]]
            a = nrb_to_phantom(a_nrb)
            b = nrb_to_phantom(b_nrb)
            d_cm = max(tree.segment_diameter_cm[s], min_render_diameter_cm)
            r_cm = d_cm / 2.0

            seg_lo = min.(a, b) .- r_cm
            seg_hi = max.(a, b) .+ r_cm

            i0 = max(1, floor(Int, seg_lo[1] / voxel_cm) + 1)
            j0 = max(1, floor(Int, seg_lo[2] / voxel_cm) + 1)
            k0 = max(1, floor(Int, seg_lo[3] / voxel_cm) + 1)
            i1 = min(nx, ceil(Int, seg_hi[1] / voxel_cm) + 1)
            j1 = min(ny, ceil(Int, seg_hi[2] / voxel_cm) + 1)
            k1 = min(nz, ceil(Int, seg_hi[3] / voxel_cm) + 1)

            ab = b - a
            ab_len2 = dot(ab, ab)

            for kk in k0:k1, jj in j0:j1, ii in i0:i1
                # Only write to myocardium or existing coronary voxels
                phantom[ii, jj, kk] in writable || continue

                px = (ii - 0.5) * voxel_cm
                py = (jj - 0.5) * voxel_cm
                pz = (kk - 0.5) * voxel_cm

                apx = px - a[1]; apy = py - a[2]; apz = pz - a[3]
                if ab_len2 <= 1e-24
                    dist2 = apx*apx + apy*apy + apz*apz
                else
                    t = clamp((apx*ab[1] + apy*ab[2] + apz*ab[3]) / ab_len2, 0.0, 1.0)
                    dx = apx - t*ab[1]; dy = apy - t*ab[2]; dz = apz - t*ab[3]
                    dist2 = dx*dx + dy*dy + dz*dz
                end

                if dist2 <= r_cm * r_cm
                    phantom[ii, jj, kk] = UInt8(255)
                    written_per_thread[tid] += 1
                end
            end
        end

        n_written = sum(written_per_thread)
        total_written += n_written
        println("[embed] $(name): $(nseg) segments → $(n_written) voxels written")
        flush(stdout)
    end

    total_125 = count(==(UInt8(125)), phantom)
    total_255 = count(==(UInt8(255)), phantom)
    println("[embed] Final: XCAT original (125)=$(total_125)  grown (255)=$(total_255)  total_writes=$(total_written)")
    flush(stdout)

    return (xcat_voxels=total_125, grown_voxels=total_255, total_writes=total_written)
end

"""
    write_phantom_raw(path, phantom; info_path=nothing, trees=nothing, stats=nothing)

Write full XCAT phantom as raw binary file. Optionally writes companion info.txt.
"""
function write_phantom_raw(path::String, phantom::Array{UInt8, 3};
                           info_path::Union{Nothing, String}=nothing,
                           trees::Union{Nothing, Dict{String, GrowthTree}}=nothing,
                           embed_stats::Union{Nothing, NamedTuple}=nothing,
                           growth_stats::Union{Nothing, Dict}=nothing)
    t0 = time()
    nx, ny, nz = size(phantom)
    open(path, "w") do io
        write(io, phantom)
    end
    fsize = filesize(path)
    println("[write_raw] $(basename(path)): $(nx)×$(ny)×$(nz) = $(round(fsize/1024^2;digits=0)) MB ($(round(time()-t0;digits=1))s)")
    flush(stdout)

    if info_path !== nothing
        open(info_path, "w") do io
            println(io, "XCAT Phantom with Grown Coronary Trees")
            println(io, "=" ^ 50)
            println(io, "")
            println(io, "Full phantom:")
            println(io, "  File: $(basename(path))")
            println(io, "  Dimensions: $(nx) × $(ny) × $(nz)")
            println(io, "  Data type: UInt8")
            println(io, "  Voxel size: 0.2 mm isotropic")
            println(io, "  Byte order: little-endian")
            println(io, "")
            println(io, "ImageJ Import:")
            println(io, "  File → Import → Raw...")
            println(io, "  Image type: 8-bit")
            println(io, "  Width: $(nx)")
            println(io, "  Height: $(ny)")
            println(io, "  Number of images: $(nz)")
            println(io, "  Little-endian byte order")
            println(io, "")
            println(io, "Key labels:")
            println(io, "  0   = background/air")
            println(io, "  15  = LV myocardium")
            println(io, "  16  = RV myocardium")
            println(io, "  17  = LA myocardium")
            println(io, "  18  = RA myocardium")
            println(io, "  19  = LV blood pool")
            println(io, "  20  = RV blood pool")
            println(io, "  125 = XCAT original coronary arteries/veins")
            println(io, "  255 = grown coronary vessels")
            println(io, "  29  = pericardium")
            println(io, "")
            if embed_stats !== nothing
                println(io, "Voxel counts:")
                println(io, "  XCAT coronaries (125): $(embed_stats.xcat_voxels)")
                println(io, "  Grown vessels   (255): $(embed_stats.grown_voxels)")
                println(io, "")
            end
            if growth_stats !== nothing && trees !== nothing
                println(io, "Tree stats:")
                for name in sort(collect(keys(trees)))
                    if haskey(growth_stats, name)
                        st = growth_stats[name]
                        nseg = length(trees[name].segment_start)
                        println(io, "  $(name): $(nseg) segments, $(st.terminals) terminals, p95=$(round(st.p95*10;digits=2))mm")
                    end
                end
            end
        end
    end

    return fsize
end

# ── Standalone domain voxelizer (kept for non-phantom use) ──

"""
    voxelize_trees(trees, domain; kwargs...) -> volume, info

Standalone voxelizer: rasterize trees into a domain-sized UInt8 volume.
Labels: 0=air, 1=tissue, 2+=vessels.
"""
function voxelize_trees(trees::Dict{String, GrowthTree}, domain::VoxelShellDomain;
                        min_render_diameter_cm::Float64=0.005,
                        diameter_scale::Float64=1.0,
                        constrain_to_domain::Bool=true)
    nx, ny, nz = size(domain.mask)
    volume = zeros(UInt8, nx, ny, nz)
    origin = domain.origin_cm
    sp = domain.spacing_cm

    Threads.@threads for k in 1:nz
        for j in 1:ny, i in 1:nx
            if domain.mask[i, j, k]
                volume[i, j, k] = UInt8(1)
            end
        end
    end

    branch_names = sort(collect(keys(trees)))
    info = Dict{String, NamedTuple}()

    for (ti, name) in enumerate(branch_names)
        label = UInt8(min(ti + 1, 255))
        tree = trees[name]
        nseg = length(tree.segment_start)
        nt = Threads.nthreads()
        thread_counts = zeros(Int, nt)

        Threads.@threads for s in 1:nseg
            tid = Threads.threadid()
            a = tree.vertices[tree.segment_start[s]]
            b = tree.vertices[tree.segment_end[s]]
            d_cm = max(tree.segment_diameter_cm[s] * diameter_scale, min_render_diameter_cm)
            r_cm = d_cm / 2.0

            seg_lo = min.(a, b) .- r_cm
            seg_hi = max.(a, b) .+ r_cm

            i0 = max(1, floor(Int, (seg_lo[1] - origin[1]) / sp[1]) + 1)
            j0 = max(1, floor(Int, (seg_lo[2] - origin[2]) / sp[2]) + 1)
            k0 = max(1, floor(Int, (seg_lo[3] - origin[3]) / sp[3]) + 1)
            i1 = min(nx, ceil(Int, (seg_hi[1] - origin[1]) / sp[1]) + 1)
            j1 = min(ny, ceil(Int, (seg_hi[2] - origin[2]) / sp[2]) + 1)
            k1 = min(nz, ceil(Int, (seg_hi[3] - origin[3]) / sp[3]) + 1)

            ab = b - a
            ab_len2 = dot(ab, ab)

            for kk in k0:k1, jj in j0:j1, ii in i0:i1
                if constrain_to_domain && !domain.mask[ii, jj, kk]
                    continue
                end
                px = origin[1] + (ii - 0.5) * sp[1]
                py = origin[2] + (jj - 0.5) * sp[2]
                pz = origin[3] + (kk - 0.5) * sp[3]

                apx = px - a[1]; apy = py - a[2]; apz = pz - a[3]
                if ab_len2 <= 1e-24
                    dist2 = apx*apx + apy*apy + apz*apz
                else
                    t = clamp((apx*ab[1] + apy*ab[2] + apz*ab[3]) / ab_len2, 0.0, 1.0)
                    dx = apx - t*ab[1]; dy = apy - t*ab[2]; dz = apz - t*ab[3]
                    dist2 = dx*dx + dy*dy + dz*dz
                end

                if dist2 <= r_cm * r_cm
                    volume[ii, jj, kk] = label
                    thread_counts[tid] += 1
                end
            end
        end

        n_filled = sum(thread_counts)
        n_vol = count(==(label), volume)
        info[name] = (label=Int(label), segments=nseg, voxels_filled=n_filled, voxels_final=n_vol)
        println("[voxelize] $(name): label=$(label) segments=$(nseg) voxels=$(n_vol)")
        flush(stdout)
    end

    return volume, info
end
