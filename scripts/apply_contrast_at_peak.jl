#!/usr/bin/env julia
#
# apply_contrast_at_peak.jl — voxelize coronary trees into a UInt16 XCAT phantom
# encoding **both** blood volume fraction and iodine concentration (cross-product
# 100×101 grid) at the peak contrast time.
#
# Input:
#   - TREE_DIR              tree segment CSVs (from VascularTreeSim)
#   - PHANTOM_RAW_IN        original XCAT UInt8 raw
#   - PEAK_IODINE_DIR       output of FlowContrastSim/scripts/extract_peak_iodine.jl,
#                           contains {tree}_peak_iodine.f32 + peak_metadata.toml
#   - OUTPUT_DIR            where to write the modified UInt16 raw + manifest
#
# Per voxel (true occupied-volume "tofu" measure, no MBV imposed):
#   f_blood        = Σ (vessel ∩ voxel volume) / voxel_volume        [clipped to 1.0]
#                    resolvable vessels (≥200 μm): sub-voxel MC cross-section
#                    sub-voxel capillaries:        π r²·(centerline length in voxel)
#   f_iodine_w     = Σ (same volume fraction × C_iodine_segment)
#   C_iodine_voxel = f_iodine_w / f_blood                            [iodine mg/mL of blood]
#
# Cross-product encoding (UInt16):
#   bin_b = round(f_blood × N_BLOOD_BINS),                bin_b ∈ 1..100  (b=0 → no change)
#   bin_i = round(C_iodine_voxel / iodine_max × N_IODINE_BINS), bin_i ∈ 0..100
#   label = LABEL_BASE + (bin_b - 1) × (N_IODINE_BINS + 1) + bin_i
#         ∈ [256, 10355]
#
# Usage:
#   julia --project=. --threads=auto scripts/apply_contrast_at_peak.jl \
#         TREE_DIR  PHANTOM_RAW_IN  PEAK_IODINE_DIR  OUTPUT_DIR  [N_SUB]

using LinearAlgebra
using Base.Threads
using Printf
using TOML

# ── Phantom geometry (vmale50, 0.02 cm isotropic) ────────────────────────────
const PHANTOM_DIMS = (1600, 1400, 500)
const VOXEL_CM = 0.02
const NRB_TO_PHANTOM_OFFSET = (2.1443, -9.5553, -20.0068)
const XCAT_ORIGIN_CM = (2.846980, -9.773884, -20.600891)

# ── Label scheme (cross-product) ─────────────────────────────────────────────
const N_BLOOD_BINS = 100      # 1..100  (bin 0 → keep original myo label)
const N_IODINE_BINS = 100     # 0..100  (0 = no iodine, 100 = at iodine_max)
const LABEL_BASE = UInt16(256)
# Encoding: label = LABEL_BASE + (bin_b - 1) * (N_IODINE_BINS + 1) + bin_i
# Max: 256 + 99 * 101 + 100 = 10355  (fits comfortably in UInt16)
const MAX_LABEL = LABEL_BASE + UInt16((N_BLOOD_BINS - 1) * (N_IODINE_BINS + 1) + N_IODINE_BINS)
const MYO_LABELS = (UInt16(15), UInt16(16), UInt16(17), UInt16(18))

# ── Render tuning ────────────────────────────────────────────────────────────
const DEFAULT_N_SUB = 5
# Crossover diameter between the two volume-measurement methods. Vessels ≥ this
# (≈ voxel size) are resolvable: sub-voxel MC gives their cross-sectional volume
# fraction. Vessels below it are sub-resolution: MC misses them, so we deposit
# their exact lumen volume π r²·(centerline length in voxel) instead. Both yield
# the same quantity — (vessel ∩ voxel) volume / voxel volume — so the total
# myocardial blood volume EMERGES from the grown geometry (no MBV imposed).
const MIN_VOXELIZE_DIAM_CM = 0.02f0
# Parse-time filter (keep 0.0 so segment↔peak_iodine indexing stays aligned).
const MIN_RENDER_DIAMETER_CM = 0.0

const OUT_RAW_BASENAME = "vmale50_with_grown_coronaries_peak_contrast_u16.raw"

# ─────────────────────────────────────────────────────────────────────────────

@inline is_myo(v::UInt16) =
    v == MYO_LABELS[1] || v == MYO_LABELS[2] ||
    v == MYO_LABELS[3] || v == MYO_LABELS[4]

function load_phantom_raw_u16(path::String)
    nx, ny, nz = PHANTOM_DIMS
    expected_u8 = nx * ny * nz
    actual = filesize(path)
    actual == expected_u8 ||
        error("Source phantom must be UInt8 ($expected_u8 B); got $actual B at $path")
    src = Array{UInt8}(undef, nx, ny, nz)
    t0 = time()
    read!(path, src)
    dst = Array{UInt16}(undef, nx, ny, nz)
    @threads :static for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            dst[i, j, k] = UInt16(src[i, j, k])
        end
    end
    @printf("[load] %s: %d×%d×%d UInt8 → UInt16 (%.1fs)\n",
            basename(path), nx, ny, nz, time()-t0)
    flush(stdout)
    dst
end

function label_histogram(phantom::Array{UInt16,3})
    nx, ny, nz = size(phantom)
    counts = [zeros(Int64, 65536) for _ in 1:nthreads()]
    @threads :static for k in 1:nz
        c = counts[threadid()]
        @inbounds for j in 1:ny, i in 1:nx
            c[Int(phantom[i,j,k]) + 1] += 1
        end
    end
    hist = zeros(Int64, 65536)
    for c in counts;  hist .+= c;  end
    hist
end

# Parse a *_segments.csv → 7×Nseg Float32 matrix [ax;ay;az;bx;by;bz;r_cm]
# (NRB→phantom shift applied). Row i in returned matrix corresponds to CSV
# segment_id i (= FlowContrastSim internal index i = peak_iodine.f32 index i).
function parse_segments_csv(csv_path::String)
    ox, oy, oz = Float32.(NRB_TO_PHANTOM_OFFSET)
    rmin = Float32(MIN_RENDER_DIAMETER_CM / 2)

    t0 = time()
    n_newlines = 0
    open(csv_path, "r") do io
        buf = Vector{UInt8}(undef, 64 * 1024 * 1024)
        while !eof(io)
            nread = readbytes!(io, buf)
            @inbounds for k in 1:nread
                buf[k] == 0x0A && (n_newlines += 1)
            end
        end
    end
    n_segs = n_newlines - 1
    @printf("[csv] %s: %d rows (%.1fs)\n", basename(csv_path), n_segs, time()-t0)
    flush(stdout)

    A = Matrix{Float32}(undef, 7, n_segs)
    t0 = time()
    open(csv_path, "r") do io
        readline(io)  # header
        i = 0
        while !eof(io)
            line = readline(io)
            isempty(line) && continue
            i += 1
            c1  = findfirst(',', line);              c2  = findnext(',', line, c1+1)
            c3  = findnext(',', line, c2+1);          c4  = findnext(',', line, c3+1)
            c5  = findnext(',', line, c4+1);          c6  = findnext(',', line, c5+1)
            c7  = findnext(',', line, c6+1);          c8  = findnext(',', line, c7+1)
            c9  = findnext(',', line, c8+1);          c10 = findnext(',', line, c9+1)
            c11 = findnext(',', line, c10+1);         c12 = findnext(',', line, c11+1)
            c13 = findnext(',', line, c12+1);         c14 = findnext(',', line, c13+1)
            x1 = parse(Float32, SubString(line, c3+1,  c4-1))
            y1 = parse(Float32, SubString(line, c4+1,  c5-1))
            z1 = parse(Float32, SubString(line, c5+1,  c6-1))
            x2 = parse(Float32, SubString(line, c6+1,  c7-1))
            y2 = parse(Float32, SubString(line, c7+1,  c8-1))
            z2 = parse(Float32, SubString(line, c8+1,  c9-1))
            d_um = parse(Float32, SubString(line, c13+1, c14-1))
            @inbounds A[1,i] = x1 + ox;  @inbounds A[2,i] = y1 + oy;  @inbounds A[3,i] = z1 + oz
            @inbounds A[4,i] = x2 + ox;  @inbounds A[5,i] = y2 + oy;  @inbounds A[6,i] = z2 + oz
            @inbounds A[7,i] = max((d_um * Float32(1e-4)) / Float32(2), rmin)
        end
    end
    @printf("[csv] %s: parsed in %.1fs (%.2f Mrows/s)\n",
            basename(csv_path), time()-t0, n_segs / (time()-t0) / 1e6)
    flush(stdout)
    A
end

function load_peak_iodine_f32(path::String, n_segs::Int)
    expected = n_segs * 4
    actual = filesize(path)
    actual == expected ||
        error("peak_iodine mismatch at $path: $actual B vs $expected B (n_segs=$n_segs)")
    v = Vector{Float32}(undef, n_segs)
    read!(path, v)
    v
end

# Rasterize with sub-voxel MC, accumulating BOTH f_blood and f_blood*C_iodine.
function rasterize_capsules_vf_iodine!(f_blood::Array{Float32,3},
                                       f_iodine_w::Array{Float32,3},
                                       phantom::Array{UInt16,3},
                                       A::Matrix{Float32},
                                       c_iodine::Vector{Float32};
                                       n_sub::Int = DEFAULT_N_SUB)
    nx, ny, nz = size(f_blood)
    n_segs = size(A, 2)
    n_segs == length(c_iodine) || error("c_iodine length mismatch ($(length(c_iodine)) vs $n_segs)")
    voxel_cm = Float32(VOXEL_CM)
    inv_voxel_cm = Float32(1 / VOXEL_CM)
    inv_n_sub_total = Float32(1 / n_sub^3)
    inv_vox_vol = Float32(1 / VOXEL_CM^3)            # 1 / voxel volume (cm⁻³)
    sub_offsets = Float32[((2 * i - 1 - n_sub) / Float32(2 * n_sub)) * voxel_cm for i in 1:n_sub]

    nt = nthreads()
    buf_idx   = [Vector{Int32}() for _ in 1:nt]
    buf_blood = [Vector{Float32}() for _ in 1:nt]
    buf_iod   = [Vector{Float32}() for _ in 1:nt]
    for t in 1:nt
        h = max(1024, div(3 * n_segs, nt))
        sizehint!(buf_idx[t], h);  sizehint!(buf_blood[t], h);  sizehint!(buf_iod[t], h)
    end

    t0 = time()
    @threads :static for s in 1:n_segs
        tid = threadid()
        bi = buf_idx[tid];  bb = buf_blood[tid];  bk = buf_iod[tid]
        @inbounds begin
            ax = A[1,s]; ay = A[2,s]; az = A[3,s]
            bx = A[4,s]; by = A[5,s]; bz = A[6,s]
            r_cm = A[7,s]
            ci = c_iodine[s]
            abx = bx - ax;  aby = by - ay;  abz = bz - az
            ab_len2 = abx*abx + aby*aby + abz*abz

            # ── Sub-resolution vessels (< MIN_VOXELIZE_DIAM_CM): MC sampling at the
            #    sub-voxel spacing misses them, so deposit their EXACT lumen volume
            #    π r² · (centerline length within each voxel) / voxel_volume. This is
            #    resolution-independent: an 8 μm capillary contributes its true (tiny)
            #    volume fraction, and the diffuse MBV then EMERGES from the geometry. ──
            if 2f0 * r_cm < MIN_VOXELIZE_DIAM_CM
                seg_len = sqrt(ab_len2)
                if seg_len > 1f-12
                    area = Float32(pi) * r_cm * r_cm
                    n_st = max(1, ceil(Int, seg_len * inv_voxel_cm * 2f0))   # ≥2 samples/voxel
                    frac_per = area * (seg_len / Float32(n_st)) * inv_vox_vol
                    last_idx = Int32(0);  acc = 0f0
                    for m in 1:n_st
                        tt = (Float32(m) - 0.5f0) / Float32(n_st)
                        px = ax + tt*abx;  py = ay + tt*aby;  pz = az + tt*abz
                        ii = floor(Int, px * inv_voxel_cm) + 1
                        jj = floor(Int, py * inv_voxel_cm) + 1
                        kk = floor(Int, pz * inv_voxel_cm) + 1
                        (ii < 1 || ii > nx || jj < 1 || jj > ny || kk < 1 || kk > nz) && continue
                        is_myo(phantom[ii, jj, kk]) || continue
                        idx = Int32(((kk - 1) * ny + (jj - 1)) * nx + ii)
                        if idx == last_idx
                            acc += frac_per
                        else
                            last_idx != 0 && (push!(bi, last_idx); push!(bb, acc); push!(bk, acc * ci))
                            last_idx = idx;  acc = frac_per
                        end
                    end
                    last_idx != 0 && (push!(bi, last_idx); push!(bb, acc); push!(bk, acc * ci))
                end
                continue
            end

            # ── Resolvable vessels (≥ MIN_VOXELIZE_DIAM_CM): sub-voxel MC gives the
            #    cross-sectional volume fraction (cylinder ∩ voxel) / voxel directly. ──
            r2 = r_cm * r_cm
            lo_x = min(ax, bx) - r_cm;  hi_x = max(ax, bx) + r_cm
            lo_y = min(ay, by) - r_cm;  hi_y = max(ay, by) + r_cm
            lo_z = min(az, bz) - r_cm;  hi_z = max(az, bz) + r_cm

            i0 = max(1,  floor(Int, lo_x * inv_voxel_cm) + 1)
            j0 = max(1,  floor(Int, lo_y * inv_voxel_cm) + 1)
            k0 = max(1,  floor(Int, lo_z * inv_voxel_cm) + 1)
            i1 = min(nx, ceil(Int,  hi_x * inv_voxel_cm) + 1)
            j1 = min(ny, ceil(Int,  hi_y * inv_voxel_cm) + 1)
            k1 = min(nz, ceil(Int,  hi_z * inv_voxel_cm) + 1)
            (i0 > i1 || j0 > j1 || k0 > k1) && continue

            degenerate = ab_len2 <= 1f-24

            for kk in k0:k1
                cz = (Float32(kk) - 0.5f0) * voxel_cm
                for jj in j0:j1
                    cy = (Float32(jj) - 0.5f0) * voxel_cm
                    for ii in i0:i1
                        v = phantom[ii, jj, kk]
                        is_myo(v) || continue
                        cx = (Float32(ii) - 0.5f0) * voxel_cm
                        inside = 0
                        for k_sub in 1:n_sub
                            spz = cz + sub_offsets[k_sub]
                            apz = spz - az
                            for j_sub in 1:n_sub
                                spy = cy + sub_offsets[j_sub]
                                apy = spy - ay
                                for i_sub in 1:n_sub
                                    spx = cx + sub_offsets[i_sub]
                                    apx = spx - ax
                                    dist2 = if degenerate
                                        apx*apx + apy*apy + apz*apz
                                    else
                                        t = clamp((apx*abx + apy*aby + apz*abz) / ab_len2, 0f0, 1f0)
                                        dx = apx - t*abx;  dy = apy - t*aby;  dz = apz - t*abz
                                        dx*dx + dy*dy + dz*dz
                                    end
                                    dist2 <= r2 && (inside += 1)
                                end
                            end
                        end
                        if inside > 0
                            frac = Float32(inside) * inv_n_sub_total
                            lidx = Int32(((kk - 1) * ny + (jj - 1)) * nx + ii)
                            push!(bi, lidx);  push!(bb, frac);  push!(bk, frac * ci)
                        end
                    end
                end
            end
        end
    end
    t_raster = time() - t0

    t0 = time()
    n_writes = 0
    for t in 1:nt
        bi = buf_idx[t]; bb = buf_blood[t]; bk = buf_iod[t]
        @inbounds for k in eachindex(bi)
            f_blood[bi[k]]    += bb[k]
            f_iodine_w[bi[k]] += bk[k]
        end
        n_writes += length(bi)
        empty!(bi);  empty!(bb);  empty!(bk)
    end
    t_merge = time() - t0
    @printf("[raster] %d segs → %d voxel-incs (%.1fs raster + %.1fs merge, %.2f Mseg/s)\n",
            n_segs, n_writes, t_raster, t_merge, n_segs / (t_raster + t_merge) / 1e6)
    flush(stdout)
    n_writes
end

# Encode (blood %, iodine %) cross-product into UInt16 labels in `phantom`.
# Only myocardium voxels (15-18) are eligible to be relabeled.
function quantize_cross_product_and_apply!(phantom::Array{UInt16,3},
                                           f_blood::Array{Float32,3},
                                           f_iodine_w::Array{Float32,3},
                                           iodine_max_mg_per_mL::Float32)
    nx, ny, nz = size(phantom)

    # f_blood now holds the EMERGENT blood volume fraction per voxel — the real
    # lumen volume of the grown tree (conducting + capillary) intersected with the
    # voxel, divided by voxel volume. No MBV is imposed: the myocardial blood
    # volume is whatever the geometry produces.
    bin_counts_b = zeros(Int64, N_BLOOD_BINS + 1)    # bins 0..N_BLOOD_BINS
    bin_counts_i = zeros(Int64, N_IODINE_BINS + 1)
    per_thread_b = [zeros(Int64, N_BLOOD_BINS + 1) for _ in 1:nthreads()]
    per_thread_i = [zeros(Int64, N_IODINE_BINS + 1) for _ in 1:nthreads()]
    inv_iodine_max = iodine_max_mg_per_mL > 0f0 ? Float32(1 / iodine_max_mg_per_mL) : 0f0

    t0 = time()
    @threads :static for k in 1:nz
        lcb = per_thread_b[threadid()]
        lci = per_thread_i[threadid()]
        @inbounds for j in 1:ny, i in 1:nx
            v = phantom[i, j, k]
            is_myo(v) || continue
            f = f_blood[i, j, k]                             # emergent blood volume fraction
            f <= 0f0 && continue
            f_total = min(f, 1f0)                            # union upper bound
            bin_b = round(Int, f_total * N_BLOOD_BINS)
            lcb[bin_b + 1] += 1
            bin_b == 0 && continue
            c_voxel = f_iodine_w[i, j, k] / max(f, eps(Float32))
            c_clipped = min(max(c_voxel, 0f0), iodine_max_mg_per_mL)
            bin_i = inv_iodine_max > 0f0 ? round(Int, c_clipped * inv_iodine_max * N_IODINE_BINS) : 0
            bin_i = max(0, min(bin_i, N_IODINE_BINS))
            lci[bin_i + 1] += 1
            bin_b_c = min(bin_b, N_BLOOD_BINS)
            label = LABEL_BASE + UInt16((bin_b_c - 1) * (N_IODINE_BINS + 1) + bin_i)
            phantom[i, j, k] = label
        end
    end
    for c in per_thread_b;  bin_counts_b .+= c;  end
    for c in per_thread_i;  bin_counts_i .+= c;  end
    @printf("[quant] cross-product encoding (%.1fs)\n", time() - t0)
    flush(stdout)
    (bin_counts_b, bin_counts_i)
end

function write_phantom_raw_u16(path::String, phantom::Array{UInt16,3})
    t0 = time()
    nx, ny, nz = size(phantom)
    open(path, "w") do io
        write(io, phantom)
    end
    fsize = filesize(path)
    @printf("[write] %s: %d×%d×%d UInt16 = %.0f MB (%.1fs)\n",
            basename(path), nx, ny, nz, fsize/1024^2, time()-t0)
    flush(stdout)
    fsize
end

const LABEL_TO_MATERIAL = Dict{Int, String}(
    0  => "air",            1  => "softtissue",     2  => "softtissue",
    3  => "cortical_bone",  4  => "muscle",         5  => "cortical_bone",
    6  => "lung",           7  => "softtissue",     8  => "cortical_bone",
    9  => "cortical_bone",  10 => "softtissue",     11 => "softtissue",
    12 => "softtissue",     13 => "softtissue",     14 => "softtissue",
    15 => "muscle",         16 => "muscle",         17 => "muscle",
    18 => "muscle",         19 => "blood",          20 => "blood",
    21 => "blood",          22 => "blood",          23 => "softtissue",
    24 => "blood",          25 => "blood",          26 => "blood",
    27 => "blood",          28 => "blood",          29 => "softtissue",
    30 => "muscle",         31 => "adipose",        32 => "cortical_bone",
    70 => "air",
)

function write_manifest(out_dir::String, raw_basename::String,
                        hist_pre::Vector{Int64}, hist_post::Vector{Int64},
                        n_writes_total::Int, n_sub::Int,
                        tree_files::Vector{String}, seg_counts::Vector{Int},
                        bin_counts_b::Vector{Int64}, bin_counts_i::Vector{Int64},
                        iodine_max::Float32, peak_meta::Dict{String, Any},
                        peak_iodine_dir::String)
    path = joinpath(out_dir, "phantom_manifest.toml")
    present_pre  = sort([(i-1, hist_pre[i])  for i in 1:length(hist_pre)  if hist_pre[i]  > 0])
    present_post = sort([(i-1, hist_post[i]) for i in 1:length(hist_post) if hist_post[i] > 0])

    open(path, "w") do io
        println(io, "# phantom_manifest.toml — peak-contrast cross-product encoding")
        println(io, "# Generated by VascularTreeSim.jl/scripts/apply_contrast_at_peak.jl")
        println(io)
        println(io, "[phantom]")
        println(io, "raw_path = \"$raw_basename\"")
        println(io, "dims = [$(PHANTOM_DIMS[1]), $(PHANTOM_DIMS[2]), $(PHANTOM_DIMS[3])]")
        println(io, "dtype = \"UInt16\"")
        println(io, "byte_order = \"little-endian\"")
        println(io, "voxel_ordering = \"x-fastest\"")
        println(io, "voxel_size_cm = [$VOXEL_CM, $VOXEL_CM, $VOXEL_CM]")
        println(io, "xcat_origin_cm = [$(XCAT_ORIGIN_CM[1]), $(XCAT_ORIGIN_CM[2]), $(XCAT_ORIGIN_CM[3])]")
        println(io)
        println(io, "[embed]")
        println(io, "stage = \"peak_contrast\"")
        println(io, "method = \"sub_voxel_monte_carlo_with_iodine\"")
        println(io, "n_sub_per_dim = $n_sub")
        println(io, "n_sub_total = $(n_sub^3)")
        println(io, "voxel_increment_writes = $n_writes_total")
        println(io, "nrb_to_phantom_offset_cm = [$(NRB_TO_PHANTOM_OFFSET[1]), $(NRB_TO_PHANTOM_OFFSET[2]), $(NRB_TO_PHANTOM_OFFSET[3])]")
        println(io, "tree_csv_files = [", join(["\"$f\"" for f in tree_files], ", "), "]")
        println(io, "tree_segment_counts = [", join(string.(seg_counts), ", "), "]")
        println(io, "writable_base_labels = [15, 16, 17, 18]")
        println(io, "peak_iodine_dir = \"$peak_iodine_dir\"")
        println(io)
        println(io, "[mixture_materials]")
        println(io, "# Cross-product (blood %, iodine %) UInt16 label encoding.")
        println(io, "# label = label_base + (bin_b - 1) × (n_iodine_bins + 1) + bin_i")
        println(io, "#   bin_b ∈ 1..n_blood_bins  (f_blood ≈ bin_b / n_blood_bins)")
        println(io, "#   bin_i ∈ 0..n_iodine_bins (C_iodine ≈ bin_i / n_iodine_bins × iodine_max_mg_per_mL)")
        println(io, "# Loader builds an XA.Material for each (bin_b, bin_i) by mass-weighted")
        println(io, "# blending of blood + iodine + muscle. See load_phantom.jl.")
        println(io, "components_base = [\"blood\", \"muscle\"]")
        println(io, "contrast_agent  = \"iodine\"")
        println(io, "n_blood_bins  = $N_BLOOD_BINS")
        println(io, "n_iodine_bins = $N_IODINE_BINS")
        println(io, "label_base    = $(Int(LABEL_BASE))")
        @printf(io, "iodine_max_mg_per_mL = %.6f\n", iodine_max)
        println(io, "encoding = \"label = label_base + (bin_b - 1) * (n_iodine_bins + 1) + bin_i\"")
        @printf(io, "max_label = %d\n", Int(MAX_LABEL))
        if haskey(peak_meta, "peak")
            pk = peak_meta["peak"]
            @printf(io, "peak_time_s = %.6f\n", Float64(pk["time_s"]))
            @printf(io, "peak_total_iodine_mass_mg = %.6f\n", Float64(pk["total_iodine_mass_mg"]))
        end
        bin_counts_b_int = Int.(bin_counts_b)
        bin_counts_i_int = Int.(bin_counts_i)
        println(io, "bin_voxel_counts_blood  = [", join(string.(bin_counts_b_int), ", "), "]   # bins 0..$N_BLOOD_BINS")
        println(io, "bin_voxel_counts_iodine = [", join(string.(bin_counts_i_int), ", "), "]   # bins 0..$N_IODINE_BINS")
        println(io)
        println(io, "[present_labels_pre_embed]")
        println(io, "# Histogram of labels in the original (untouched) phantom.")
        for (l, n) in present_pre
            println(io, "\"$l\" = $n")
        end
        println(io)
        println(io, "[present_labels_post_embed_nonempty]")
        println(io, "# Histogram of labels in the embedded phantom (only entries with >0 voxels).")
        for (l, n) in present_post
            println(io, "\"$l\" = $n")
        end
        println(io)
        println(io, "[materials]")
        println(io, "# label → BasisSimulator material symbol.  Mixture labels (256..$(Int(MAX_LABEL)))")
        println(io, "# are NOT listed here — the loader constructs them from [mixture_materials].")
        for k in sort(collect(keys(LABEL_TO_MATERIAL)))
            println(io, "\"$k\" = \"$(LABEL_TO_MATERIAL[k])\"")
        end
    end
    @printf("[write] %s\n", path)
    path
end

# ─────────────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 4
        println("Usage: julia --project=. --threads=auto scripts/apply_contrast_at_peak.jl \\")
        println("              TREE_DIR  PHANTOM_RAW_IN  PEAK_IODINE_DIR  OUTPUT_DIR  [N_SUB]")
        exit(1)
    end
    tree_dir = ARGS[1]
    raw_in   = ARGS[2]
    peak_dir = ARGS[3]
    out_dir  = ARGS[4]
    n_sub    = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : DEFAULT_N_SUB
    isdir(tree_dir) || error("tree_dir not found: $tree_dir")
    isfile(raw_in)  || error("phantom raw not found: $raw_in")
    isdir(peak_dir) || error("peak_iodine_dir not found: $peak_dir")
    isdir(out_dir)  || mkpath(out_dir)

    # Read peak_metadata.toml from PEAK_IODINE_DIR
    meta_path = joinpath(peak_dir, "peak_metadata.toml")
    isfile(meta_path) || error("peak_metadata.toml not found in $peak_dir")
    peak_meta = TOML.parsefile(meta_path)
    iodine_max = Float32(peak_meta["peak"]["max_iodine_concentration_mg_per_mL"])
    if !(iodine_max > 0f0)
        @warn "iodine_max from peak_metadata is non-positive ($iodine_max); using contrast amplitude as fallback"
        iodine_max = Float32(peak_meta["contrast_bolus"]["amplitude_mg_per_mL"])
    end
    @printf("threads = %d\n", nthreads())
    println("tree_dir         = $tree_dir")
    println("phantom_raw_in   = $raw_in")
    println("peak_iodine_dir  = $peak_dir")
    println("output_dir       = $out_dir")
    @printf("n_sub_per_dim    = %d  (%d sub-samples/voxel)\n", n_sub, n_sub^3)
    @printf("iodine_max       = %.4f mg/mL  (bin %d ⇒ ~that value)\n",
            iodine_max, N_IODINE_BINS)
    flush(stdout)

    # Discover CSVs (must match peak_iodine.f32 by tree name)
    csvs = sort(filter(f -> endswith(f, "_segments.csv"), readdir(tree_dir; join=true)))
    isempty(csvs) && error("No *_segments.csv in $tree_dir")
    println("Trees:")
    for f in csvs; println("  $(basename(f))"); end;  flush(stdout)

    # 1) load phantom (UInt8 → UInt16)
    phantom = load_phantom_raw_u16(raw_in)
    t0 = time()
    hist_pre = label_histogram(phantom)
    @printf("[hist] pre-embed (%.1fs)\n", time()-t0); flush(stdout)

    # 2) allocate accumulators
    nx, ny, nz = size(phantom)
    @printf("[alloc] f_blood + f_iodine_w Float32[%d×%d×%d] = %.1f GB total\n",
            nx, ny, nz, 2 * nx*ny*nz*4 / 1024^3)
    flush(stdout)
    f_blood    = zeros(Float32, nx, ny, nz)
    f_iodine_w = zeros(Float32, nx, ny, nz)

    # 3) per-tree: load CSV + peak_iodine.f32, rasterize
    n_writes_total = 0
    seg_counts = Int[]
    for csv_path in csvs
        @printf("\n=== %s ===\n", basename(csv_path));  flush(stdout)
        tree_name = lowercase(replace(basename(csv_path), "_segments.csv" => ""))
        iodine_path = joinpath(peak_dir, "$(tree_name)_peak_iodine.f32")
        isfile(iodine_path) || error("missing peak_iodine for $tree_name at $iodine_path")
        A = parse_segments_csv(csv_path)
        nsegs = size(A, 2)
        push!(seg_counts, nsegs)
        c_iodine = load_peak_iodine_f32(iodine_path, nsegs)
        @printf("[iodine] %s_peak_iodine.f32: max=%.4f mg/mL\n",
                tree_name, Float64(maximum(c_iodine)))
        flush(stdout)
        n_writes_total += rasterize_capsules_vf_iodine!(f_blood, f_iodine_w, phantom, A,
                                                        c_iodine; n_sub=n_sub)
        A = nothing
        c_iodine = nothing
        GC.gc()
    end

    # 4) quantize cross-product → labels
    println()
    bin_counts_b, bin_counts_i = quantize_cross_product_and_apply!(phantom, f_blood, f_iodine_w, iodine_max)
    f_blood = nothing
    f_iodine_w = nothing
    GC.gc()

    # 5) outputs
    raw_out = joinpath(out_dir, OUT_RAW_BASENAME)
    println()
    write_phantom_raw_u16(raw_out, phantom)
    t0 = time()
    hist_post = label_histogram(phantom)
    @printf("[hist] post-embed (%.1fs)\n", time()-t0); flush(stdout)
    write_manifest(out_dir, OUT_RAW_BASENAME, hist_pre, hist_post,
                   n_writes_total, n_sub,
                   [basename(f) for f in csvs], seg_counts,
                   bin_counts_b, bin_counts_i, iodine_max, peak_meta, peak_dir)

    # Summary
    println()
    n_full = bin_counts_b[N_BLOOD_BINS + 1]
    n_partial = sum(bin_counts_b[2:N_BLOOD_BINS])
    n_iod_zero = bin_counts_i[1]
    n_iod_pos = sum(bin_counts_i[2:N_IODINE_BINS+1])
    @printf("done. blood: full(100%%)=%d  partial(1..99%%)=%d  total_grown=%d\n",
            n_full, n_partial, n_full + n_partial)
    @printf("      iodine: %d voxels with iodine>0   %d voxels with iodine=0\n",
            n_iod_pos, n_iod_zero)
    @printf("      label range used: 256 ≤ label ≤ %d  (UInt16 max label %d)\n",
            Int(MAX_LABEL), 65535)
end

main()
