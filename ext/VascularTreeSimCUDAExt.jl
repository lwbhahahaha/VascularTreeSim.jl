"""
    VascularTreeSimCUDAExt — CUDA acceleration for VascularTreeSim.

Provides GPU-accelerated distance kernels for the competitive growth engine.
Each GPU thread handles one coverage point and brute-forces over all segments,
leveraging massive parallelism instead of spatial indexing.

Activated automatically when `using CUDA` is called alongside VascularTreeSim.
"""
module VascularTreeSimCUDAExt

using VascularTreeSim
using CUDA
using StaticArrays

# ── GPU state container ──

"""
Holds device arrays that persist across growth rounds to minimize CPU↔GPU transfers.
Point arrays are uploaded once. Min distances and ownership live on GPU.
"""
mutable struct GPUDistanceState
    # Coverage point coordinates (uploaded once, read-only)
    d_px::CuVector{Float64}
    d_py::CuVector{Float64}
    d_pz::CuVector{Float64}
    # Persistent distance/ownership state
    d_min_dist::CuVector{Float64}
    d_owner::CuVector{Int32}
    n_points::Int
end

# ── CUDA Kernels ──

"""
Brute-force minimum segment distance kernel.
Each thread = one point, iterates over ALL n_segs segments.
"""
function _kernel_min_seg_dist!(min_dist, owner,
                                px, py, pz,
                                ax, ay, az, bx, by, bz,
                                n_segs::Int32, tree_idx::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing

    best_d2 = Inf

    @inbounds for s in Int32(1):n_segs
        _abx = bx[s] - ax[s]
        _aby = by[s] - ay[s]
        _abz = bz[s] - az[s]
        _apx = px[i] - ax[s]
        _apy = py[i] - ay[s]
        _apz = pz[i] - az[s]

        denom = _abx * _abx + _aby * _aby + _abz * _abz

        if denom <= 1e-24
            d2 = _apx * _apx + _apy * _apy + _apz * _apz
        else
            t = (_apx * _abx + _apy * _aby + _apz * _abz) / denom
            t = max(0.0, min(1.0, t))
            dx = _apx - t * _abx
            dy = _apy - t * _aby
            dz = _apz - t * _abz
            d2 = dx * dx + dy * dy + dz * dz
        end

        d2 < best_d2 && (best_d2 = d2)
    end

    d = sqrt(best_d2)
    @inbounds if d < min_dist[i]
        min_dist[i] = d
        owner[i] = tree_idx
    end
    return nothing
end

"""
Incremental kernel — only processes segments in range [seg_start, seg_end].
"""
function _kernel_min_seg_dist_range!(min_dist, owner,
                                      px, py, pz,
                                      ax, ay, az, bx, by, bz,
                                      seg_start::Int32, seg_end::Int32, tree_idx::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing

    best_d = @inbounds min_dist[i]
    best_owner = @inbounds owner[i]

    @inbounds for s in seg_start:seg_end
        _abx = bx[s] - ax[s]
        _aby = by[s] - ay[s]
        _abz = bz[s] - az[s]
        _apx = px[i] - ax[s]
        _apy = py[i] - ay[s]
        _apz = pz[i] - az[s]

        denom = _abx * _abx + _aby * _aby + _abz * _abz

        if denom <= 1e-24
            d2 = _apx * _apx + _apy * _apy + _apz * _apz
        else
            t = (_apx * _abx + _apy * _aby + _apz * _abz) / denom
            t = max(0.0, min(1.0, t))
            dx = _apx - t * _abx
            dy = _apy - t * _aby
            dz = _apz - t * _abz
            d2 = dx * dx + dy * dy + dz * dz
        end

        d = sqrt(d2)
        if d < best_d
            best_d = d
            best_owner = tree_idx
        end
    end

    @inbounds min_dist[i] = best_d
    @inbounds owner[i] = best_owner
    return nothing
end

"""
Seed tree kernel — distance from all points to a single root vertex.
"""
function _kernel_seed_dist!(min_dist, owner,
                             px, py, pz,
                             rx::Float64, ry::Float64, rz::Float64,
                             tree_idx::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing

    @inbounds begin
        dx = px[i] - rx
        dy = py[i] - ry
        dz = pz[i] - rz
        d = sqrt(dx * dx + dy * dy + dz * dz)
        if d < min_dist[i]
            min_dist[i] = d
            owner[i] = tree_idx
        end
    end
    return nothing
end

# ── Launch helpers ──

const CUDA_BLOCK_SIZE = 256

function _launch_config(n::Int)
    threads = min(CUDA_BLOCK_SIZE, n)
    blocks = cld(n, threads)
    return threads, blocks
end

# ── Interface implementations ──

function VascularTreeSim._gpu_init_distance_state(points_cm::Matrix{Float64})
    n = size(points_cm, 1)
    d_px = CuArray(points_cm[:, 1])
    d_py = CuArray(points_cm[:, 2])
    d_pz = CuArray(points_cm[:, 3])
    d_min_dist = CUDA.fill(Inf, n)
    d_owner = CUDA.zeros(Int32, n)
    println("[GPU] initialized: $(n) points on $(CUDA.name(CUDA.device()))")
    flush(stdout)
    return GPUDistanceState(d_px, d_py, d_pz, d_min_dist, d_owner, n)
end

function VascularTreeSim._gpu_full_distance_scan!(state::GPUDistanceState,
                                                    seg_idx::VascularTreeSim.SegmentSpatialIndex,
                                                    tree_idx::Int)
    nseg = length(seg_idx.ax)
    nseg == 0 && return nothing

    d_ax = CuArray(seg_idx.ax)
    d_ay = CuArray(seg_idx.ay)
    d_az = CuArray(seg_idx.az)
    d_bx = CuArray(seg_idx.bx)
    d_by = CuArray(seg_idx.by)
    d_bz = CuArray(seg_idx.bz)

    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_min_seg_dist!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        d_ax, d_ay, d_az, d_bx, d_by, d_bz,
        Int32(nseg), Int32(tree_idx))
    CUDA.synchronize()

    CUDA.unsafe_free!(d_ax); CUDA.unsafe_free!(d_ay); CUDA.unsafe_free!(d_az)
    CUDA.unsafe_free!(d_bx); CUDA.unsafe_free!(d_by); CUDA.unsafe_free!(d_bz)
    return nothing
end

function VascularTreeSim._gpu_seed_distance!(state::GPUDistanceState,
                                               root_vertex::SVector{3,Float64},
                                               tree_idx::Int)
    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_seed_dist!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        root_vertex[1], root_vertex[2], root_vertex[3],
        Int32(tree_idx))
    CUDA.synchronize()
    return nothing
end

function VascularTreeSim._gpu_incremental_scan!(state::GPUDistanceState,
                                                  seg_idx::VascularTreeSim.SegmentSpatialIndex,
                                                  tree_idx::Int,
                                                  seg_start::Int, seg_end::Int)
    seg_end < seg_start && return nothing
    nseg_total = length(seg_idx.ax)
    nseg_total == 0 && return nothing

    # Upload only new segment endpoints for the incremental range
    # But kernel needs absolute indices, so upload full arrays
    d_ax = CuArray(seg_idx.ax)
    d_ay = CuArray(seg_idx.ay)
    d_az = CuArray(seg_idx.az)
    d_bx = CuArray(seg_idx.bx)
    d_by = CuArray(seg_idx.by)
    d_bz = CuArray(seg_idx.bz)

    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_min_seg_dist_range!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        d_ax, d_ay, d_az, d_bx, d_by, d_bz,
        Int32(seg_start), Int32(seg_end), Int32(tree_idx))
    CUDA.synchronize()

    CUDA.unsafe_free!(d_ax); CUDA.unsafe_free!(d_ay); CUDA.unsafe_free!(d_az)
    CUDA.unsafe_free!(d_bx); CUDA.unsafe_free!(d_by); CUDA.unsafe_free!(d_bz)
    return nothing
end

function VascularTreeSim._gpu_download_distances(state::GPUDistanceState)
    min_dist = Array(state.d_min_dist)
    owner = Int.(Array(state.d_owner))
    return min_dist, owner
end

function VascularTreeSim._gpu_free!(state::GPUDistanceState)
    CUDA.unsafe_free!(state.d_px); CUDA.unsafe_free!(state.d_py); CUDA.unsafe_free!(state.d_pz)
    CUDA.unsafe_free!(state.d_min_dist); CUDA.unsafe_free!(state.d_owner)
    return nothing
end

# ── Extension initialization ──

function __init__()
    if CUDA.functional()
        VascularTreeSim._gpu_backend[] = :cuda
        dev = CUDA.device()
        mem_gb = round(CUDA.totalmem(dev) / 1024^3; digits=1)
        println("[VascularTreeSim] CUDA extension loaded: $(CUDA.name(dev)) ($(mem_gb) GB)")
        flush(stdout)
    else
        @warn "CUDA.jl loaded but no functional GPU detected. GPU acceleration disabled."
    end
end

end # module VascularTreeSimCUDAExt
