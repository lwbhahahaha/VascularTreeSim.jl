"""
    GPU acceleration interface for VascularTreeSim.

Provides dispatch hooks that are overridden by the CUDA package extension.
When CUDA.jl is loaded (`using CUDA`), the extension automatically activates
GPU-accelerated distance kernels in the growth engine.

# Usage
```julia
using VascularTreeSim
using CUDA  # triggers extension loading

gpu_available()  # => true if CUDA is functional
```
"""

# ── GPU backend state ──

const _gpu_backend = Ref{Symbol}(:cpu)

"""
    gpu_available() -> Bool

Returns `true` if a GPU backend (CUDA) is loaded and functional.
"""
gpu_available() = _gpu_backend[] != :cpu

# ── GPU dispatch functions (overridden by extension) ──

# Function declarations (no method bodies — methods added by CUDA extension)
# Julia package extensions require this pattern to avoid method-overwriting errors.

"""Initialize GPU state: upload coverage points to device memory."""
function _gpu_init_distance_state end

"""Compute min distance from all points to all segments of a tree on GPU."""
function _gpu_full_distance_scan! end

"""Compute distance from all points to a single root vertex on GPU."""
function _gpu_seed_distance! end

"""Incrementally update min distances using only new segments on GPU."""
function _gpu_incremental_scan! end

"""Download current min_dist and owner arrays from GPU to CPU."""
function _gpu_download_distances end

"""Free GPU memory associated with the distance state."""
function _gpu_free! end
