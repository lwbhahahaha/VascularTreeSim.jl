# Find NRB → phantom coordinate transform
using VascularTreeSim
using Statistics

CONFIG_PATH = joinpath(dirname(@__DIR__), "configs", "coronary.toml")
PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/shu_nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
const NX, NY, NZ = 1600, 1400, 500
const VOXEL_CM = 0.02

println("Loading phantom...")
phantom = Array{UInt8}(undef, NX, NY, NZ)
read!(PHANTOM_PATH, phantom)

# Phantom coronary (label 26) stats
function label_stats(phantom, label)
    cx, cy, cz, n = 0.0, 0.0, 0.0, 0
    xmin, ymin, zmin = Inf, Inf, Inf
    xmax, ymax, zmax = -Inf, -Inf, -Inf
    for k in 1:NZ, j in 1:NY, i in 1:NX
        phantom[i,j,k] == label || continue
        x = (i - 0.5) * VOXEL_CM; y = (j - 0.5) * VOXEL_CM; z = (k - 0.5) * VOXEL_CM
        cx += x; cy += y; cz += z; n += 1
        xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z)
        xmax = max(xmax, x); ymax = max(ymax, y); zmax = max(zmax, z)
    end
    return (centroid=[cx/n, cy/n, cz/n], lo=[xmin,ymin,zmin], hi=[xmax,ymax,zmax], n=n)
end

ph = label_stats(phantom, UInt8(26))
println("Phantom coronary: centroid=$(round.(ph.centroid;digits=3)) bbox=$(round.(ph.lo;digits=2))→$(round.(ph.hi;digits=2)) n=$(ph.n)")
ph_size = ph.hi .- ph.lo
println("  size=$(round.(ph_size;digits=2))")

# NRB coronary centerlines
config = load_organ_config(CONFIG_PATH)
surfaces = parse_xcat_nrb(config.nrb_path)
obj = xcat_object_dict(surfaces)

nrb_all = Float64[]
for spec in config.vessel_trees, sname in spec.surface_names
    haskey(obj, sname) || continue
    cline = xcat_centerline_from_surface(obj[sname])
    for pt in cline.centers
        p = pt .* config.coordinate_scale
        append!(nrb_all, [p[1], p[2], p[3]])
    end
end
nrb_pts = reshape(nrb_all, 3, :)'
nrb_c = vec(mean(nrb_pts; dims=1))
nrb_lo = vec(minimum(nrb_pts; dims=1))
nrb_hi = vec(maximum(nrb_pts; dims=1))
nrb_size = nrb_hi .- nrb_lo
println("NRB coronary: centroid=$(round.(nrb_c;digits=3)) bbox=$(round.(nrb_lo;digits=2))→$(round.(nrb_hi;digits=2)) n=$(size(nrb_pts,1))")
println("  size=$(round.(nrb_size;digits=2))")

# Exhaustive search
function find_best_transform(ph_c, ph_size, nrb_c, nrb_size)
    perms = [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]
    sgns = [(s1,s2,s3) for s1 in [1,-1] for s2 in [1,-1] for s3 in [1,-1]]
    best_err = Inf
    best = nothing
    for perm in perms, sgn in sgns
        mapped_c = [sgn[d] * nrb_c[perm[d]] for d in 1:3]
        offset = ph_c .- mapped_c
        mapped_sz = [nrb_size[perm[d]] for d in 1:3]
        err = sum((ph_size .- mapped_sz).^2)
        if err < best_err
            best_err = err
            best = (perm=perm, sign=sgn, offset=offset, err=err)
        end
    end
    return best
end

r = find_best_transform(ph.centroid, ph_size, nrb_c, nrb_size)
println("\n=== Best transform ===")
println("perm=$(r.perm) sign=$(r.sign) offset=$(round.(r.offset;digits=4)) err=$(round(r.err;digits=4))")

axes_name = ["nrb_x","nrb_y","nrb_z"]
for d in 1:3
    s = r.sign[d] > 0 ? "" : "-"
    println("  phantom_$(["x","y","z"][d]) = $(s)$(axes_name[r.perm[d]]) + $(round(r.offset[d];digits=4))")
end

# Verify: transform NRB bbox corners → phantom
println("\nBbox verification:")
mapped_lo = [r.sign[d] * (r.sign[d]>0 ? nrb_lo : nrb_hi)[r.perm[d]] + r.offset[d] for d in 1:3]
mapped_hi = [r.sign[d] * (r.sign[d]>0 ? nrb_hi : nrb_lo)[r.perm[d]] + r.offset[d] for d in 1:3]
for d in 1:3
    a = ["x","y","z"][d]
    println("  $(a): phantom=[$(round(ph.lo[d];digits=2)),$(round(ph.hi[d];digits=2))]  mapped_nrb=[$(round(mapped_lo[d];digits=2)),$(round(mapped_hi[d];digits=2))]")
end
