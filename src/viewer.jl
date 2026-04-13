"""
    HTML viewer generation.

Two modes:
  1. domain_check_html — domain-only viewer for user confirmation (before growth)
  2. growth_viewer_html — full viewer with trees (delegates to build_viewer.py)
"""

function _sample_points_for_viewer(pts::Matrix{Float64}; max_points::Int=500_000)
    n = size(pts, 1)
    if n <= max_points
        return pts
    end
    # Random subsample for display
    rng = Random.MersenneTwister(42)
    indices = sort(Random.randperm(rng, n)[1:max_points])
    return pts[indices, :]
end

"""
    domain_check_html(path, domain; max_display=500_000)

Generate a domain-only HTML viewer for user confirmation before growth.
Shows myocardium, epicardial surface, and endocardial surfaces.
"""
function domain_check_html(path::AbstractString, domain::VoxelShellDomain;
        max_display::Int=500_000)

    # Collect domain points
    all_pts = voxel_mask_points(domain)
    display_pts = _sample_points_for_viewer(all_pts; max_points=max_display)

    # Build JSON traces manually (simple enough for domain-only)
    traces = String[]

    # Domain (myocardium) trace
    n = size(display_pts, 1)
    dx = join([@sprintf("%.3f", display_pts[i,1]) for i in 1:n], ",")
    dy = join([@sprintf("%.3f", display_pts[i,2]) for i in 1:n], ",")
    dz = join([@sprintf("%.3f", display_pts[i,3]) for i in 1:n], ",")
    push!(traces, """{
        "type":"scatter3d","mode":"markers","name":"Myocardium ($(size(all_pts,1)) pts, showing $(n))",
        "x":[$dx],"y":[$dy],"z":[$dz],
        "marker":{"size":1.5,"color":"#6b7280","opacity":0.15},"hoverinfo":"skip"
    }""")

    # Epicardial surface
    epi = domain.outer_surface_points
    ne = size(epi, 1)
    ex = join([@sprintf("%.3f", epi[i,1]) for i in 1:ne], ",")
    ey = join([@sprintf("%.3f", epi[i,2]) for i in 1:ne], ",")
    ez = join([@sprintf("%.3f", epi[i,3]) for i in 1:ne], ",")
    push!(traces, """{
        "type":"scatter3d","mode":"markers","name":"Epicardium ($(ne) pts)",
        "x":[$ex],"y":[$ey],"z":[$ez],
        "marker":{"size":1.5,"color":"#ef4444","opacity":0.25},"hoverinfo":"skip","visible":false
    }""")

    # Endocardial surfaces
    colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ec4899","#6366f1","#14b8a6"]
    for (idx, cavity_pts) in enumerate(domain.cavity_surface_points)
        nc = size(cavity_pts, 1)
        color = colors[mod1(idx, length(colors))]
        cx = join([@sprintf("%.3f", cavity_pts[i,1]) for i in 1:nc], ",")
        cy = join([@sprintf("%.3f", cavity_pts[i,2]) for i in 1:nc], ",")
        cz = join([@sprintf("%.3f", cavity_pts[i,3]) for i in 1:nc], ",")
        push!(traces, """{
            "type":"scatter3d","mode":"markers","name":"Cavity $(idx) ($(nc) pts)",
            "x":[$cx],"y":[$cy],"z":[$cz],
            "marker":{"size":1.5,"color":"$(color)","opacity":0.20},"hoverinfo":"skip","visible":false
        }""")
    end

    traces_json = join(traces, ",\n")
    n_traces = length(traces)

    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Domain Check</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body style="margin:0;font-family:Arial,sans-serif">
<div style="padding:10px 14px">
  <h2 style="margin:0 0 6px">Domain Confirmation</h2>
  <div style="color:#555">Check: no holes in myocardium surface, cavities properly excluded</div>
</div>
<div style="padding:0 14px 8px;display:flex;gap:8px">
  <button onclick="toggle(0)">Toggle Myocardium</button>
  <button onclick="toggle(1)">Toggle Epicardium</button>
  <button onclick="for(let i=2;i<$(n_traces);i++) toggle(i)">Toggle Cavities</button>
  <button onclick="Plotly.restyle('plot',{visible:true},Array.from({length:$(n_traces)},(_,i)=>i))">Show All</button>
</div>
<div id="plot" style="width:100vw;height:85vh"></div>
<script>
const traces = [$traces_json];
Plotly.newPlot('plot', traces, {
  scene:{xaxis:{title:'X (cm)'},yaxis:{title:'Y (cm)'},zaxis:{title:'Z (cm)'},aspectmode:'data'},
  margin:{l:0,r:0,b:0,t:0}
},{displaylogo:false,responsive:true});
function toggle(i){
  const p=document.getElementById('plot');
  Plotly.restyle('plot',{visible:p.data[i].visible!==false?false:true},[i]);
}
</script></body></html>"""

    open(path, "w") do io
        write(io, html)
    end
    @info "Domain check viewer: $path ($(round(filesize(path)/1e6; digits=1)) MB)"
    return path
end

function growth_viewer_html(path::AbstractString, domain, trees::Dict{String, GrowthTree},
        stats::Dict{String, NamedTuple}, color_map::Dict{String, String};
        domain_stride::Int=1, surface_stride::Int=4)

    output_dir = dirname(path)

    # Write tree CSVs (the Python script reads them)
    for (branch, tree) in trees
        csv_path = joinpath(output_dir, lowercase(branch) * "_segments.csv")
        write_growth_csv(csv_path, branch, tree)
    end

    # Run the Python builder script
    script_path = joinpath(output_dir, "build_viewer.py")
    if !isfile(script_path)
        @warn "build_viewer.py not found at $script_path — cannot generate viewer HTML"
        return path
    end

    cmd = `python3 $script_path`
    try
        run(cmd)
        @info "Viewer HTML generated: $path"
    catch e
        @warn "Failed to generate viewer HTML" exception=e
    end

    return path
end
