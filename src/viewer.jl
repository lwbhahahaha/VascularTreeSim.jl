"""
    HTML viewer — data-driven branch names and colors.

Works with any number of trees with any names. Colors come from the config
color_map dict, not hardcoded values.
"""

_js_array_vw(vec) = "[" * join(vec, ",") * "]"
_js_string_vw(s::AbstractString) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => " ") * "\""

function _diameter_binned_trace_specs(tree::GrowthTree; bins_um::Vector{Float64}=[0.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, Inf], widths::Vector{Int}=[1, 2, 3, 4, 5, 6, 8])
    specs = NamedTuple[]
    for i in 1:length(widths)
        lo_um = bins_um[i]
        hi_um = bins_um[i + 1]
        xs = Float64[]
        ys = Float64[]
        zs = Float64[]
        seg_count = 0
        for s in eachindex(tree.segment_start)
            d_um = 1.0e4 * tree.segment_diameter_cm[s]
            if lo_um <= d_um < hi_um
                a = tree.vertices[tree.segment_start[s]]
                b = tree.vertices[tree.segment_end[s]]
                push!(xs, a[1]); push!(ys, a[2]); push!(zs, a[3])
                push!(xs, b[1]); push!(ys, b[2]); push!(zs, b[3])
                push!(xs, NaN); push!(ys, NaN); push!(zs, NaN)
                seg_count += 1
            end
        end
        if seg_count > 0
            push!(specs, (lo_um=lo_um, hi_um=hi_um, width=widths[i], xs=xs, ys=ys, zs=zs, count=seg_count))
        end
    end
    return specs
end

function _hover_arrays(tree::GrowthTree)
    rows = _segment_rows_with_parent(tree)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    texts = String[]
    for r in rows
        push!(xs, r.xmid_cm)
        push!(ys, r.ymid_cm)
        push!(zs, r.zmid_cm)
        push!(texts, "<b>segment $(r.segment_id)</b><br>parent: $(r.parent_segment_id)<br>type: $(r.label)<br>length: $(round(r.length_mm; digits=3)) mm<br>diameter: $(round(r.diameter_um; digits=2)) um")
    end
    return xs, ys, zs, texts
end

function growth_viewer_html(path::AbstractString, domain, trees::Dict{String, GrowthTree},
        stats::Dict{String, NamedTuple}, color_map::Dict{String, String};
        domain_stride::Int=1, surface_stride::Int=4)

    domain_points = _viewer_domain_points(domain; stride=domain_stride)
    outer_idx = collect(1:surface_stride:size(domain.outer_surface_points, 1))
    outer_points = domain.outer_surface_points[outer_idx, :]
    cavity_points = Matrix{Float64}(undef, 0, 3)
    for pts in domain.cavity_surface_points
        idx = collect(1:surface_stride:size(pts, 1))
        sample = pts[idx, :]
        cavity_points = isempty(cavity_points) ? copy(sample) : vcat(cavity_points, sample)
    end

    branch_names = sort(collect(keys(trees)))

    open(path, "w") do io
        println(io, "<!doctype html><html><head><meta charset='utf-8'><title>VascularTreeGrowth Viewer</title><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body style='margin:0;font-family:Arial,sans-serif'>")
        println(io, "<div style='padding:10px 14px'><h2 style='margin:0 0 6px 0'>VascularTreeGrowth: Competitive Growth Result</h2><div style='color:#555'>Vessel line width is binned by segment diameter. Hover on midpoint markers to inspect segment details.</div></div>")

        # Toggle buttons — data-driven
        print(io, "<div style='padding:0 14px 8px 14px;display:flex;gap:8px;flex-wrap:wrap'>")
        print(io, "<button id='toggle-domain'>Toggle Domain</button>")
        print(io, "<button id='toggle-outer'>Toggle Outer Surface</button>")
        print(io, "<button id='toggle-inner'>Toggle Inner Surface</button>")
        for name in branch_names
            print(io, "<button id='toggle-$(lowercase(name))'>Toggle $(name)</button>")
        end
        print(io, "<button id='show-all'>Show All</button>")
        println(io, "</div>")

        println(io, "<div id='plot' style='width:100vw;height:84vh'></div><script>const traces=[];")

        # Domain, outer, inner surface traces
        println(io, "traces.push({type:'scatter3d',mode:'markers',name:'Domain Interior',x:$(_js_array_vw(domain_points[:,1])),y:$(_js_array_vw(domain_points[:,2])),z:$(_js_array_vw(domain_points[:,3])),marker:{size:2.4,color:'#6b7280',opacity:0.28},hovertemplate:'Domain interior<extra></extra>'});")
        println(io, "traces.push({type:'scatter3d',mode:'markers',name:'Outer Surface',x:$(_js_array_vw(outer_points[:,1])),y:$(_js_array_vw(outer_points[:,2])),z:$(_js_array_vw(outer_points[:,3])),marker:{size:1.8,color:'#dc2626',opacity:0.30},hoverinfo:'skip'});")
        println(io, "traces.push({type:'scatter3d',mode:'markers',name:'Inner Surface',x:$(_js_array_vw(cavity_points[:,1])),y:$(_js_array_vw(cavity_points[:,2])),z:$(_js_array_vw(cavity_points[:,3])),marker:{size:1.8,color:'#2563eb',opacity:0.28},hoverinfo:'skip'});")

        # Tree traces — data-driven
        for name in branch_names
            tree = trees[name]
            color = get(color_map, name, "#888888")
            hx, hy, hz, htext = _hover_arrays(tree)
            st = get(stats, name, (terminals=0, p50=NaN, p95=NaN, max=NaN, added=0))
            terminals = _branch_terminals(tree)
            tx = [tree.vertices[i][1] for i in terminals]
            ty = [tree.vertices[i][2] for i in terminals]
            tz = [tree.vertices[i][3] for i in terminals]
            specs = _diameter_binned_trace_specs(tree)
            first_bin = true
            for spec in specs
                label = isfinite(spec.hi_um) ? "$(name) $(Int(round(spec.lo_um)))-$(Int(round(spec.hi_um))) um" : "$(name) >=$(Int(round(spec.lo_um))) um"
                println(io, "traces.push({type:'scatter3d',mode:'lines',name:'$label',x:$(_js_array_vw(spec.xs)),y:$(_js_array_vw(spec.ys)),z:$(_js_array_vw(spec.zs)),line:{color:'$(color)',width:$(spec.width)},opacity:0.95,hoverinfo:'skip',showlegend:$(first_bin ? "true" : "false")});")
                first_bin = false
            end
            println(io, "traces.push({type:'scatter3d',mode:'markers',name:'$(name) segments',x:$(_js_array_vw(hx)),y:$(_js_array_vw(hy)),z:$(_js_array_vw(hz)),customdata:[$(join((_js_string_vw(t) for t in htext), ","))],marker:{size:2.5,color:'$(color)',opacity:0.5},hovertemplate:'%{customdata}<extra></extra>',showlegend:false});")
            p95_val = isa(st, NamedTuple) && hasproperty(st, :p95) ? round(st.p95 * 10; digits=2) : "N/A"
            max_val = isa(st, NamedTuple) && hasproperty(st, :max) ? round(st.max * 10; digits=2) : "N/A"
            term_count = isa(st, NamedTuple) && hasproperty(st, :terminals) ? st.terminals : length(terminals)
            println(io, "traces.push({type:'scatter3d',mode:'markers',name:'$(name) terminals',x:$(_js_array_vw(tx)),y:$(_js_array_vw(ty)),z:$(_js_array_vw(tz)),marker:{size:1.8,color:'$(color)',opacity:0.65},hovertemplate:'$(name) terminals<br>count: $(term_count)<br>p95 distance: $(p95_val) mm<br>max distance: $(max_val) mm<extra></extra>',showlegend:false});")
        end

        # Layout + toggle logic — data-driven
        println(io, "Plotly.newPlot('plot', traces, {scene:{xaxis:{title:'X (cm)'},yaxis:{title:'Y (cm)'},zaxis:{title:'Z (cm)'},aspectmode:'data'},margin:{l:0,r:0,b:0,t:0}}, {displaylogo:false,responsive:true});")

        # Build index groups dynamically
        println(io, "const branchIdx = {};")
        for name in branch_names
            println(io, "branchIdx['$(lowercase(name))'] = [];")
        end
        println(io, "for(let i=3;i<traces.length;i++){ const n=traces[i].name || '';")
        for name in branch_names
            println(io, "  if(n.startsWith('$(name)')) branchIdx['$(lowercase(name))'].push(i);")
        end
        println(io, "}")

        println(io, "function toggleGroup(indices){ const plot=document.getElementById('plot'); const current=indices.map(i => plot.data[i].visible === false ? false : true); const nextVisible=current.every(v => v === false); Plotly.restyle('plot',{visible: nextVisible}, indices); }")
        println(io, "function showAll(){ Plotly.restyle('plot',{visible:true}, Array.from({length: traces.length}, (_, i) => i)); }")
        println(io, "document.getElementById('toggle-domain').onclick = () => toggleGroup([0]);")
        println(io, "document.getElementById('toggle-outer').onclick = () => toggleGroup([1]);")
        println(io, "document.getElementById('toggle-inner').onclick = () => toggleGroup([2]);")
        for name in branch_names
            println(io, "document.getElementById('toggle-$(lowercase(name))').onclick = () => toggleGroup(branchIdx['$(lowercase(name))']);")
        end
        println(io, "document.getElementById('show-all').onclick = () => showAll();")
        println(io, "</script></body></html>")
    end
    return path
end
