"""
    CSV export with parent_segment_id column for deterministic topology reconstruction.
"""

function _segment_rows_with_parent(tree::GrowthTree)
    rows = NamedTuple[]

    # Build a map: for each segment s, find which segment leads into segment_start[s]
    # parent_segment_id = the segment whose end vertex == this segment's start vertex
    seg_end_to_seg = Dict{Int, Int}()
    for s in eachindex(tree.segment_start)
        seg_end_to_seg[tree.segment_end[s]] = s
    end

    for s in eachindex(tree.segment_start)
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        length_cm = norm(b - a)
        diameter_cm = tree.segment_diameter_cm[s]

        # Find parent segment: the segment whose end_vertex == this segment's start_vertex
        start_v = tree.segment_start[s]
        parent_seg_id = get(seg_end_to_seg, start_v, 0)

        push!(rows, (
            segment_id=s,
            parent_segment_id=parent_seg_id,
            x1_cm=a[1], y1_cm=a[2], z1_cm=a[3],
            x2_cm=b[1], y2_cm=b[2], z2_cm=b[3],
            xmid_cm=(a[1] + b[1]) / 2,
            ymid_cm=(a[2] + b[2]) / 2,
            zmid_cm=(a[3] + b[3]) / 2,
            length_mm=10.0 * length_cm,
            diameter_um=1.0e4 * diameter_cm,
            label=tree.segment_label[s],
        ))
    end
    return rows
end

function write_growth_csv(path::AbstractString, branch::String, tree::GrowthTree)
    rows = _segment_rows_with_parent(tree)
    open(path, "w") do io
        println(io, "branch,segment_id,parent_segment_id,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,xmid_cm,ymid_cm,zmid_cm,length_mm,diameter_um,label")
        for r in rows
            println(io, join((branch, r.segment_id, r.parent_segment_id,
                r.x1_cm, r.y1_cm, r.z1_cm, r.x2_cm, r.y2_cm, r.z2_cm,
                r.xmid_cm, r.ymid_cm, r.zmid_cm,
                r.length_mm, r.diameter_um, r.label), ","))
        end
    end
    return path
end
