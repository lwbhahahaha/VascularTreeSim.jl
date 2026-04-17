#!/usr/bin/env python3
"""
Build XCAT coronary viewer with TRUE PROPORTIONAL diameters.

Vessels rendered as mesh3d cylinders at their real physical radius (cm).
A 4000μm vessel is ~500× thicker than an 8μm vessel — exactly as in reality.

Anatomy layers: Domain, Chambers, Great Vessels, Pericardium, XCAT Coronaries
Grown trees: LAD / LCX / RCA as mesh3d cylinders + diameter slider filtering
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path("/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation")
OUT = ROOT / "VascularTreeSim.jl" / "output"
TITLE = "XCAT Coronary Tree: True-Scale Vessel Diameters"
SUBTITLE = "mesh3d cylinders at real physical radii. Toggle layers, filter by diameter."

# Anatomy CSVs
DOMAIN_CSV = OUT / "domain_points.csv"
CHAMBERS_CSV = OUT / "chambers_points.csv"
GREAT_VESSELS_CSV = OUT / "great_vessels_points.csv"
PERICARDIUM_CSV = OUT / "pericardium_points.csv"
XCAT_CORONARY_CSV = OUT / "coronary_arteries_points.csv"

TREE_SPECS = {
    "LAD": {"csv": OUT / "lad_segments.csv", "color": "#1f77ff", "color_rgb": (31, 119, 255)},
    "LCX": {"csv": OUT / "lcx_segments.csv", "color": "#e3342f", "color_rgb": (227, 52, 47)},
    "RCA": {"csv": OUT / "rca_segments.csv", "color": "#22aa44", "color_rgb": (34, 170, 68)},
}

MAX_DOMAIN_DISPLAY_POINTS = 300_000
MAX_DISPLAY_SEGMENTS = 60_000  # auto-filter by diameter if total segments exceed this
N_SIDES = 6  # hexagonal cross-section for cylinders


# ── Helpers ──

def load_points_csv(path: Path, max_points: int = 0):
    xs, ys, zs = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        if max_points <= 0:
            for row in reader:
                xs.append(round(float(row["x_cm"]), 3))
                ys.append(round(float(row["y_cm"]), 3))
                zs.append(round(float(row["z_cm"]), 3))
        else:
            rng = random.Random(42)
            reservoir = []
            for i, row in enumerate(reader):
                pt = (round(float(row["x_cm"]), 3),
                      round(float(row["y_cm"]), 3),
                      round(float(row["z_cm"]), 3))
                if i < max_points:
                    reservoir.append(pt)
                else:
                    j = rng.randint(0, i)
                    if j < max_points:
                        reservoir[j] = pt
            for pt in reservoir:
                xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
    return xs, ys, zs


def load_tree_lines(path: Path, min_diameter: float = 0.0):
    """Stream CSV and materialize only rows with diameter >= min_diameter.

    For very large files (10^8+ segments), pre-filtering by diameter avoids
    allocating a dict-per-segment for rows we would discard anyway.
    """
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = float(row["diameter_um"])
            if d < min_diameter:
                continue
            rows.append({
                "x1": float(row["x1_cm"]), "y1": float(row["y1_cm"]), "z1": float(row["z1_cm"]),
                "x2": float(row["x2_cm"]), "y2": float(row["y2_cm"]), "z2": float(row["z2_cm"]),
                "diameter_um": d,
                "length_mm": float(row["length_mm"]),
                "segment_id": int(row["segment_id"]),
                "parent_segment_id": int(row["parent_segment_id"]),
                "label": row.get("label", ""),
            })
    return rows


def scan_tree_diameters(path: Path):
    """First pass: stream CSV and collect only diameter column as a float32 numpy array.

    Avoids allocating per-row dicts for trees with 10^8+ segments. Returns a
    numpy array of all diameters.
    """
    # Chunk-based append to avoid repeated realloc; 1M floats per chunk = 4 MB.
    chunk_size = 1_000_000
    chunks = []
    buf = np.empty(chunk_size, dtype=np.float32)
    n = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if n == chunk_size:
                chunks.append(buf)
                buf = np.empty(chunk_size, dtype=np.float32)
                n = 0
            buf[n] = float(row["diameter_um"])
            n += 1
    if n > 0:
        chunks.append(buf[:n])
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def find_tree_root_from_csv(path: Path):
    """Find root vertex (parent_segment_id=0 or largest-diameter fallback) in a single pass."""
    root_xyz = None
    fallback_d = -1.0
    fallback_xyz = None
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["parent_segment_id"]) == 0:
                if root_xyz is None:
                    root_xyz = (float(row["x1_cm"]), float(row["y1_cm"]), float(row["z1_cm"]))
                    # Keep scanning briefly only if no explicit root; but since we found one, break.
                    return root_xyz
            d = float(row["diameter_um"])
            if d > fallback_d:
                fallback_d = d
                fallback_xyz = (float(row["x1_cm"]), float(row["y1_cm"]), float(row["z1_cm"]))
    return root_xyz if root_xyz is not None else fallback_xyz


def find_tree_root(rows):
    roots = [r for r in rows if r["parent_segment_id"] == 0]
    if not roots:
        roots = sorted(rows, key=lambda r: r["diameter_um"], reverse=True)
    root = roots[0]
    return (root["x1"], root["y1"], root["z1"])


def dist3d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def split_xcat_coronaries(coronary_csv, tree_roots):
    xs, ys, zs = load_points_csv(coronary_csv)
    result = {name: ([], [], []) for name in tree_roots}
    names = list(tree_roots.keys())
    roots = [tree_roots[n] for n in names]
    for x, y, z in zip(xs, ys, zs):
        best_name = names[0]; best_d = float("inf")
        for name, root in zip(names, roots):
            d = dist3d((x, y, z), root)
            if d < best_d:
                best_d = d; best_name = name
        result[best_name][0].append(x)
        result[best_name][1].append(y)
        result[best_name][2].append(z)
    return result


def log_bins(dmin, dmax, count=12):
    lo = math.log10(max(dmin, 1e-6))
    hi = math.log10(max(dmax, dmin * 1.001))
    edges = [10 ** (lo + (hi - lo) * i / count) for i in range(count + 1)]
    edges[0] = dmin; edges[-1] = dmax
    return edges


# ── Cylinder mesh generation ──

def _perp_vectors(axis):
    """Two unit vectors perpendicular to axis."""
    if abs(axis[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


# Pre-compute ring offsets for N_SIDES
_angles = np.linspace(0, 2 * np.pi, N_SIDES, endpoint=False)
_cos = np.cos(_angles)
_sin = np.sin(_angles)


def build_cylinder_mesh(segments, color_rgb, n_sides=N_SIDES):
    """Build a single mesh3d trace from all segments as hexagonal prism cylinders.

    Each segment (p1→p2, radius_cm) becomes a prism with n_sides faces.
    All prisms are combined into one mesh3d for efficiency.

    Returns a Plotly mesh3d trace dict.
    """
    n = len(segments)
    if n == 0:
        return None

    n_verts_per = 2 * n_sides
    n_faces_per = 2 * n_sides
    total_verts = n * n_verts_per
    total_faces = n * n_faces_per

    vx = np.empty(total_verts)
    vy = np.empty(total_verts)
    vz = np.empty(total_verts)
    fi = np.empty(total_faces, dtype=np.int32)
    fj = np.empty(total_faces, dtype=np.int32)
    fk = np.empty(total_faces, dtype=np.int32)

    cos_a = _cos[:n_sides]
    sin_a = _sin[:n_sides]

    for idx, seg in enumerate(segments):
        p1 = np.array([seg["x1"], seg["y1"], seg["z1"]])
        p2 = np.array([seg["x2"], seg["y2"], seg["z2"]])
        r = seg["diameter_um"] * 1e-4 / 2.0  # μm → cm radius

        axis = p2 - p1
        length = np.linalg.norm(axis)
        if length < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
            length = 1e-12
        axis_n = axis / length
        u, v = _perp_vectors(axis_n)

        # Ring offsets: r * (cos*u + sin*v)
        offsets = r * (cos_a[:, None] * u[None, :] + sin_a[:, None] * v[None, :])

        v_start = idx * n_verts_per
        # Bottom ring
        bottom = p1[None, :] + offsets
        vx[v_start:v_start+n_sides] = bottom[:, 0]
        vy[v_start:v_start+n_sides] = bottom[:, 1]
        vz[v_start:v_start+n_sides] = bottom[:, 2]
        # Top ring
        top = p2[None, :] + offsets
        vx[v_start+n_sides:v_start+n_verts_per] = top[:, 0]
        vy[v_start+n_sides:v_start+n_verts_per] = top[:, 1]
        vz[v_start+n_sides:v_start+n_verts_per] = top[:, 2]

        # Faces: quad strip as triangle pairs
        f_start = idx * n_faces_per
        for s in range(n_sides):
            s_next = (s + 1) % n_sides
            bi = v_start + s
            bj = v_start + s_next
            ti = v_start + n_sides + s
            tj = v_start + n_sides + s_next
            fi[f_start + 2*s]     = bi;  fj[f_start + 2*s]     = bj;  fk[f_start + 2*s]     = tj
            fi[f_start + 2*s + 1] = bi;  fj[f_start + 2*s + 1] = tj;  fk[f_start + 2*s + 1] = ti

    # Round to reduce JSON size
    vx = np.round(vx, 5)
    vy = np.round(vy, 5)
    vz = np.round(vz, 5)

    r, g, b = color_rgb
    return {
        "type": "mesh3d",
        "x": vx.tolist(), "y": vy.tolist(), "z": vz.tolist(),
        "i": fi.tolist(), "j": fj.tolist(), "k": fk.tolist(),
        "color": f"rgb({r},{g},{b})",
        "opacity": 0.9,
        "flatshading": True,
        "hoverinfo": "skip",
        "lighting": {"ambient": 0.6, "diffuse": 0.7, "specular": 0.3, "roughness": 0.5},
        "lightposition": {"x": 1000, "y": 1000, "z": 1000},
    }


def make_hover_trace(rows, color, branch):
    xs, ys, zs, texts = [], [], [], []
    for r in rows:
        xs.append(round((r["x1"] + r["x2"]) / 2, 4))
        ys.append(round((r["y1"] + r["y2"]) / 2, 4))
        zs.append(round((r["z1"] + r["z2"]) / 2, 4))
        texts.append(
            f"{branch} #{r['segment_id']}  "
            f"d={r['diameter_um']:.1f}\u00b5m  "
            f"L={r['length_mm']:.3f}mm  "
            f"{r['label']}"
        )
    return {
        "type": "scatter3d", "mode": "markers",
        "name": f"{branch} hover",
        "x": xs, "y": ys, "z": zs,
        "text": texts, "hoverinfo": "text",
        "marker": {"size": 2, "color": color, "opacity": 0.0},
        "showlegend": False,
    }


# ── Main ──

def main():
    traces = []
    groups = {}

    # ── Anatomy layers (same as before) ──
    if DOMAIN_CSV.exists():
        dx, dy, dz = load_points_csv(DOMAIN_CSV, max_points=MAX_DOMAIN_DISPLAY_POINTS)
        print(f"  Domain: {len(dx):,} display points")
        groups["domain"] = [len(traces)]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": "Myocardium",
            "x": dx, "y": dy, "z": dz,
            "marker": {"size": 1.5, "color": "#6b7280", "opacity": 0.15},
            "hoverinfo": "skip",
        })

    if CHAMBERS_CSV.exists():
        cx, cy, cz = load_points_csv(CHAMBERS_CSV)
        print(f"  Chambers: {len(cx):,} points")
        groups["chambers"] = [len(traces)]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": "Chambers",
            "x": cx, "y": cy, "z": cz,
            "marker": {"size": 1.5, "color": "#dc2626", "opacity": 0.10},
            "hoverinfo": "skip", "visible": False,
        })

    if GREAT_VESSELS_CSV.exists():
        gx, gy, gz = load_points_csv(GREAT_VESSELS_CSV)
        print(f"  Great Vessels: {len(gx):,} points")
        groups["great_vessels"] = [len(traces)]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": "Great Vessels",
            "x": gx, "y": gy, "z": gz,
            "marker": {"size": 1.5, "color": "#f59e0b", "opacity": 0.20},
            "hoverinfo": "skip", "visible": False,
        })

    if PERICARDIUM_CSV.exists():
        px, py, pz = load_points_csv(PERICARDIUM_CSV)
        print(f"  Pericardium: {len(px):,} points")
        groups["pericardium"] = [len(traces)]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": "Pericardium",
            "x": px, "y": py, "z": pz,
            "marker": {"size": 1.0, "color": "#a3a3a3", "opacity": 0.08},
            "hoverinfo": "skip", "visible": False,
        })

    # ── Pass 1: scan diameters only (memory-efficient for 10^8+ segments) ──
    tree_diams = {}  # branch -> np.float32 array
    tree_roots = {}
    total_segs = 0
    for branch, spec in TREE_SPECS.items():
        if not spec["csv"].exists():
            print(f"  {branch}: CSV not found, skipping")
            continue
        diams = scan_tree_diameters(spec["csv"])
        tree_diams[branch] = diams
        tree_roots[branch] = find_tree_root_from_csv(spec["csv"])
        total_segs += diams.size
        print(f"  {branch}: {diams.size:,} segments, diam {diams.min():.1f}-{diams.max():.1f} \u03bcm")

    if total_segs == 0:
        print("No tree data found!"); return

    # ── Determine display-diameter threshold to cap total rendered segments ──
    min_display_diam = 0.0
    if total_segs > MAX_DISPLAY_SEGMENTS:
        # Partition-select to find the MAX_DISPLAY_SEGMENTS-th largest diameter
        # across all trees without fully sorting.
        combined = np.concatenate(list(tree_diams.values()))
        k = MAX_DISPLAY_SEGMENTS
        if k >= combined.size:
            min_display_diam = float(combined.min())
        else:
            # np.partition puts the k-th largest at position -k (O(n) avg).
            part = np.partition(combined, -k)
            min_display_diam = float(part[-k])
        del combined
        print(f"  Display threshold: {min_display_diam:.1f} μm (target ≤{MAX_DISPLAY_SEGMENTS:,} segments)")

    # ── Pass 2: load only rows with diameter >= threshold ──
    all_rows = {}
    all_diams = []
    for branch, spec in TREE_SPECS.items():
        if branch not in tree_diams:
            continue
        rows = load_tree_lines(spec["csv"], min_diameter=min_display_diam)
        all_rows[branch] = rows
        all_diams.extend(r["diameter_um"] for r in rows)
        before = tree_diams[branch].size
        if min_display_diam > 0:
            print(f"  {branch}: filtered {before:,} → {len(rows):,} segments (≥{min_display_diam:.1f} μm)")
        else:
            print(f"  {branch}: loaded {len(rows):,} segments")
    # Free pass-1 diameter arrays now that filtering is done
    tree_diams.clear()

    # ── XCAT Coronaries ──
    xcat_colors = {"LAD": "#7dd3fc", "LCX": "#fca5a5", "RCA": "#86efac"}
    if XCAT_CORONARY_CSV.exists() and tree_roots:
        xcat_split = split_xcat_coronaries(XCAT_CORONARY_CSV, tree_roots)
        for name in ["LAD", "LCX", "RCA"]:
            if name not in xcat_split: continue
            sx, sy, sz = xcat_split[name]
            if not sx: continue
            key = f"xcat_{name}"
            print(f"  XCAT {name}: {len(sx):,} points")
            groups[key] = [len(traces)]
            traces.append({
                "type": "scatter3d", "mode": "markers", "name": f"XCAT {name}",
                "x": sx, "y": sy, "z": sz,
                "marker": {"size": 2.0, "color": xcat_colors[name], "opacity": 0.35},
                "hoverinfo": "skip", "visible": False,
            })

    if not all_diams:
        print("No tree data after filtering!"); return

    # ── Grown tree mesh3d cylinders (diameter-binned for filter support) ──
    dmin = math.floor(min(all_diams))
    dmax = math.ceil(max(all_diams))
    bins = log_bins(dmin, dmax, count=12)

    total_verts = 0
    total_faces = 0

    for branch, spec in TREE_SPECS.items():
        if branch not in all_rows: continue
        rows = all_rows[branch]
        groups[branch] = []
        color_rgb = spec["color_rgb"]

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                bucket = [r for r in rows if lo <= r["diameter_um"] <= hi]
            else:
                bucket = [r for r in rows if lo <= r["diameter_um"] < hi]
            if not bucket: continue

            label = f"{branch} {int(round(lo))}-{int(round(hi))} \u03bcm"
            mesh = build_cylinder_mesh(bucket, color_rgb)
            if mesh is None: continue
            mesh["name"] = label
            mesh["meta"] = {"d_lo": lo, "d_hi": hi}

            nv = len(mesh["x"])
            nf = len(mesh["i"])
            total_verts += nv
            total_faces += nf

            groups[branch].append(len(traces))
            traces.append(mesh)

        # Hover trace (scatter3d markers, invisible)
        hover = make_hover_trace(rows, spec["color"], branch)
        groups[branch].append(len(traces))
        traces.append(hover)

        print(f"  {branch} mesh: {sum(len(t.get('x',[])) for t in traces if t.get('name','').startswith(branch) and t['type']=='mesh3d'):,} verts")

    print(f"Total mesh: {total_verts:,} vertices, {total_faces:,} faces")

    # ── Toggle buttons ──
    btn_html = ""
    btn_js = ""
    toggle_items = [
        ("domain", "Domain", True), ("chambers", "Chambers", False),
        ("great_vessels", "Great Vessels", False), ("pericardium", "Pericardium", False),
        ("xcat_LAD", "XCAT LAD", False), ("xcat_LCX", "XCAT LCX", False),
        ("xcat_RCA", "XCAT RCA", False),
        ("LAD", "LAD", True), ("LCX", "LCX", True), ("RCA", "RCA", True),
    ]
    for key, label, _ in toggle_items:
        if key in groups:
            btn_id = f"toggle-{key.lower().replace('_', '-')}"
            btn_html += f'    <button id="{btn_id}">Toggle {label}</button>\n'
            btn_js += f'    document.getElementById(\'{btn_id}\').onclick=()=>toggle(groups["{key}"]);\n'

    tree_keys = [k for k in ["LAD", "LCX", "RCA"] if k in groups]
    tree_concat_parts = ", ".join(f'(groups["{k}"]||[])' for k in tree_keys)
    tree_idx_js = f"[].concat({tree_concat_parts})" if tree_keys else "[]"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{TITLE}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body style="margin:0;font-family:Arial,sans-serif">
  <div style="padding:10px 14px">
    <h2 style="margin:0 0 6px 0">{TITLE}</h2>
    <div style="color:#555">{SUBTITLE}</div>
  </div>
  <div style="padding:0 14px 8px 14px;display:flex;gap:8px;flex-wrap:wrap">
{btn_html}    <button id="show-all">Show All</button>
  </div>
  <div style="padding:0 14px 8px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <label>Min Diameter (\\u03bcm) <input id="min-d-input" type="number" value="{int(dmin)}" min="{int(dmin)}" max="{int(dmax)}" step="1" style="width:90px"></label>
    <input id="min-d-slider" type="range" min="{int(dmin)}" max="{int(dmax)}" value="{int(dmin)}" step="1" style="width:220px">
    <label>Max Diameter (\\u03bcm) <input id="max-d-input" type="number" value="{int(dmax)}" min="{int(dmin)}" max="{int(dmax)}" step="1" style="width:90px"></label>
    <input id="max-d-slider" type="range" min="{int(dmin)}" max="{int(dmax)}" value="{int(dmax)}" step="1" style="width:220px">
    <button id="apply-range">Apply Diameter Range</button>
  </div>
  <div id="plot" style="width:100vw;height:82vh"></div>
  <script>
    const traces = {json.dumps(traces, separators=(',', ':'))};
    Plotly.newPlot('plot', traces, {{
      scene: {{
        xaxis: {{title: 'X (cm)'}},
        yaxis: {{title: 'Y (cm)'}},
        zaxis: {{title: 'Z (cm)'}},
        aspectmode: 'data'
      }},
      margin: {{l:0,r:0,b:0,t:0}}
    }}, {{displaylogo:false, responsive:true}});

    var groups = {json.dumps(groups)};
    var treeIdx = {tree_idx_js};
    var plotDiv = document.getElementById('plot');
    function rerender() {{ Plotly.react(plotDiv, plotDiv.data, plotDiv.layout); }}
    function toggle(indices) {{
      var allVis = indices.every(function(i){{ return plotDiv.data[i].visible !== false; }});
      indices.forEach(function(i){{ plotDiv.data[i].visible = !allVis; }});
      rerender();
    }}
    function showAll() {{
      plotDiv.data.forEach(function(t){{ t.visible = true; }});
      rerender();
      applyRange();
    }}
    function syncRangeInputs(src) {{
      var mi=document.getElementById('min-d-input'), ma=document.getElementById('max-d-input');
      var ms=document.getElementById('min-d-slider'), xs=document.getElementById('max-d-slider');
      if(src==='slider'){{ mi.value=ms.value; ma.value=xs.value; }}
      else{{ ms.value=mi.value; xs.value=ma.value; }}
    }}
    function applyRange() {{
      syncRangeInputs('input');
      var minD=Number(document.getElementById('min-d-input').value);
      var maxD=Number(document.getElementById('max-d-input').value);
      if(minD>maxD){{ var t=minD; minD=maxD; maxD=t; }}
      document.getElementById('min-d-input').value=minD;
      document.getElementById('max-d-input').value=maxD;
      document.getElementById('min-d-slider').value=minD;
      document.getElementById('max-d-slider').value=maxD;
      treeIdx.forEach(function(i){{
        var m=plotDiv.data[i].meta||{{}};
        var lo=Number(m.d_lo||0), hi=Number(m.d_hi||0);
        if(!lo&&!hi){{ plotDiv.data[i].visible=true; return; }}
        plotDiv.data[i].visible = !(hi<minD||lo>maxD);
      }});
      rerender();
    }}
{btn_js}    document.getElementById('show-all').onclick=function(){{showAll();}};
    document.getElementById('apply-range').onclick=function(){{applyRange();}};
    document.getElementById('min-d-slider').oninput=function(){{syncRangeInputs('slider');applyRange();}};
    document.getElementById('max-d-slider').oninput=function(){{syncRangeInputs('slider');applyRange();}};
    document.getElementById('min-d-input').onkeydown=function(e){{ if(e.key==='Enter') applyRange(); }};
    document.getElementById('max-d-input').onkeydown=function(e){{ if(e.key==='Enter') applyRange(); }};
    applyRange();
  </script>
</body>
</html>
"""
    out_path = OUT / "xcat_coronary_viewer.html"
    out_path.write_text(html)
    print(f"Written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
