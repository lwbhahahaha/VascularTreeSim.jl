"""
    NURBS parsing — organ-agnostic.

Parses XCAT-format .nrb files into NurbsSurface structs, evaluates B-spline
surfaces, and samples surface point clouds.
"""

struct XCATNurbsSurface
    name::String
    m::Int
    n::Int
    u_knots::Vector{Float64}
    v_knots::Vector{Float64}
    control_points::Array{SVector{3, Float64}, 2}
end

function xcat_object_dict(surfaces::AbstractVector{XCATNurbsSurface})
    Dict(surface.name => surface for surface in surfaces)
end

function xcat_uv_counts(surface::XCATNurbsSurface)
    n_v, n_u = size(surface.control_points)
    return n_u, n_v
end

function xcat_degrees(surface::XCATNurbsSurface)
    n_u, n_v = xcat_uv_counts(surface)
    p_u = length(surface.u_knots) - n_u - 1
    p_v = length(surface.v_knots) - n_v - 1
    return p_u, p_v
end

function _xcat_basis(i::Int, p::Int, u::Float64, knots::Vector{Float64}, n_basis::Int)
    if p == 0
        left = knots[i]
        right = knots[i + 1]
        return ((left <= u < right) || (u == knots[end] && i == n_basis && left <= u <= right)) ? 1.0 : 0.0
    end
    left_den = knots[i + p] - knots[i]
    right_den = knots[i + p + 1] - knots[i + 1]
    left_term = 0.0
    right_term = 0.0
    if left_den > 0
        left_term = ((u - knots[i]) / left_den) * _xcat_basis(i, p - 1, u, knots, n_basis)
    end
    if right_den > 0
        right_term = ((knots[i + p + 1] - u) / right_den) * _xcat_basis(i + 1, p - 1, u, knots, n_basis)
    end
    return left_term + right_term
end

function xcat_surface_point(surface::XCATNurbsSurface, u::Real, v::Real)
    u_f = clamp(Float64(u), 0.0, 1.0)
    v_f = clamp(Float64(v), 0.0, 1.0)
    n_u, n_v = xcat_uv_counts(surface)
    p_u, p_v = xcat_degrees(surface)
    basis_u = [_xcat_basis(i, p_u, u_f, surface.u_knots, n_u) for i in 1:n_u]
    basis_v = [_xcat_basis(j, p_v, v_f, surface.v_knots, n_v) for j in 1:n_v]
    acc = SVector(0.0, 0.0, 0.0)
    @inbounds for j in 1:n_v
        b_v = basis_v[j]
        b_v == 0.0 && continue
        for i in 1:n_u
            b = b_v * basis_u[i]
            b == 0.0 && continue
            acc += b * surface.control_points[j, i]
        end
    end
    return acc
end

function xcat_surface_normal(surface::XCATNurbsSurface, u::Real, v::Real; δ::Float64=1e-4)
    u0 = max(0.0, Float64(u) - δ)
    u1 = min(1.0, Float64(u) + δ)
    v0 = max(0.0, Float64(v) - δ)
    v1 = min(1.0, Float64(v) + δ)
    pu0 = xcat_surface_point(surface, u0, v)
    pu1 = xcat_surface_point(surface, u1, v)
    pv0 = xcat_surface_point(surface, u, v0)
    pv1 = xcat_surface_point(surface, u, v1)
    tu = pu1 - pu0
    tv = pv1 - pv0
    n = cross(tu, tv)
    n_norm = norm(n)
    n_norm > 0 || return SVector(0.0, 0.0, 1.0)
    return n / n_norm
end

function xcat_sample_surface(surface::XCATNurbsSurface; n_u::Int=80, n_v::Int=80, orient_outward::Bool=false)
    u_values = collect(range(0.0, 1.0; length=n_u))
    v_values = collect(range(0.0, 1.0; length=n_v))
    points = Array{SVector{3, Float64}, 2}(undef, n_v, n_u)
    normals = Array{SVector{3, Float64}, 2}(undef, n_v, n_u)
    for j in 1:n_v, i in 1:n_u
        u = u_values[i]
        v = v_values[j]
        points[j, i] = xcat_surface_point(surface, u, v)
        normals[j, i] = xcat_surface_normal(surface, u, v)
    end
    if orient_outward
        center = let acc = SVector(0.0, 0.0, 0.0)
            for p in points
                acc += p
            end
            acc / length(points)
        end
        for j in 1:n_v, i in 1:n_u
            if dot(normals[j, i], points[j, i] - center) < 0
                normals[j, i] = -normals[j, i]
            end
        end
    end
    return points, normals, u_values, v_values
end

function _parse_xcat_control_point(line::AbstractString)
    parts = split(strip(line), ',')
    return SVector(parse(Float64, parts[1]), parse(Float64, parts[2]), parse(Float64, parts[3]))
end

function _parse_xcat_knot_vector(lines::Vector{String}, idx::Int)
    knots = Float64[]
    while idx <= length(lines)
        token = strip(lines[idx])
        (token == "Control Points" || token == "U Knot Vector" || token == "V Knot Vector") && break
        isempty(token) && break
        push!(knots, parse(Float64, token))
        idx += 1
    end
    return knots, idx
end

function _reshape_xcat_control_points(points::Vector{SVector{3, Float64}}, m::Int, n::Int)
    expected = m * n
    length(points) == expected || error("Expected $expected control points, found $(length(points))")
    grid = Array{SVector{3, Float64}, 2}(undef, m, n)
    k = 1
    for u in 1:m, v in 1:n
        grid[u, v] = points[k]
        k += 1
    end
    return grid
end

function parse_xcat_nrb(path::AbstractString)
    lines = readlines(path)
    surfaces = XCATNurbsSurface[]
    i = 1
    while i <= length(lines)
        name = strip(lines[i])
        if isempty(name)
            i += 1
            continue
        end
        if i + 2 > length(lines) || !occursin(":M", strip(lines[i + 1])) || !occursin(":N", strip(lines[i + 2]))
            i += 1
            continue
        end
        m = parse(Int, strip(split(strip(lines[i + 1]), ':')[1]))
        n = parse(Int, strip(split(strip(lines[i + 2]), ':')[1]))
        i += 3
        strip(lines[i]) == "U Knot Vector" || error("Expected U Knot Vector after $name")
        i += 1
        u_knots, i = _parse_xcat_knot_vector(lines, i)
        strip(lines[i]) == "V Knot Vector" || error("Expected V Knot Vector after U knots for $name")
        i += 1
        v_knots, i = _parse_xcat_knot_vector(lines, i)
        strip(lines[i]) == "Control Points" || error("Expected Control Points for $name")
        i += 1
        n_points = m * n
        points = Vector{SVector{3, Float64}}(undef, n_points)
        for k in 1:n_points
            points[k] = _parse_xcat_control_point(lines[i])
            i += 1
        end
        control_points = _reshape_xcat_control_points(points, m, n)
        push!(surfaces, XCATNurbsSurface(name, m, n, u_knots, v_knots, control_points))
    end
    return surfaces
end
