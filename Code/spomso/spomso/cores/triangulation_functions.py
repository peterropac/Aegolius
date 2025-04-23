# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def check_convex(v3s: np.ndarray) -> np.ndarray:
    """
    Checks if 3 vertices are convex.

    Args:
        v3s: Coordinates of the vertices with shape (D >= 2, 3).
    Returns:
        True if they are convex, otherwise False.
    """

    v1 = v3s[:2, 1] - v3s[:2, 0]
    v2 = v3s[:2, 2] - v3s[:2, 1]
    c = np.cross(v1, v2)
    return c > 0


def check_convex_all(vs: np.ndarray) -> np.ndarray:
    """
    Checks if a polygon is convex.

    Args:
        vs: Coordinates of the vertices with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Convexity of each vertex.
    """

    v1s = vs[:2, 1:-1] - vs[:2, :-2]
    v2s = vs[:2, 2:] - vs[:2, 1:-1]
    c = np.cross(v1s.T, v2s.T)
    c_last = np.cross(vs[:2, 0] - vs[:2, -1], vs[:2, 1] - vs[:2, 0])
    c = np.concatenate((c, [c_last]), axis=0)
    return c


def is_inside_triangle(vs: np.ndarray, v3s: np.ndarray) -> np.ndarray:
    """
    Checks if points are inside a triangle.

    Args:
        vs: Coordinates of the points to check with shape (D >= 2, N), where N is the number of points.
        v3s: Coordinates of the vertices of the triangle with shape (D >= 2, 3).
    Returns:
        True if they are inside the triangle and not one of the vertices, otherwise False.
    """

    d = (v3s[1, 1] - v3s[1, 2]) * (v3s[0, 0] - v3s[0, 2]) + (v3s[0, 2] - v3s[0, 1]) * (v3s[1, 0] - v3s[1, 2])
    lambda1 = ((v3s[1, 1] - v3s[1, 2]) * (vs[0, :] - v3s[0, 2]) + (v3s[0, 2] - v3s[0, 1]) * (vs[1, :] - v3s[1, 2])) / d
    lambda2 = ((v3s[1, 2] - v3s[1, 0]) * (vs[0, :] - v3s[0, 2]) + (v3s[0, 0] - v3s[0, 2]) * (vs[1, :] - v3s[1, 2])) / d
    lambda3 = 1 - lambda1 - lambda2

    c = lambda1 * lambda2 * lambda3
    cc = (c >= 0) * (lambda1 < 1) * (lambda2 < 1) * (lambda3 < 1)
    return cc


def is_ear(vs: np.ndarray, v3s: np.ndarray) -> bool:
    """
    Checks if a triangle is an ear (Ear-Clipping triangulation method).

    Args:
        vs: Coordinates of the vertices of the polygon with shape (D >= 2, N), where N is the number of points.
        v3s: Coordinates of the vertices of the triangle with shape (D >= 2, 3).
    Returns:
        True if the triangle is an ear, otherwise False.
    """

    convex_condition = check_convex(v3s)
    inside_condition = np.any(is_inside_triangle(vs, v3s))
    return convex_condition * (~inside_condition)


def triangulate(vs: np.ndarray) -> np.ndarray:
    """
    Triangulates a polygon defined by its vertices using the Ear-Clipping triangulation method.

    Args:
        vs: Coordinates of the vertices of the polygon with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Array of triangles with shape (D >= 2, 3, N - 2), where N is the number of vertices.
    """

    points = vs.copy()
    out = np.zeros((3, 3, points.shape[1] - 2))

    c, i = 0, 0
    while points.shape[1] > 3:
        ixs_t = [i - 1, i, np.mod(i + 1, points.shape[1])]
        if is_ear(points, points[:, ixs_t]):
            out[:, :, c] = points[:, ixs_t]
            points = np.delete(points, i, axis=1)
            c, i = c + 1, 0
        else:
            i += 1
    out[:, :, -1] = points

    return out


def check_intersection(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> tuple:
    """
    Checks if there is an intersection between two lines defined by 4 points.

    Args:
        v1: Coordinate of the first point with shape (3,), defining the start of the first line.
        v2: Coordinate of the second point with shape (3,), defining the end of the first line.
        v3: Coordinate of the third point with shape (3,), defining the start of the second line.
        v4: Coordinate of the fourth point with shape (3,), defining the end of the second line.

    Returns: True if there is an intersection, and the coordinate of the intersection.
    """

    b = (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0])
    t1 = ((v3[0] - v1[0]) * (v4[1] - v3[1]) - (v3[1] - v1[1]) * (v4[0] - v3[0])) / b
    t2 = ((v3[0] - v1[0]) * (v2[1] - v1[1]) - (v3[1] - v1[1]) * (v2[0] - v1[0])) / b
    return (t1 > 0) * (t1 < 1) * (t2 > 0) * (t2 < 1), np.asarray(
        (v1[0] + (v2[0] - v1[0]) * t1, v1[1] + (v2[1] - v1[1]) * t1))


def check_intersection_all(vs: np.ndarray) -> tuple:
    """
    Checks if there is an intersection between all pairs of segments in a polygon.

    Args:
        vs: Coordinates of the vertices of a polygon, with shape (3, N), where N is the number of vertices.

    Returns: The indices of the segments where there is an intersection,
            the indices of the intersected segments,
            the coordinates of the intersections.
    """

    ixs = np.linspace(0, vs.shape[1] + 1, vs.shape[1] + 1, endpoint=False, dtype=int)
    ixs[-1] = 0
    mask = np.zeros(vs.shape[1] + 1, dtype=bool)
    vs = vs[:, ixs]

    six, eix, cs = [], [], []
    for c in range(vs.shape[1] - 2):
        m1 = mask.copy()
        m2 = mask.copy()
        m1[c + 1:-1] = True
        m2[c + 2:] = True

        print("m1", m1)
        print("m2", m2)

        x, y = vs[:2, m1], vs[:2, m2]

        print("x", x)
        print("y", y)

        v, u = y - x, vs[:2, c + 1] - vs[:2, c]
        v, u = v / np.linalg.norm(v, axis=0), u / np.linalg.norm(u)

        dot = v[0] * u[0] + v[1] * u[1]
        dot_mask = np.abs(dot) == 1
        x, y = x[:, ~dot_mask], y[:, ~dot_mask]
        m1[m1], m2[m2] = ~dot_mask, ~dot_mask
        print("m1", m1)
        print("m2", m2)

        print("d", dot_mask, "u", u, "v", v, dot)
        print("x", x)
        print("y", y)

        if x.size == 0 or y.size == 0:
            continue

        out = check_intersection(vs[:2, c], vs[:2, c + 1], x, y)

        print(out)

        if np.any(out[0]):
            six.append(c)
            eix.append(ixs[m1][out[0]])
            cs.append(out[1][:, out[0]])

        # print(six, eix, cs)

    return six, eix, cs


def create_points_sets(vs, idata):

    print("idata", idata)

    leading_ixs = idata[0]
    cross_ixs = np.asarray(idata[1])
    coors = np.asarray(idata[2])
    n_intersections = cross_ixs.size
    intersection_ixs = np.linspace(0, n_intersections, n_intersections, endpoint=False, dtype=int) + vs.shape[1]
    print("cross_ixs", cross_ixs)
    c_cross_ixs = np.concatenate(cross_ixs, axis=0)
    print("cross_ixs", c_cross_ixs, n_intersections)

    groups = []
    c = 0
    for i in range(len(leading_ixs)):
        for j in range(cross_ixs[i].size):
            g1 = [leading_ixs[i], int(intersection_ixs[c]), int(np.mod(cross_ixs[i][j] + 1, vs.shape[1]))]
            g2 = [int(cross_ixs[i][j]), int(intersection_ixs[c]), int(np.mod(leading_ixs[i] + 1, vs.shape[1]))]
            print(c, g1, g2)
            groups.append(g1)
            groups.append(g2)
            c += 1

    print("groups", groups)
    i = 0
    for v in range(len(groups)):
        group_reformat = 0
        group = groups[i]
        group = np.squeeze(group).tolist()
        print("group", group, len(groups))

        for g in range(len(groups)):
            if g == i:
                continue
            if group[0] == groups[g][-1] and group[-1] == groups[g][0]:
                group = np.concatenate((group, groups[g][1:-1])).tolist()
                group_reformat = 1
                break

            c4 = group[1] - groups[g][1] == -1 # 1 < 2
            c5 = int(np.mod(group[0] + 1, vs.shape[1])) == groups[g][2]
            c10 = (group[0] < group[2]) + (groups[g][0] < groups[g][2])

            c3 = (group[1] == intersection_ixs[0]) * (groups[g][1] == intersection_ixs[-1])
            c6 = int(np.mod(groups[g][0] + 1, vs.shape[1])) == group[2]
            c7 = len(group) == 3 and len(groups[g]) == 3
            c8 = group[0] in leading_ixs
            c9 = (group[2] not in intersection_ixs) and (group[0] not in intersection_ixs)
            c11 = group[1] is not groups[g][1]
            print("c1", c3, c4, c5, c6)


            if c4*c5*c7*c10:
                    group = [groups[g][0], groups[g][1], group[1], group[-1]]
                    if group[0] == group[-1]:
                        group = group[:-1]
                    group_reformat = 1
                    break

            if c3*c6*c7*c8*c9*c11:
                    group = [group[0], group[1], groups[g][1], groups[g][2]]
                    if group[0] == group[-1]:
                        group = group[:-1]
                    group_reformat = 1
                    break

        if group_reformat:
            groups[i] = group
            groups = [groups[k] for k in range(len(groups)) if not k == g]
            print("groups re", groups)
            continue

        for j in range(vs.shape[1]):

            if (group[-1] not in c_cross_ixs) and (group[-1] not in leading_ixs):
                if np.mod(group[-1] + 1, vs.shape[1]) == group[0]:
                    break
                elif group[-1] >= vs.shape[1] - 1:
                    pass
                else:
                    group.append(group[-1] + 1)

            if (group[0] not in c_cross_ixs) and (group[0] - 1 not in leading_ixs):
                if np.mod(group[-1] + 1, vs.shape[1]) == group[0]:
                    break
                elif group[0] == 0:
                    pass
                elif group[0] >= vs.shape[1]:
                    pass
                else:
                    group.append(group[0] - 1)

        groups[i] = group

        i += 1
        if i == len(groups):
            break

    # remove double groups

    print("groups", groups)
    print("coors", coors, np.squeeze(coors), coors.shape)
    coors_c = np.moveaxis(coors, 1, 0)
    coors_c = np.reshape(coors_c, (2, n_intersections))
    print("coors_c", coors_c, coors_c.shape)
    evs = np.concatenate((vs[:2, :], coors_c), axis=1)
    evs = np.concatenate((evs, np.zeros((1, evs.shape[1]))), axis=0)
    vs_groups = [evs[:, groups[i]] for i in range(n_intersections + 1)]

    return vs_groups


def interior_triangle(co: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Partitions the coordinate system into the interior (-1) and exterior (+1) of a triangle defined by its vertices.

    Args:
        co: Point cloud of coordinates with shape (D >= 2, N);
            D - number of dimensions;
            N - number of points in the point cloud.
        points: Coordinates of the vertices with shape (D >= 2, 3).
    Returns:
        Map of the interior and exterior of the triangle (N,).
    """

    zero = np.average(points, axis=1)
    c1 = points[:, 0] / 2 + points[:, 1] / 2 - zero
    c2 = points[:, 1] / 2 + points[:, 2] / 2 - zero
    c3 = points[:, 2] / 2 + points[:, 0] / 2 - zero

    v1 = points[:, 1] - points[:, 0]
    v2 = points[:, 2] - points[:, 1]
    v3 = points[:, 0] - points[:, 2]

    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    v3n = v3 / np.linalg.norm(v3)

    n1 = v1n.copy()
    n1[0] = -v1n[1]
    n1[1] = v1n[0]
    n1 = n1 - 2 * n1 * (n1.dot(c1) < 0)

    n2 = v2n.copy()
    n2[0] = -v2n[1]
    n2[1] = v2n[0]
    n2 = n2 - 2 * n2 * (n2.dot(c2) < 0)

    n3 = v3n.copy()
    n3[0] = -v3n[1]
    n3[1] = v3n[0]
    n3 = n3 - 2 * n3 * (n3.dot(c3) < 0)

    normals = np.asarray([n1, n2, n3])

    sp = -np.ones(co.shape[1])
    for k in range(3):
        m = np.sign(np.dot(co.T - points[:, k], normals[k]))
        sp[:] = np.maximum(sp, m)

    return sp


def interior_convex(co: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Partitions the coordinate system into the interior (-1) and exterior (+1) of a convex polygon.

    Args:
        co: Point cloud of coordinates with shape (D >= 2, N);
            D - number of dimensions;
            N - number of points in the point cloud.
        points: Coordinates of the vertices with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Map of the interior and exterior of the polygon (N,).
    """

    zero = np.average(points, axis=1)

    normals = points.copy()
    for i in range(points.shape[1]):
        k = (i+1)%points.shape[1]
        ci = points[:, i] / 2 + points[:, k] / 2 - zero
        vi = points[:, k] - points[:, i]
        vin = vi / np.linalg.norm(vi)
        ni = vin.copy()
        ni[0] = -vin[1]
        ni[1] = vin[0]
        ni = ni * (1 - 2 * (ni.dot(ci) < 0))
        normals[:, i] = ni

    sp = -np.ones(co.shape[1])
    for k in range(points.shape[1]):
        m = np.sign(np.dot(co.T - points[:, k], normals[:, k]))
        sp[:] = np.maximum(sp, m)

    return sp


def interior_polygon(co: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Partitions the coordinate system into the interior (-1) and exterior (+1) of a polygon.

    Args:
        co: Point cloud of coordinates with shape (D >= 2, N);
            D - number of dimensions;
            N - number of points in the point cloud.
        points: Coordinates of the vertices with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Map of the interior and exterior of the polygon (N,).
    """

    interior = np.ones(co.shape[1])

    convexity = check_convex_all(points)
    if np.all(convexity >= 0):
        sp = interior_convex(co, points)
        interior[sp <= 0] = -1
    elif np.all(convexity <= 0):
        points[:, :] = points[:, ::-1]
        sp = interior_convex(co, points)
        interior[sp <= 0] = -1
    else:
        intersection_data = check_intersection_all(points)
        if intersection_data[0]:
            points_sets = create_points_sets(points, intersection_data)
            print("points sets", points_sets)
            for new_points in points_sets:
                print(new_points, "new_points")
                new_interior = interior_polygon(co, new_points)
                interior[new_interior <= 0] = -1
        else:
            if np.count_nonzero(convexity >= 0) < points.shape[0]//2:
                points = points[:, ::-1]
            triangles = triangulate(points)
            for i in range(points.shape[1] - 2):
                sp = interior_convex(co, triangles[:, :, i])
                interior[sp <= 0] = -1

    return interior






