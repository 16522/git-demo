import math
import numpy as np


def point_in_polygon(point, polygon):

    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def segment_length(segment):

    (x1, y1), (x2, y2) = segment
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def is_point_on_segment(p, seg):

    (x1, y1), (x2, y2) = seg
    return min(x1, x2) <= p[0] <= max(x1, x2) and min(y1, y2) <= p[1] <= max(y1, y2)


def extended_line_intersections(segment, box):

    intersections = []
    (x1, y1), (x2, y2) = segment


    edges = [
        [(box[0][0], box[0][1]), (box[1][0], box[1][1])],
        [(box[1][0], box[1][1]), (box[2][0], box[2][1])],
        [(box[2][0], box[2][1]), (box[3][0], box[3][1])],
        [(box[3][0], box[3][1]), (box[0][0], box[0][1])]
    ]

    for edge in edges:
        intersection = line_intersection([(x1, y1), (x2, y2)], edge)
        if intersection and is_point_on_segment(intersection, edge):
            intersections.append(intersection)


    valid_intersections = []
    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):

            valid_intersections.append(intersections[i])
            valid_intersections.append(intersections[j])
            break
        if valid_intersections:
            break


    if len(valid_intersections) == 2:
        return [valid_intersections[0], valid_intersections[1]]
    elif len(valid_intersections) > 2:
        return [valid_intersections[0], valid_intersections[1]]
    else:
        return []


def find_perfect_lines_and_intersections(boxes, lines):
    perfect_lines_intersections = []

    for box_idx, box in enumerate(boxes, start=1):
        group_intersections = []
        for line in lines:
            if segment_length(line) > 30:
                (x1, y1), (x2, y2) = line
                if point_in_polygon((x1, y1), box) and point_in_polygon((x2, y2), box):
                    intersections = extended_line_intersections(line, box)
                    if intersections:
                        group_intersections.append(intersections)
        perfect_lines_intersections.append(group_intersections)

    return perfect_lines_intersections



boxes = [
    [(0, 0), (100, 0), (100, 100), (0, 100)],
    [(150, 150), (250, 150), (250, 250), (150, 250)]
]

lines = [
    [(10, 10), (90, 90)],
    [(160, 160), (240, 240)],
    [(50, 50), (60, 50)],
    [(300, 300), (310, 310)]
]


perfect_lines_intersections = find_perfect_lines_and_intersections(boxes, lines)

# 输出结果
print(perfect_lines_intersections)
