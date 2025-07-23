import math


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


def find_perfect_lines(boxes, lines):
    perfect_lines = {i: [] for i in range(1, len(boxes) + 1)}

    for box_idx, box in enumerate(boxes, start=1):
        for line in lines:
            if segment_length(line) > 30:
                (x1, y1), (x2, y2) = line
                if point_in_polygon((x1, y1), box) and point_in_polygon((x2, y2), box):

                    perfect_lines[box_idx].append(line)

    return perfect_lines



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


perfect_lines = find_perfect_lines(boxes, lines)


result = {
    "lines": {i: [] for i in range(1, len(boxes) + 1)},  # 这里只是为了格式一致，实际不使用
    "perfect_line": perfect_lines
}

import json

print(json.dumps(result, indent=2, ensure_ascii=False))
