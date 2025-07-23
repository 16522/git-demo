import math


def rotate_point(px, py, center_x, center_y, rot):

    px_new = center_x + (px - center_x) * math.cos(rot) + (py - center_y) * math.sin(rot)
    py_new = center_y - (px - center_x) * math.sin(rot) + (py - center_y) * math.cos(rot)
    return px_new, py_new


def rotate_boxes(boxes):
    result = []
    for box_data in boxes:
        boxes_dict = box_data.get("box", {})
        rotate_box_dict = {}

        for j, (top_left, bottom_right, rot) in boxes_dict.items():
            top_left_x, top_left_y = top_left
            bottom_right_x, bottom_right_y = bottom_right


            center_x = (top_left_x + bottom_right_x) / 2
            center_y = (top_left_y + bottom_right_y) / 2


            corners = [
                [top_left_x, top_left_y],  # 左上角
                [bottom_right_x, top_left_y],  # 右上角
                [bottom_right_x, bottom_right_y],  # 右下角
                [top_left_x, bottom_right_y]  # 左下角
            ]


            rotated_corners = []
            for corner in corners:
                px, py = corner
                px_rot, py_rot = rotate_point(px, py, center_x, center_y, rot)
                rotated_corners.append([px_rot, py_rot])


            rotate_box_dict[j] = rotated_corners


        result.append({
            "box": boxes_dict,
            "rotate_box": rotate_box_dict
        })

    return result


boxes = [
    {
        "box": {
            1: [[100, 50], [300, 200], math.radians(45)],  # 旋转 45 度
            2: [[50, 250], [250, 400], math.radians(0)]  # 无旋转
        }
    }
]

rotated_boxes = rotate_boxes(boxes)


import pprint

pprint.pprint(rotated_boxes)
