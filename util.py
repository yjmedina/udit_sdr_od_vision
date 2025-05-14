from typing import Dict, Any
def process_annotation(ann: Dict[str, Any], image: Dict[str, Any]) -> str:
    top_x, top_y, width, height = ann['bbox']

    center_x = top_x + width / 2
    norm_center_x = center_x / image['width']
    norm_width = width / image['width']

    center_y = top_y + height / 2
    norm_center_y = center_y / image['height']
    norm_height = height / image['height']

    yolo_format = f"{ann['category_id'] - 1} {norm_center_x} {norm_center_y} {norm_width} {norm_height}"
    return yolo_format