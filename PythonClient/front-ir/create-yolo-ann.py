import os
import re

script_path = os.path.dirname(p=os.path.realpath(__file__))

def convert_to_yolo_format(x_min, y_min, width, height, img_width=640, img_height=480):
    x_min = int(x_min)
    y_min = int(y_min)
    width = int(width)
    height = int(height)
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return round(x_center_norm, 4), round(y_center_norm, 4), round(width_norm, 4), round(height_norm, 4)

# Example function to create the .txt files for each image
def create_yolo_annotations(images_folder):
    files = os.listdir(images_folder)
    for file in files[:]:
        fileName = str(file)
        if fileName.endswith(".png"):
            match = re.search(r'(\d+)\.png$', fileName)
            group = int(match.group(1))
            txtFile = os.path.join(images_folder, fileName.replace(".png", ".txt"))
            with open(txtFile, 'w') as t:
                t.write(f"0 {yolo_coords[group][0]} {yolo_coords[group][1]} {yolo_coords[group][2]} {yolo_coords[group][3]}\n")

# Example usage
images_folder = os.path.join(script_path, "screenshots", "positive1")

norm_coords = {
    500 : "311 205 18 18",
    400 : "311 205 18 18",
    300 : "306 200 25 25",
    200 : "300 195 40 40",
    100 : "290 170 70 70",
    50 : "210 90 200 200",
    20 : "140 60 360 360"
}

yolo_coords = norm_coords
for c in yolo_coords:
    yolo_coords[c] = convert_to_yolo_format(*yolo_coords[c].split())



create_yolo_annotations(images_folder)
