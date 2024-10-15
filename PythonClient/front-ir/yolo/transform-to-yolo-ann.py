import os

script_path = os.path.dirname(p=os.path.realpath(__file__))

def convert_to_yolo_format(x_min, y_min, width, height):
    img_width = 640
    img_height = 480
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    res = f"{round(x_center_norm, 4)}", f"{round(y_center_norm, 4)}", f"{round(width_norm, 4)}", f"{round(height_norm, 4)}"
    return ' '.join(res)

# Example function to create the .txt files for each image
def create_yolo_annotations(images_folder, annotations, img_width, img_height):
    for img_info in annotations:
        img_name, num_objects, *coords = img_info.split()  # Parsing the annotation line
        img_name = img_name.replace('.png', '')  # Remove image extension for .txt name
        img_name = img_name.replace('positive/', '')  # Remove image extension for .txt name

        # Prepare path to the .txt file
        txt_path = os.path.join(images_folder, f"{img_name}.txt")

        with open(txt_path, 'w') as f:
            for i in range(int(num_objects)):
                x_min, y_min, width, height = map(int, coords[i*4:i*4+4])
                x_center_norm, y_center_norm, width_norm, height_norm = convert_to_yolo_format(
                    x_min, y_min, width, height)
                # Assuming object class '0' for this example
                f.write(f"0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

# Example usage
images_folder = os.path.join(script_path, "screenshots", "positive")
annotation_file = os.path.join(script_path, "screenshots", "positive.txt")

with open(annotation_file, 'r') as file:
    annotations = file.readlines()

img_width = 640
img_height = 480

create_yolo_annotations(images_folder, annotations, img_width, img_height)
