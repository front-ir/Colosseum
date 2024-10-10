import os
import re
import shutil

script_path = os.path.dirname(p=os.path.realpath(__file__))

# Example function to create the .txt files for each image
def split_classes(images_folder):
    files = os.listdir(images_folder)
    for file in files[:]:
        fileName = str(file)
        if fileName.endswith(".txt"):
            match = re.search(r'^0_xyz_.*?(\d+)\.txt$', fileName)
            group = int(match.group(1))
            print(group)
            if classes[group] > 100:
                os.remove(os.path.join(images_folder, fileName))
                os.remove(os.path.join(images_folder, fileName.replace(".txt", ".png")))
            else: classes[group] += 1

# Example usage
images_folder = os.path.join(script_path, "screenshots", "positive12")

classes = {
    500 : 0,
    400 : 0,
    300 : 0,
    200 : 0,
    100 : 0,
    50 : 0,
    20 : 0
}

split_classes(images_folder)
