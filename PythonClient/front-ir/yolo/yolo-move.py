import os
import re
import shutil

script_path = os.path.dirname(p=os.path.realpath(__file__))

# Example function to create the .txt files for each image
def split_classes(images_folder):
    files = os.listdir(images_folder)
    for file in files[:]:
        fileName = str(file)
        match = re.search(r'^0_xyz_.*', fileName)
        if match:
            filePath = os.path.join(images_folder, fileName)
            newFilePath = filePath.replace("positive1", "positive2")
            shutil.copy(filePath, newFilePath)

# Example usage
images_folder = os.path.join(script_path, "screenshots", "positive1")

classes = {
    500 : 0,
    400 : 1,
    300 : 2,
    200 : 3,
    100 : 4,
    50 : 5,
    20 : 6
}

split_classes(images_folder)
