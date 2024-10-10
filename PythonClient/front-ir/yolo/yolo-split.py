import os
import re

script_path = os.path.dirname(p=os.path.realpath(__file__))

# Example function to create the .txt files for each image
def split_classes(images_folder):
    files = os.listdir(images_folder)
    for file in files[:]:
        fileName = str(file)
        if fileName.endswith(".txt"):
            #match = re.search(r'(\d+)\.txt$', fileName)
            #group = int(match.group(1))
            filePath = os.path.join(images_folder, fileName)
            with open(filePath, 'r+') as r:
                elements = r.readline().split()
                elements[0] = '0'
                r.seek(0)
                r.write(' '.join(elements))
                r.truncate()

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
