import os
import re

script_path = os.path.dirname(p=os.path.realpath(__file__))


# Define the folder where the images are located
folder_path = os.path.join(script_path, "screenshots", "val")

# Regular expression pattern to match "_10.png"
pattern = re.compile(r"_10\.png$")

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if pattern.search(filename):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

print("All matching files deleted.")
